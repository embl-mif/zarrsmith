"""Dataset-level operations shared by ``MicroscopeManager`` and ``ZarrManager``.

Each function works on a single ``(data, metadata)`` pair (a list of dask arrays
plus the ZarrSmith metadata dictionary) so that the single-dataset managers and the
multi-dataset ``ZarrManager`` (raw + groups + labels) run exactly the same
validation code.
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from numbers import Real
from typing import Any, Callable, Optional

import dask.array as da

from .axes import (
    SPATIAL_AXIS_SET,
    UNRECORDED_SCALE_KEY,
    index_list_from_selection,
    normalize_axes,
    normalize_data_type,
)

logger = logging.getLogger("zarrsmith")


def _axes_of(metadata: dict[str, Any]) -> str:
    return str(metadata.get("axes", "")).lower()


def _n_spatial(metadata: dict[str, Any]) -> int:
    return sum(1 for ax in _axes_of(metadata) if ax in SPATIAL_AXIS_SET)


def channel_count(data: Sequence[da.Array], metadata: dict[str, Any]) -> Optional[int]:
    """Number of channels of the base level, or ``None`` without a channel axis."""
    axes = _axes_of(metadata)
    return int(data[0].shape[axes.index("c")]) if "c" in axes else None


# ---------------------------------------------------------------------------
# Metadata updates
# ---------------------------------------------------------------------------

def _check_channel_values(key, value, *, data, metadata, label):
    n = channel_count(data, metadata)
    if n is None:
        return None, f"{label} has no channel axis. Skipping '{key}'."
    if len(value) != n:
        return None, (
            f"Length of '{key}' ({len(value)}) does not match number of channels "
            f"({n}) in {label}. Skipping."
        )
    if key == "channel_colors":
        from .ngff import parse_channel_color

        value = [parse_channel_color(v, i) for i, v in enumerate(value)]
    return value, None


def _check_channel_windows(key, value, *, data, metadata, label):
    """One ``(start, end)`` display window per channel (``None``: from the data); ``None``
    alone sets every channel from the data. Intensity data only."""
    if normalize_data_type(metadata.get("data_type")) == "label":
        # worded as the "no channel axis" skips, which ZarrManager does not warn about
        return None, f"{label} is label data and has no channel axis for display windows. Skipping '{key}'."
    n = channel_count(data, metadata) or 1  # without a c axis: one channel
    if value is None:
        return None, None
    if not isinstance(value, (list, tuple)) or len(value) != n:
        return None, (
            f"'{key}' needs one (start, end) or None per channel ({n}) in {label}, "
            f"got {value!r}. Skipping."
        )
    windows = []
    for window in value:
        if window is None:
            windows.append(None)
            continue
        try:
            start, end = (float(v) for v in window)
        except (TypeError, ValueError):
            raise ValueError(f"Each '{key}' entry must be (start, end) or None, got {window!r}.") from None
        if not end > start:
            raise ValueError(f"A display window must have end > start, got {window!r}.")
        windows.append((start, end))
    return windows, None


def _check_scales(key, value, *, data, metadata, label):
    if not isinstance(value, list) or len(value) != len(data):
        raise ValueError(
            f"'scales' must be a list with one entry per pyramid level "
            f"({len(data)}) in {label}."
        )
    for scale in value:
        if not isinstance(scale, (list, tuple)) or len(scale) != _n_spatial(metadata):
            raise ValueError(f"Each scale entry must match the spatial axes of {label}.")
    return value, None


def _check_translations(key, value, *, data, metadata, label):
    """A position (dict or numbers, for every level) or one per level; ``None`` clears it."""
    from .translation import normalize_translations

    try:
        return normalize_translations(value, _axes_of(metadata), len(data)), None
    except (TypeError, ValueError) as e:
        raise type(e)(f"Invalid 'translations' for {label}: {e}") from e


def _check_extra_attributes(key, value, **_):
    if value is not None and not isinstance(value, dict):
        raise TypeError("'extra_attributes' must be a dict or None.")
    return (dict(value) if value else None), None


def _check_time_increment(key, value, **_):
    if value is not None and (
        not isinstance(value, Real) or isinstance(value, bool) or value <= 0
    ):
        raise ValueError("'time_increment' must be a positive number or None.")
    return value, None


def _check_time_unit(key, value, **_):
    if value is not None and not isinstance(value, str):
        raise ValueError("'time_increment_unit' must be a string or None.")
    return value, None


def _check_units(key, value, *, metadata, label, **_):
    if not isinstance(value, (tuple, list)):
        raise TypeError("'units' must be a tuple or list.")
    if len(value) != _n_spatial(metadata):
        raise ValueError(f"'units' must match the spatial axes of {label}.")
    return value, None


def _check_data_type(key, value, **_):
    return normalize_data_type(value), None


#: key -> validator. A validator returns ``(value, skip_reason)`` and raises on
#: invalid input; a non-``None`` ``skip_reason`` makes the update warn and skip.
_VALIDATORS: dict[str, Callable[..., tuple[Any, Optional[str]]]] = {
    "channel_names": _check_channel_values,
    "channel_colors": _check_channel_values,
    "channel_windows": _check_channel_windows,
    "scales": _check_scales,
    "translations": _check_translations,
    "extra_attributes": _check_extra_attributes,
    "time_increment": _check_time_increment,
    "time_increment_unit": _check_time_unit,
    "units": _check_units,
    "data_type": _check_data_type,
}

#: Metadata keys that :func:`apply_metadata_updates` accepts.
VALID_UPDATE_KEYS: tuple[str, ...] = tuple(_VALIDATORS)


def known_updates(updates: dict[str, Any]) -> dict[str, Any]:
    """Return only the supported entries of ``updates``, warning about the rest."""
    known = {}
    for key, value in updates.items():
        if key in _VALIDATORS:
            known[key] = value
        else:
            warnings.warn(f"Unsupported or unknown metadata key: '{key}'", stacklevel=3)
    return known


def apply_metadata_updates(
    data: Sequence[da.Array],
    metadata: dict[str, Any],
    updates: dict[str, Any],
    *,
    label: str = "dataset",
    warn_no_channel_axis: bool = True,
) -> list[str]:
    """Validate ``updates`` against ``(data, metadata)`` and apply them in place.

    Invalid values raise ``ValueError``/``TypeError``. Channel updates that do
    not fit the dataset are skipped with a warning (silently when
    ``warn_no_channel_axis`` is false and the dataset has no channel axis).
    ``updates`` must already be filtered with :func:`known_updates`.
    Returns the keys that were applied.
    """
    applied = []
    for key, value in updates.items():
        value, skip = _VALIDATORS[key](
            key, value, data=data, metadata=metadata, label=label
        )
        if skip is not None:
            if warn_no_channel_axis or "no channel axis" not in skip:
                warnings.warn(skip, stacklevel=3)
            continue
        if key == "data_type":
            metadata["is_label"] = value == "label"
        if key == "scales":  # given by the user: every voxel size is now known
            metadata.pop(UNRECORDED_SCALE_KEY, None)
        metadata[key] = value
        applied.append(key)
        logger.info("Updated metadata entry '%s' of %s", key, label)
    return applied


# ---------------------------------------------------------------------------
# Channel reordering
# ---------------------------------------------------------------------------

def reorder_channels(
    data: Sequence[da.Array],
    metadata: dict[str, Any],
    new_order: Sequence[int],
    *,
    label: str = "dataset",
    skip_if_no_channel_axis: bool = False,
) -> list[da.Array]:
    """Return ``data`` with channels permuted; ``metadata`` is updated in place."""
    if not data:
        raise ValueError("No data loaded.")
    axes = _axes_of(metadata)
    if "c" not in axes:
        if skip_if_no_channel_axis:
            return list(data)
        raise ValueError(f"{label} has no channel axis to reorder.")

    c_dim = axes.index("c")
    n = data[0].shape[c_dim]
    new_order = [int(i) for i in new_order]
    if sorted(new_order) != list(range(n)):
        raise ValueError(f"new_order must be a permutation of 0..{n - 1} for {label}.")

    reordered = []
    for level in data:
        slicer = [slice(None)] * level.ndim
        slicer[c_dim] = new_order
        reordered.append(level[tuple(slicer)])

    for key in ("channel_names", "channel_colors", "channel_windows"):
        if metadata.get(key) is not None and len(metadata[key]) == n:
            metadata[key] = [metadata[key][i] for i in new_order]
    metadata["size"] = [tuple(level.shape) for level in reordered]
    metadata["chunksize"] = [tuple(level.chunksize) for level in reordered]
    return reordered


# ---------------------------------------------------------------------------
# Subsetting
# ---------------------------------------------------------------------------

def pyramid_like(metadata: dict[str, Any], start_level: int = 0) -> tuple[int, tuple[int, int, int]]:
    """The pyramid of an image from ``start_level`` on: ``(levels, (z, y, x) factor)``.

    ``levels`` counts ``start_level`` and every level below it; the factor is the one between
    ``start_level`` and the next level, from their voxel sizes (1 on an axis the image does not
    have). A pyramid built with them over data matching ``start_level`` (e.g. labels of an image
    segmented at a coarser level) has level ``k`` matching level ``start_level + k`` of the image.
    With a single level from ``start_level`` on, the factor is ``(2, 2, 2)`` (unused).
    """
    scales = metadata.get("scales") or []
    if not 0 <= start_level < len(scales):
        raise ValueError(f"start_level {start_level} is out of range: the image has {len(scales)} level(s).")
    levels = len(scales) - start_level
    if levels == 1:
        return 1, (2, 2, 2)
    spatial = [ax for ax in _axes_of(metadata) if ax in SPATIAL_AXIS_SET]
    ratio = {ax: scales[start_level + 1][i] / scales[start_level][i] for i, ax in enumerate(spatial)}
    return levels, tuple(max(1, int(round(ratio.get(ax, 1.0)))) for ax in "zyx")


def infer_downscale_factors(metadata: dict[str, Any]) -> int | tuple[int, ...]:
    """Infer the per-spatial-axis downscale factors between pyramid levels 0 and 1.

    Returns ``2`` for single-level data. Isotropic factors are returned as an
    ``int``; anisotropic ones (e.g. ``(1, 2, 2)``) as a tuple in spatial-axis
    order, so the original pyramid layout is rebuilt faithfully.
    """
    sizes = metadata.get("size", [])
    if len(sizes) < 2:
        return 2
    axes = _axes_of(metadata)
    factors = []
    for i, ax in enumerate(axes):
        if ax not in SPATIAL_AXIS_SET:
            continue
        s0, s1 = sizes[0][i], sizes[1][i]
        factors.append(max(1, int(round(s0 / s1))) if s1 else 2)
    if not factors:
        return 2
    return factors[0] if len(set(factors)) == 1 else tuple(factors)


def subset_levels(
    data: Sequence[da.Array],
    metadata: dict[str, Any],
    *,
    T=None, C=None, Z=None, Y=None, X=None,
    rebuild_pyramid: bool = True,
    label: str = "dataset",
) -> tuple[list[da.Array], dict[str, Any]]:
    """Subset the base level by axis selections and optionally rebuild the pyramid.

    Selections on axes the dataset does not have are ignored. Indices are
    validated against the dataset shape.
    """
    from .pyramid import build_pyramid
    from .subset import subset_dask_array, subset_metadata

    if not data:
        raise ValueError("No data loaded.")

    axes = _axes_of(metadata)
    normalize_axes(axes, ndim=data[0].ndim)
    shape = data[0].shape
    selections = {
        name: (sel if name.lower() in axes else None)
        for name, sel in zip("TCZYX", (T, C, Z, Y, X))
    }
    for name, sel in selections.items():
        if sel is None:
            continue
        size = shape[axes.index(name.lower())]
        indices = index_list_from_selection(sel, size)
        if not indices:  # e.g. slice(30, 100) of a single timepoint: an empty axis cannot be written
            raise ValueError(
                f"{name}={sel!r} selects nothing on axis '{name.lower()}' of {label}, "
                f"which has {size} element(s) (indices 0 to {size - 1})."
            )
        if min(indices) < 0 or max(indices) >= size:
            raise ValueError(
                f"Index for axis '{name.lower()}' out of range in {label}. Axis size is {size}."
            )

    num_levels = len(data)
    factors = infer_downscale_factors(metadata)

    new_data = [subset_dask_array(data[0], axes=axes, **selections)]
    new_metadata = subset_metadata(metadata, **selections)
    new_metadata["chunksize"] = [tuple(arr.chunksize) for arr in new_data]

    if rebuild_pyramid and num_levels > 1:
        new_data, new_metadata = build_pyramid(
            new_data, new_metadata, num_levels=num_levels, downscale_factor=factors
        )
    return new_data, new_metadata
