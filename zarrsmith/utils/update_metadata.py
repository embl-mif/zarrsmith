"""Write metadata updates into the NGFF attributes of an existing image group.

Used by :meth:`zarrsmith.microscope_manager.zarr_manager.ZarrManager.update_metadata`
with ``in_place=True``. Only the NGFF entries tied to the updated keys are rewritten:
dataset paths, OMERO display windows, the downsampling record and any other key of
the group are kept as they are.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable

import numpy as np
import zarr
from ome_zarr.writer import add_metadata

from .axes import normalize_axes, normalize_data_type
from .ngff import (
    _NGFF_ATTRIBUTE_KEYS,
    _axes_entries,
    _build_coordinate_transformations,
    _build_omero_metadata,
    _drop_ngff_keys,
    _extra_group_attributes,
    _get_group_ome_attrs,
    _infer_ngff_version,
    _ngff_format,
    _write_extra_attributes,
)
from .colors import parse_color
from .display_window import update_display_windows

#: Updated keys that change the ``coordinateTransformations`` of every level.
_TRANSFORM_KEYS = frozenset({"scales", "translations", "time_increment"})
#: Updated keys that change the ``axes`` entries (their units).
_AXES_KEYS = frozenset({"units", "time_increment_unit"})
#: Updated keys that change the OMERO channels.
_CHANNEL_KEYS = frozenset({"channel_names", "channel_colors"})


def check_metadata_writable(
    group: zarr.Group,
    metadata: dict[str, Any],
    keys: Iterable[str],
    *,
    label: str = "dataset",
) -> None:
    """Raise if ``metadata`` (already updated in memory) cannot be written to ``group``.

    Checked before anything is written, so a refused update leaves the store as it was.
    """
    ngff_version = _infer_ngff_version(group)
    zarr_format = group.metadata.zarr_format
    if (str(ngff_version), zarr_format) not in {("0.4", 2), ("0.5", 3)}:
        raise ValueError(
            f"{label} is NGFF {ngff_version} on zarr v{zarr_format}. zarrsmith updates metadata "
            "in place only in NGFF 0.4 (zarr v2) or 0.5 (zarr v3) images."
        )
    multiscales = _get_group_ome_attrs(group).get("multiscales") or []
    if not multiscales or not multiscales[0].get("datasets"):
        raise ValueError(f"{label} has no multiscales metadata to update.")
    n_levels = len(multiscales[0]["datasets"])
    keys = set(keys)
    if keys & _TRANSFORM_KEYS and len(metadata.get("scales") or []) != n_levels:
        raise ValueError(
            f"{label} has {n_levels} level(s) on disk but {len(metadata.get('scales') or [])} "
            "scale(s) in memory."
        )
    if "data_type" in keys and normalize_data_type(metadata.get("data_type")) == "label":
        dtype = group[str(multiscales[0]["datasets"][0]["path"])].dtype
        if not np.issubdtype(np.dtype(dtype), np.integer):
            raise ValueError(f"{label} cannot become a label image: its dtype {dtype} is not an integer.")
    extra = metadata.get("extra_attributes") if "extra_attributes" in keys else None
    clash = sorted(set(extra or {}) & _NGFF_ATTRIBUTE_KEYS)
    if clash:
        raise ValueError(f"Extra attributes cannot use NGFF metadata keys: {', '.join(clash)}.")


def write_metadata_updates(
    group: zarr.Group,
    metadata: dict[str, Any],
    keys: Iterable[str],
    *,
    label_source: dict[str, Any] | None = None,
) -> None:
    """Write the entries ``keys`` of ``metadata`` into the NGFF attributes of ``group``.

    ``metadata`` is the normalized ZarrSmith metadata of the image stored in ``group``,
    with the updates already applied (see
    :func:`zarrsmith.utils.dataset_ops.apply_metadata_updates`); call
    :func:`check_metadata_writable` first. ``label_source`` is the ``image-label``
    ``source`` entry written when ``data_type`` turns the image into a label, e.g.
    ``{"source": {"image": "../../"}}`` for a label under ``<image>/labels/<name>``.

    - ``scales``, ``translations``, ``time_increment``: every level's
      ``coordinateTransformations`` (a multiscale-level one is folded into them);
    - ``units``, ``time_increment_unit``: the ``axes`` entries;
    - ``channel_names``, ``channel_colors``: label and color of the OMERO channels;
    - ``channel_windows``: the display window (``start``/``end``) of the OMERO channels,
      estimated from the coarsest stored level for a channel given ``None``;
    - ``data_type``: the ``image-label`` block, or OMERO channels for intensity data;
    - ``extra_attributes``: the group attributes that are not NGFF metadata.
    """
    keys = set(keys)
    ngff_version = _infer_ngff_version(group)
    fmt = _ngff_format(ngff_version)
    image_meta = copy.deepcopy(_get_group_ome_attrs(group))
    multiscales_all = image_meta["multiscales"]
    multiscales = multiscales_all[0]
    datasets = multiscales["datasets"]
    level0 = group[str(datasets[0]["path"])]
    axes = normalize_axes(metadata["axes"], ndim=level0.ndim)
    data_type = normalize_data_type(metadata.get("data_type"))

    entry: dict[str, Any] = {}
    if keys & (_TRANSFORM_KEYS | _AXES_KEYS):
        if keys & _TRANSFORM_KEYS:
            transforms = _build_coordinate_transformations(
                axes=axes,
                scales=metadata["scales"],
                time_increment=metadata.get("time_increment"),
                translations=metadata.get("translations"),
            )
            for dataset, ct in zip(datasets, transforms):
                dataset["coordinateTransformations"] = ct
            # The levels now hold the full physical transform.
            multiscales.pop("coordinateTransformations", None)
        if keys & _AXES_KEYS:
            multiscales["axes"] = _axes_entries(axes, metadata)
        entry["multiscales"] = multiscales_all

    if data_type == "intensity" and keys & (_CHANNEL_KEYS | {"data_type"}):
        omero = image_meta.get("omero")
        if not isinstance(omero, dict) or not omero.get("channels"):
            omero = _build_omero_metadata(level0, axes, metadata)
        else:
            names = list(metadata.get("channel_names") or [])
            colors = list(metadata.get("channel_colors") or [])
            for i, channel in enumerate(omero["channels"]):
                if "channel_names" in keys and i < len(names):
                    channel["label"] = names[i]
                if "channel_colors" in keys and i < len(colors):
                    channel["color"] = parse_color(colors[i])
        entry["omero"] = omero

    if "data_type" in keys:
        # Legacy zarrsmith key, read before the NGFF blocks: drop it so it does not go stale.
        _drop_ngff_keys(group, fmt, ("data_type",))
        if "data_type" in group.attrs:
            del group.attrs["data_type"]
        if data_type == "label":
            _drop_ngff_keys(group, fmt, ("omero",))
            entry["image-label"] = {"version": fmt.version, **(label_source or {})}
        else:
            _drop_ngff_keys(group, fmt, ("image-label",))

    if entry:
        if fmt.zarr_format == 3:
            entry["version"] = fmt.version
        add_metadata(group, entry, fmt=fmt)

    # The windows given (None: estimated from the stored data, as on a write), also for an
    # image just turned to intensity data, whose OMERO channels are new.
    if data_type == "intensity" and keys & {"channel_windows", "data_type"}:
        metadata["channel_windows"] = update_display_windows(
            group, axes, metadata.get("channel_windows") if "channel_windows" in keys else None)

    if "extra_attributes" in keys:
        extra = metadata.get("extra_attributes") or {}
        for key in set(_extra_group_attributes(group)) - set(extra):
            del group.attrs[key]
        _write_extra_attributes(group, extra)
