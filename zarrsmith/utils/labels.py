"""Derive images from an OME-Zarr image: projections, thresholded labels, label filtering,
and channels masked by labels (``zarrsmith project``, ``threshold``, ``filter-labels``, ``mask``).

Every function takes opened images (``ZarrManager``) and returns an ``ArrayManager`` with
the new image and its metadata, to write with ``to_zarr``; nothing is written here. The
pyramid of the input is kept level for level, so the outputs line up with it.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import dask.array as da
import numpy as np

SPATIAL = ("z", "y", "x")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fit_shape(array, shape: Sequence[int]):
    """Crop or zero-pad ``array`` (numpy or dask) at the high end of each axis to ``shape``.

    Pyramids built separately from the same image can round a coarse level a pixel or
    two differently; the origin stays where it is."""
    shape = tuple(int(n) for n in shape)
    if tuple(array.shape) == shape:
        return array
    if array.ndim != len(shape):
        raise ValueError(f"Rank mismatch: array shape {tuple(array.shape)} vs target {shape}")
    lib = da if isinstance(array, da.Array) else np
    array = array[tuple(slice(0, n) for n in shape)]
    pad = [(0, n - m) for n, m in zip(shape, array.shape)]
    if any(after for _, after in pad):
        array = lib.pad(array, pad, mode="constant", constant_values=0)
    return array


def resolve_channel(spec, names: Sequence[str] | None, n_channels: int) -> int:
    """A channel given by index (``2``, ``"2"``) or by name, as an index."""
    names = list(names or [])
    text = str(spec).strip()
    if text.lstrip("-").isdigit():
        index = int(text)
    elif text in names:
        index = names.index(text)
    else:
        raise ValueError(f"Unknown channel {spec!r}: give an index (0-{n_channels - 1}) or one of {names}.")
    if not 0 <= index < n_channels:
        raise ValueError(f"Channel {index} is out of range for {n_channels} channel(s).")
    return index


def _axes(metadata: Mapping[str, Any]) -> str:
    return "".join(metadata["axes"]).lower()


def _drop_axis_per_level(values, index: int):
    return [tuple(v for i, v in enumerate(level) if i != index) for level in values]


def _drop_spatial(metadata: dict, axis: str) -> None:
    """Remove a spatial axis from the per-spatial-axis metadata (scales, units, translations)."""
    spatial = [ax for ax in _axes(metadata) if ax in SPATIAL]
    if axis not in spatial:
        return
    i = spatial.index(axis)
    if metadata.get("scales"):
        metadata["scales"] = _drop_axis_per_level(metadata["scales"], i)
    if metadata.get("units") and len(metadata["units"]) == len(spatial):
        metadata["units"] = tuple(u for j, u in enumerate(metadata["units"]) if j != i)
    if metadata.get("translations"):
        metadata["translations"] = [{k: v for k, v in level.items() if k != axis} if level else level
                                    for level in metadata["translations"]]
    from .axes import UNRECORDED_SCALE_KEY

    if metadata.get(UNRECORDED_SCALE_KEY):
        metadata[UNRECORDED_SCALE_KEY] = tuple(a for a in metadata[UNRECORDED_SCALE_KEY] if a != axis)


def _array_manager(levels, metadata):
    from ..microscope_manager import ArrayManager

    metadata["size"] = [tuple(int(n) for n in level.shape) for level in levels]
    metadata["chunksize"] = [tuple(int(c[0]) if isinstance(c, tuple) else int(c) for c in level.chunks)
                             for level in levels]
    metadata["dtype"] = str(levels[0].dtype)
    return ArrayManager(levels, metadata)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project(image, *, axis: str = "z", method: str = "max"):
    """Project ``image`` (a ``ZarrManager``) along a spatial ``axis`` (``max``, ``sum`` or
    ``mean``), on every pyramid level, lazily. The axis is removed from the output and from
    its scales, units and translations; the output is intensity data (``sum`` gives
    ``float64`` for integer inputs, ``mean`` ``float32``)."""
    axis = axis.lower()
    if method not in ("max", "sum", "mean"):
        raise ValueError(f"method must be max, sum or mean, got {method!r}.")
    metadata = copy.deepcopy(image.metadata)
    axes = _axes(metadata)
    if axis not in axes or axis not in SPATIAL:
        raise ValueError(f"The image has no spatial axis {axis!r} to project along (axes {axes}).")
    k = axes.index(axis)
    levels = []
    for level in image.data:
        level = da.asarray(level)
        if method == "max":
            levels.append(level.max(axis=k))
        elif method == "sum":
            levels.append(level.sum(axis=k))
        else:
            levels.append(level.mean(axis=k, dtype=np.float32).astype(np.float32))
    _drop_spatial(metadata, axis)
    metadata["axes"] = axes.replace(axis, "")
    metadata["data_type"] = "intensity"
    return _array_manager(levels, metadata)


# ---------------------------------------------------------------------------
# Threshold -> labels
# ---------------------------------------------------------------------------

def _plane(data, axes: str, fixed: Mapping[str, int]):
    index = tuple(fixed[ax] if ax in fixed else slice(None) for ax in axes)
    return np.asarray(data[index])


def threshold_labels(image, channel, *, level: int = 0, method: str = "otsu", factor: float = 1.0,
                     levels: str = "all", dtype="uint32"):
    """Labels of the objects of one channel: a global threshold (``otsu``) times ``factor`` at
    pyramid ``level``, then connected components (``skimage.measure.label``), per timepoint.

    The labelling is made once, at ``level``; the other pyramid levels are that labelling
    resized with nearest neighbour to their exact shapes, so an object keeps its id on every
    level. ``levels="all"`` keeps every level of the image (the finer ones upsampled), so the
    labels line up level for level with it; ``"from_level"`` keeps ``level`` and the coarser
    ones. The output has no channel axis and is label data."""
    from skimage.filters import threshold_otsu
    from skimage.measure import label as connected_components
    from skimage.transform import resize

    if method != "otsu":
        raise ValueError(f"Only the otsu threshold is available, got {method!r}.")
    if levels not in ("all", "from_level"):
        raise ValueError(f"levels must be 'all' or 'from_level', got {levels!r}.")
    metadata = copy.deepcopy(image.metadata)
    axes = _axes(metadata)
    if "c" not in axes:
        raise ValueError(f"The image has no channel axis: axes {axes}.")
    if not 0 <= level < len(image.data):
        raise ValueError(f"level {level} is out of range for an image with {len(image.data)} level(s).")
    data = image.data[level]
    c = resolve_channel(channel, metadata.get("channel_names"), int(data.shape[axes.index("c")]))
    out_axes = axes.replace("c", "")
    n_t = int(data.shape[axes.index("t")]) if "t" in axes else None

    frames = []
    for t in range(n_t) if n_t is not None else [None]:
        fixed = {"c": c, **({"t": t} if t is not None else {})}
        plane = _plane(data, axes, fixed)
        mask = plane > threshold_otsu(plane) * float(factor)
        frames.append(connected_components(mask))
    canonical = np.stack(frames, axis=out_axes.index("t")) if n_t is not None else frames[0]
    dtype = np.dtype(dtype)
    if canonical.size and canonical.max() > np.iinfo(dtype).max:
        raise ValueError(f"{int(canonical.max())} objects do not fit in {dtype}.")
    canonical = canonical.astype(dtype)

    c_index = axes.index("c")
    keep = range(len(image.data)) if levels == "all" else range(level, len(image.data))
    out_levels = []
    for k in keep:
        shape = tuple(n for i, n in enumerate(image.data[k].shape) if i != c_index)
        chunks = tuple(int(ch[0]) for i, ch in enumerate(da.asarray(image.data[k]).chunks) if i != c_index)
        if k == level:
            array = canonical
        else:
            array = _resize_nearest(canonical, shape, out_axes, resize)
        out_levels.append(da.from_array(array, chunks=chunks))

    for key in ("size", "chunksize"):
        metadata[key] = _drop_axis_per_level(metadata[key], c_index)
    if levels == "from_level":
        for key in ("scales", "translations"):
            if metadata.get(key):
                metadata[key] = list(metadata[key])[level:]
    for key in ("channel_names", "channel_colors", "display_windows"):
        metadata.pop(key, None)
    metadata["axes"] = out_axes
    metadata["data_type"] = "label"
    return _array_manager(out_levels, metadata)


def _resize_nearest(labels: np.ndarray, shape, axes: str, resize) -> np.ndarray:
    """Nearest-neighbour resize, one timepoint at a time (t is never resized)."""
    if labels.shape == tuple(shape):
        return labels
    kwargs = dict(order=0, preserve_range=True, anti_aliasing=False)
    if "t" not in axes:
        return resize(labels, shape, **kwargs).astype(labels.dtype)
    ti = axes.index("t")
    frame_shape = tuple(n for i, n in enumerate(shape) if i != ti)
    frames = [resize(np.take(labels, t, axis=ti), frame_shape, **kwargs).astype(labels.dtype)
              for t in range(labels.shape[ti])]
    return np.stack(frames, axis=ti)


# ---------------------------------------------------------------------------
# Filter labels by their measurements
# ---------------------------------------------------------------------------

def parse_limits(text: str | None) -> dict[str, tuple[float | None, float | None]]:
    """``"solidity:0.8:;mean_intensity_ch1:100:"`` -> ``{column: (min, max)}`` (a bound may be empty)."""
    limits: dict[str, tuple[float | None, float | None]] = {}
    for token in (text or "").split(";"):
        token = token.strip()
        if not token:
            continue
        parts = [p.strip() for p in token.split(":")]
        if len(parts) != 3 or not parts[0]:
            raise ValueError(f"Invalid filter '{token}'. Expected 'property:min:max' (either bound may be empty).")
        lo = float(parts[1]) if parts[1] else None
        hi = float(parts[2]) if parts[2] else None
        if lo is not None and hi is not None and hi <= lo:
            raise ValueError(f"Invalid filter '{token}': max must be greater than min.")
        limits[parts[0]] = (lo, hi)
    return limits


def select_rows(table, limits: Mapping[str, tuple[float | None, float | None]]):
    """The rows of ``table`` within every ``{column: (min, max)}`` limit (bounds inclusive)."""
    keep = np.ones(len(table), dtype=bool)
    for column, (lo, hi) in limits.items():
        if column not in table.columns:
            raise ValueError(f"Unknown column '{column}'. Available columns: {sorted(table.columns)}")
        if lo is not None:
            keep &= (table[column] >= lo).to_numpy()
        if hi is not None:
            keep &= (table[column] <= hi).to_numpy()
    return table[keep]


def _remap_block(block: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    out = np.zeros_like(block)
    inside = (block >= 0) & (block < lookup.size)
    out[inside] = lookup[block[inside]]
    return out


def filter_labels(labels, table, limits: Mapping[str, tuple[float | None, float | None]] | None = None, *,
                  relabel: bool = True):
    """Keep the labels whose row of ``table`` (a DataFrame with a ``label`` column, and a
    ``timepoint`` column for labels with a ``t`` axis, e.g. from :func:`zarrsmith.utils.measure.measure_labels`)
    is within ``limits``; every other id becomes 0, on every pyramid level, lazily.

    With ``relabel`` the kept objects are renumbered 1..N (per timepoint), the same way on
    every level. Returns ``(ArrayManager, kept rows)``; the kept rows get the new ``label``
    and the old one as ``original_label`` (next to it)."""
    if "label" not in table.columns:
        raise ValueError(f"The table has no 'label' column. Available columns: {sorted(table.columns)}")
    kept = select_rows(table, limits or {})
    metadata = copy.deepcopy(labels.metadata)
    axes = _axes(metadata)
    has_t = "t" in axes
    if has_t and "timepoint" not in kept.columns:
        raise ValueError(f"The labels have a time axis (axes {axes}) but the table has no 'timepoint' column.")
    groups = ({int(t): sorted(int(v) for v in g["label"]) for t, g in kept.groupby("timepoint")} if has_t
              else {None: sorted(int(v) for v in kept["label"])})
    maps = {t: {old: (new if relabel else old) for new, old in enumerate(ids, start=1)} for t, ids in groups.items()}

    def lookup(mapping, dtype):
        size = (max(mapping) + 1) if mapping else 1
        table_ = np.zeros(size, dtype=dtype)
        for old, new in mapping.items():
            table_[old] = new
        return table_

    out_levels = []
    for level in labels.data:
        level = da.asarray(level)
        if not has_t:
            out_levels.append(level.map_blocks(_remap_block, lookup(maps[None], level.dtype), dtype=level.dtype))
            continue
        ti = axes.index("t")
        frames = []
        for t in range(level.shape[ti]):
            frame = da.take(level, [t], axis=ti)
            frames.append(frame.map_blocks(_remap_block, lookup(maps.get(t, {}), level.dtype), dtype=level.dtype))
        out_levels.append(da.concatenate(frames, axis=ti).rechunk(level.chunks))

    kept = kept.copy()
    position = list(kept.columns).index("label")
    if has_t:
        new = [maps[int(t)][int(v)] for t, v in zip(kept["timepoint"], kept["label"])]
    else:
        new = [maps[None][int(v)] for v in kept["label"]]
    kept["original_label"] = kept["label"]
    kept["label"] = new
    columns = [c for c in kept.columns if c not in ("label", "original_label")]
    columns[position:position] = ["label", "original_label"]
    return _array_manager(out_levels, metadata), kept[columns]


# ---------------------------------------------------------------------------
# Mask a channel with labels
# ---------------------------------------------------------------------------

def mask_channel(image, labels, channel, *, suffix: str = "_clean", append: bool = True):
    """Zero one channel of ``image`` outside the objects of ``labels``, on every pyramid level,
    lazily. The two must share their pyramid (same number of levels); a level whose shape was
    rounded differently is cropped or zero-padded to the image's. ``append`` keeps every
    channel and adds the masked one (named ``<name><suffix>``, with the channel's colour);
    otherwise the output holds only the masked channel."""
    metadata = copy.deepcopy(image.metadata)
    axes = _axes(metadata)
    if "c" not in axes:
        raise ValueError(f"The image has no channel axis: axes {axes}.")
    c_axis = axes.index("c")
    n_channels = int(image.data[0].shape[c_axis])
    c = resolve_channel(channel, metadata.get("channel_names"), n_channels)
    label_axes = _axes(labels.metadata)
    if len(image.data) != len(labels.data):
        raise ValueError(f"The image has {len(image.data)} pyramid level(s) but the labels have {len(labels.data)}: "
                         "they must share their pyramid (e.g. labels from `zarrsmith threshold` of this image).")
    if ("t" in axes) != ("t" in label_axes):
        raise ValueError(f"The image and the labels disagree on a time axis (image {axes}, labels {label_axes}).")
    if "c" in label_axes:
        raise ValueError(f"The labels have a channel axis (axes {label_axes}).")

    out_levels = []
    for level, label_level in zip(image.data, labels.data):
        level = da.asarray(level)
        channel_data = da.take(level, [c], axis=c_axis)  # keeps a c axis of 1
        target = tuple(n for i, n in enumerate(channel_data.shape) if i != c_axis)
        lab = fit_shape(da.asarray(label_level), target)
        lab = da.expand_dims(lab, axis=c_axis)
        masked = da.where(lab != 0, channel_data, 0).astype(level.dtype)
        masked = masked.rechunk(tuple(1 if i == c_axis else ch for i, ch in enumerate(level.chunks)))
        out_levels.append(da.concatenate([level, masked], axis=c_axis) if append else masked)

    names = list(metadata.get("channel_names") or [])
    name = f"{names[c] if c < len(names) else 'channel'}{suffix}"
    colors = list(metadata.get("channel_colors") or [])
    color = colors[c] if c < len(colors) else None
    if append:
        if names:
            metadata["channel_names"] = names + [name]
        if colors:
            metadata["channel_colors"] = colors + [color or "FFFFFF"]
    else:
        metadata["channel_names"] = [name]
        if colors:
            metadata["channel_colors"] = [color or "FFFFFF"]
    metadata.pop("display_windows", None)
    metadata["data_type"] = "intensity"
    return _array_manager(out_levels, metadata)
