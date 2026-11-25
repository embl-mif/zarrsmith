"""Display window of the channels of an image (OMERO ``window`` ``start``/``end``).

Viewers open a channel with ``start``/``end`` as its contrast. The writer's default is
the whole range of the data type, so an image opens nearly black: after writing,
:func:`write_display_windows` stores the 0.5th-99.9th percentile of the values instead,
estimated by :func:`sample_display_windows` from the coarsest level (a small read).
``min``/``max`` (the range of the data type) are left as they are.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import zarr

from .axes import SPATIAL_AXIS_SET, normalize_axes

#: Percentiles of the values used as the display window (blank, zero voxels ignored).
DISPLAY_PERCENTILES: Tuple[float, float] = (0.5, 99.9)
#: Voxels read per channel to estimate the window: about 8 MB of uint16.
MAX_SAMPLE_VOXELS = 1 << 22
#: Timepoints sampled per channel, spread over the ``t`` axis.
MAX_SAMPLE_TIMEPOINTS = 3

Window = Tuple[float, float]


def window_from_values(values: np.ndarray, percentiles: Sequence[float] = DISPLAY_PERCENTILES) -> Window:
    """``(start, end)``: the ``percentiles`` of the non-zero, finite ``values``; ``(0, 1)`` if none."""
    values = np.asarray(values).ravel()
    if values.dtype.kind == "f":
        values = values[np.isfinite(values)]
    values = values[values != 0]
    if values.size == 0:
        return 0.0, 1.0
    low, high = (float(v) for v in np.percentile(values, percentiles))
    return low, (high if high > low else low + 1.0)


def _central_box(shape: dict, max_voxels: int, n_timepoints: int, chunks: Optional[dict] = None) -> dict:
    """``{axis: slice}`` of a box at the centre of the spatial axes, of about
    ``max_voxels / n_timepoints`` voxels, shrunk by the same factor on every axis.

    With ``chunks`` (``{axis: chunk size}``) the box is made of whole chunks, at least one
    per axis: a read decompresses every chunk it touches, so a box across chunk borders
    would cost several times its size. The read is then at most about ``max_voxels``,
    or one chunk per axis when a chunk is bigger.
    """
    spatial = [ax for ax in shape if ax in SPATIAL_AXIS_SET]
    total = n_timepoints * math.prod(shape[ax] for ax in spatial)
    budget = max_voxels / max(1, n_timepoints)
    factor = min(1.0, (budget * n_timepoints / total) ** (1.0 / len(spatial))) if spatial and total else 1.0
    sizes = {ax: max(1, int(shape[ax] * factor)) for ax in spatial}
    step = {ax: int((chunks or {}).get(ax) or 0) for ax in spatial}
    step = {ax: (c if 0 < c < shape[ax] else 0) for ax, c in step.items()}
    # whole chunks on the chunked axes; one budget for the whole box (flat chunks such as
    # 1 x 4096 x 4096 would otherwise take a chunk per plane of the box)
    count = {ax: max(1, sizes[ax] // step[ax]) for ax in spatial if step[ax]}

    def volume():
        return math.prod(count[ax] * step[ax] if ax in count else sizes[ax] for ax in spatial)

    while volume() > budget and any(k > 1 for k in count.values()):
        widest = max((ax for ax in count if count[ax] > 1), key=lambda ax: count[ax] * step[ax])
        count[widest] -= 1
    box = {}
    for ax in spatial:
        n = shape[ax]
        if ax in count:
            k, chunk = count[ax], step[ax]
            first = max(0, min((n // chunk) // 2 - k // 2, -(-n // chunk) - k))  # around the central chunk
            box[ax] = slice(first * chunk, min(n, (first + k) * chunk))
        else:
            start = (n - sizes[ax]) // 2
            box[ax] = slice(start, start + sizes[ax])
    return box


def sample_display_windows(
    level: Any,
    axes: Sequence[str] | str,
    *,
    percentiles: Sequence[float] = DISPLAY_PERCENTILES,
    max_voxels: int = MAX_SAMPLE_VOXELS,
) -> List[Window]:
    """Display window of every channel of ``level`` (a zarr or dask array), one per channel
    (a single one without a ``c`` axis).

    Reads about ``max_voxels`` voxels per channel (whole chunks, at least one): the centre
    of the spatial axes, at up to :data:`MAX_SAMPLE_TIMEPOINTS` timepoints spread over the
    ``t`` axis. The cost does not grow with the image, so a single full-resolution level is
    fine too; pass the coarsest pyramid level, usually read whole, for a better estimate.
    """
    axes = normalize_axes(axes, ndim=len(level.shape))
    shape = dict(zip(axes, (int(n) for n in level.shape)))
    timepoints = [0]
    if "t" in shape:
        timepoints = sorted({int(round(v)) for v in np.linspace(0, shape["t"] - 1,
                                                                 min(MAX_SAMPLE_TIMEPOINTS, shape["t"]))})
    # read chunks: zarr's chunks (the inner chunks of a sharded array), or dask's
    chunk_shape = getattr(level, "chunksize", None) or getattr(level, "chunks", None)
    chunks = None
    if chunk_shape is not None and len(chunk_shape) == len(axes) and all(isinstance(c, int) for c in chunk_shape):
        chunks = dict(zip(axes, chunk_shape))
    box = _central_box(shape, max_voxels, len(timepoints), chunks)
    windows = []
    for c in range(shape.get("c", 1)):
        values = []
        for t in timepoints:
            index = tuple(t if ax == "t" else c if ax == "c" else box[ax] for ax in axes)
            values.append(np.asarray(level[index]).ravel())
        windows.append(window_from_values(np.concatenate(values), percentiles))
    return windows


def _omero(group: zarr.Group) -> Tuple[dict, Optional[dict], bool]:
    """``(attributes, omero block or None, nested in "ome")`` of ``group``."""
    attrs = group.attrs.asdict()
    nested = isinstance(attrs.get("ome"), dict)  # NGFF 0.5 keeps everything under "ome"
    return attrs, (attrs["ome"] if nested else attrs).get("omero"), nested


def write_display_windows(group: zarr.Group, windows: Sequence[Optional[Window]]) -> None:
    """Store ``windows`` as the ``start``/``end`` of the OMERO channels of ``group``, in order
    (a ``None`` entry leaves that channel as it is).

    A group without OMERO channels (e.g. labels) is left as it is.
    """
    attrs, omero, nested = _omero(group)
    if not omero or not omero.get("channels"):
        return
    for channel, window in zip(omero["channels"], windows):
        if window is None:
            continue
        start, end = window
        channel.setdefault("window", {}).update({"start": float(start), "end": float(end)})
    if nested:
        group.attrs["ome"] = attrs["ome"]
    else:
        group.attrs["omero"] = omero


def read_display_windows(group: zarr.Group) -> List[Window]:
    """``(start, end)`` of every OMERO channel of ``group`` that has a window (``[]`` without)."""
    _, omero, _ = _omero(group)
    return [(float(c["window"]["start"]), float(c["window"]["end"]))
            for c in (omero or {}).get("channels", [])
            if isinstance(c.get("window"), dict) and {"start", "end"} <= set(c["window"])]


def recorded_windows(group: zarr.Group, dtype) -> Optional[List[Optional[Window]]]:
    """The display windows of ``group`` someone chose: ``None`` for a channel whose window
    is the whole range of the data type (the writer's default) or is missing, and ``None``
    altogether when no channel has one."""
    _, omero, _ = _omero(group)
    channels = (omero or {}).get("channels") or []
    full = None
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(np.dtype(dtype))
        full = (float(info.min), float(info.max))
    out: List[Optional[Window]] = []
    for c in channels:
        w = c.get("window") if isinstance(c, dict) else None
        if not isinstance(w, dict) or w.get("start") is None or w.get("end") is None:
            out.append(None)
            continue
        window = (float(w["start"]), float(w["end"]))
        bounds = (w.get("min"), w.get("max"))
        default = window == full or (None not in bounds and window == tuple(float(v) for v in bounds))
        out.append(None if default or not window[1] > window[0] else window)
    return out if any(w is not None for w in out) else None


def update_display_windows(
    group: zarr.Group,
    axes: Sequence[str] | str,
    windows: Optional[Sequence[Optional[Window]]] = None,
    *,
    from_data: bool = True,
) -> List[Optional[Window]]:
    """Write the display windows of ``group``: ``windows`` (one per channel), and for the
    channels without one (``None``, or every channel when ``windows`` is ``None``) the
    estimate from the coarsest stored level, unless ``from_data`` is false.

    Returns the windows written, ``None`` for a channel left as it is (``[]`` for a group
    without OMERO channels or levels).
    """
    _, omero, _ = _omero(group)
    attrs = group.attrs.asdict()
    multiscales = (attrs["ome"] if isinstance(attrs.get("ome"), dict) else attrs).get("multiscales") or []
    if not omero or not omero.get("channels") or not multiscales or not multiscales[0].get("datasets"):
        return []
    n = len(omero["channels"])
    chosen = list(windows) if windows is not None and len(windows) == n else [None] * n
    if from_data and any(w is None for w in chosen):
        coarsest = group[str(multiscales[0]["datasets"][-1]["path"])]
        estimated = sample_display_windows(coarsest, axes)
        chosen = [w if w is not None else (estimated[i] if i < len(estimated) else None)
                  for i, w in enumerate(chosen)]
    write_display_windows(group, chosen)
    return chosen


def widen_contrast_ranges(layers) -> None:
    """Put the contrast slider of napari image ``layers`` of an integer type over the whole
    range of the type, keeping their contrast (napari limits the slider to the contrast
    a layer is opened with, e.g. a stored display window)."""
    for layer in layers:
        if not hasattr(layer, "contrast_limits_range"):
            continue
        data = layer.data[0] if getattr(layer, "multiscale", False) else layer.data
        dtype = np.dtype(getattr(data, "dtype", np.float32))
        if np.issubdtype(dtype, np.integer):
            limits = tuple(layer.contrast_limits)
            info = np.iinfo(dtype)
            layer.contrast_limits_range = (float(info.min), float(info.max))
            layer.contrast_limits = limits


def apply_display_windows(layers, windows: Sequence[Optional[Window]]) -> None:
    """Give napari image ``layers`` (one per channel, in order) the contrast ``windows``
    (``None``: left as it is), with the slider over the whole range of an integer data
    type, so the contrast can still be widened (napari would otherwise limit the slider
    to the window)."""
    for layer, window in zip(layers, windows):
        if window is None or not hasattr(layer, "contrast_limits"):
            continue
        widen_contrast_ranges([layer])
        layer.contrast_limits = tuple(float(v) for v in window)
