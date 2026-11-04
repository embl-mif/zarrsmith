"""Fuse the tiles of a :class:`~zarrsmith.mosaic_manager.MosaicManager` into one OME-Zarr image.

Fusion runs in two steps:

1. **Level 0** is computed region by region into a temporary zarr store whose
   chunks are the regions: for each region only the tile voxels that fall in it
   are read, downsampled and merged. Regions are sized to cover whole source
   chunks in ``y``/``x`` (so compressed tile chunks are decoded about once) and cut
   into ``z`` slabs of bounded size, so memory stays bounded whatever the depth and
   no dask graph over all tiles is ever built. One whole region (read, merge,
   write) is the unit of parallel work; regions never share a chunk, so workers
   write without locks. Sigmoid weights are only evaluated where tiles or groups
   overlap.
2. The fused level 0 (already downsampled, so small compared with the raw tiles)
   is written to the final OME-Zarr with :meth:`MicroscopeManager.to_zarr`, so
   pyramid levels, chunks, sharding, compression, zarr format and singleton axes
   behave exactly as for any other ZarrSmith conversion (``zarrsmith convert``).

Tiles of several timepoints give one output with a ``t`` axis. The geometry (output
grid, tile offsets, transitions between tiles and groups) comes from the tiles of one
**reference timepoint**, and every timepoint is fused with it, tile by tile. Stage
positions reported at other timepoints are not used, so a position stays at the same
place throughout the output.

With ``reference_timepoint="all"`` every timepoint uses its own stage positions instead:
the output grid covers the tiles of all timepoints, and the tile offsets and transitions
are computed for each timepoint.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import zarr

from ..utils.metadata import ChannelInfo, build_metadata
from ..utils.ngff import ZarrWriteConfig
from ..utils.output_path import output_refusal
from .fields import check_single_field, fields_of, multiple_fields_message
from .tile import Tile, series_keys

logger = logging.getLogger("zarrsmith")

BLENDS = ("sigmoid", "max", "sum", "midline")
PROJECTIONS = (None, "max", "sum")

#: Default steepness of the ``sigmoid`` transition (see :func:`fuse_tiles`).
SIGMOID_STEEPNESS = 4.0

#: Number of full-resolution z planes read at once (bounds memory per worker).
_SLAB_PLANES = 64

#: Minimum size (y, x) of the regions fused at once, in output voxels. Regions grow to
#: cover whole source chunks (see :func:`_region_shape`) and are split along z so that
#: each holds at most ``_REGION_VOXELS`` voxels.
_REGION_3D = (256, 256)
_REGION_2D = (1024, 1024)
#: Largest region side (y or x) chosen to match the source chunks, in output voxels.
_REGION_MAX = 2048
#: Voxels per 3D region (32 MB per float32 canvas): bounds the memory of each worker.
_REGION_VOXELS = 2**23
#: Codec of the temporary level 0 with ``temp_compression=True`` (zarr's default, zstd).
_TEMP_COMPRESSORS = "auto"

#: Percentiles of the fused values used as the display window written to the output.
DISPLAY_PERCENTILES = (0.5, 99.9)
#: Values kept from every fused region to estimate the display window (no extra reads).
_WINDOW_SAMPLES = 65536

#: ``reference_timepoint`` value that places every timepoint at its own stage positions.
ALL_TIMEPOINTS = "all"

#: Root attribute recording the timepoints and the reference of a fused timelapse.
PROVENANCE_KEY = "zarrsmith_fusion"

#: ``to_zarr`` options that fusion sets itself.
_RESERVED_WRITE_OPTIONS = {"overwrite", "compute", "data_type"}


@dataclass
class _Placed:
    """A tile positioned on the output grid (integer offsets/sizes, ``z, y, x``)."""

    tile: Tile
    offset: Tuple[int, int, int]
    size: Tuple[int, int, int]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _place_tiles(tiles: Sequence[Tile], ds: Tuple[int, int, int]):
    """Place ``tiles`` on a common output grid.

    Returns ``(placed, grid_shape, out_voxel, origin)``. All tiles must share the
    voxel size and unit. A tile covers ``shape // ds`` output voxels starting at
    ``round((tile origin - global origin) / output voxel size)``.
    """
    vox = np.array(tiles[0].voxel_size)
    for t in tiles:
        if not np.allclose(t.voxel_size, vox, rtol=1e-3):
            raise ValueError(f"Tiles must share one voxel size, got {t.voxel_size} and {tuple(vox)}.")
        if t.units != tiles[0].units:
            raise ValueError(f"Tiles must share one unit, got {t.units!r} and {tiles[0].units!r}.")
    out_vox = vox * np.array(ds)
    origin = np.min([t.origin for t in tiles], axis=0)

    placed = []
    for t in tiles:
        offset = tuple(int(v) for v in np.rint((np.array(t.origin) - origin) / out_vox))
        size = tuple(int(n) // d for n, d in zip(t.shape, ds))
        if min(size) < 1:
            raise ValueError(f"Tile of shape {t.shape} is smaller than the downsampling factor {ds}.")
        placed.append(_Placed(t, offset, size))
    grid = tuple(int(v) for v in np.max([[o + s for o, s in zip(p.offset, p.size)] for p in placed], axis=0))
    return placed, grid, tuple(out_vox), tuple(float(v) for v in origin)


def _crop_bounds(crop, grid: Tuple[int, int, int]) -> List[Tuple[int, int]]:
    """``(start, stop)`` per ``z, y, x`` axis from a ``{"z": slice, ...}`` crop on the output grid."""
    bounds = []
    for axis, n in zip("zyx", grid):
        sel = (crop or {}).get(axis, (crop or {}).get(axis.upper(), slice(None)))
        if isinstance(sel, (int, np.integer)):
            sel = slice(int(sel), int(sel) + 1)
        if not isinstance(sel, slice):
            raise ValueError(f"crop[{axis!r}] must be a slice or an integer, got {sel!r}.")
        start, stop, step = sel.indices(n)
        if step != 1:
            raise ValueError(f"crop[{axis!r}] cannot have a step; use the downsampling factor instead.")
        if stop <= start:
            raise ValueError(f"crop[{axis!r}] is empty for an output of size {n} along {axis}.")
        bounds.append((start, stop))
    return bounds


def _downsample(block: np.ndarray, ds: Tuple[int, int, int]) -> np.ndarray:
    """Block-mean downsample a block whose shape is a multiple of ``ds``, to float32."""
    if ds == (1, 1, 1):
        return block.astype(np.float32)
    nz, ny, nx = block.shape
    r = block.reshape(nz // ds[0], ds[0], ny // ds[1], ds[1], nx // ds[2], ds[2])
    return r.mean(axis=(1, 3, 5), dtype=np.float32)


def _cast(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a float array to ``dtype``, rounding and clipping for integers."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(arr), info.min, info.max).astype(dtype)
    return arr.astype(dtype)


def _combine(acc: np.ndarray, cov: np.ndarray, blk: np.ndarray, blend: str) -> None:
    """Merge ``blk`` into the ``acc`` view in place (``cov`` marks voxels already written)."""
    if blend == "sum":
        acc += blk
    else:  # max / midline: running maximum within a group
        acc[...] = np.where(cov, np.maximum(acc, blk), blk)
    cov[...] = True


def _group_extents(placed: Sequence[_Placed], axis: int) -> Dict[Optional[str], Tuple[int, int]]:
    """``(start, stop)`` of every group along ``axis``, in output voxels."""
    extents: Dict[Optional[str], List[int]] = {}
    for p in placed:
        lo, hi = p.offset[axis], p.offset[axis] + p.size[axis]
        e = extents.setdefault(p.tile.group, [lo, hi])
        e[0], e[1] = min(e[0], lo), max(e[1], hi)
    return {g: (lo, hi) for g, (lo, hi) in extents.items()}


def _midline_ranges(placed: Sequence[_Placed], axis: int) -> Dict[Optional[str], Tuple[float, float]]:
    """Owner interval (output voxels, along ``axis``) of every group.

    Groups are ordered by the centre of their extent; consecutive groups are
    separated at the middle of their overlap (or of the gap between them).
    """
    ordered = sorted(_group_extents(placed, axis).items(), key=lambda kv: kv[1][0] + kv[1][1])
    cuts = [(a[1][1] + b[1][0]) / 2 for a, b in zip(ordered[:-1], ordered[1:])]
    bounds = [-math.inf] + cuts + [math.inf]
    return {g: (bounds[i], bounds[i + 1]) for i, (g, _) in enumerate(ordered)}


def _sigmoid_transitions(placed: Sequence[_Placed], axis: int) -> Dict[Optional[str], List[Tuple[int, int, int]]]:
    """Overlaps of every group with its neighbours along ``axis`` (groups ordered as for
    ``midline``), as ``(start, stop, direction)`` in output voxels: ``+1`` if the
    group's weight rises across the overlap, ``-1`` if it falls."""
    ordered = sorted(_group_extents(placed, axis).items(), key=lambda kv: kv[1][0] + kv[1][1])
    out: Dict[Optional[str], List[Tuple[int, int, int]]] = {g: [] for g, _ in ordered}
    for (ga, (alo, ahi)), (gb, (blo, bhi)) in zip(ordered[:-1], ordered[1:]):
        start, stop = max(alo, blo), min(ahi, bhi)
        if stop > start:  # groups separated by a gap never meet
            out[ga].append((start, stop, -1))
            out[gb].append((start, stop, +1))
    return out


# ---------------------------------------------------------------------------
# one tile inside one output region
# ---------------------------------------------------------------------------

@dataclass
class _Region:
    """One output region of one channel and the tiles that overlap it."""

    channel_index: int
    bounds: Tuple[int, int, int, int]  # y0, y1, x0, x1 in output voxels
    placed: List[_Placed]
    midline: Optional[tuple]
    transitions: Optional[tuple] = None  # sigmoid: (axis, _sigmoid_transitions(...))
    zrange: Tuple[int, int] = (0, 0)  # z0, z1 in output voxels (whole depth for projections)
    time_index: Optional[int] = None  # position on the output t axis, None without one
    samples: int = _WINDOW_SAMPLES  # values kept for the display window


def _source_chunks(data) -> Optional[Tuple[int, int, int]]:
    """``(z, y, x)`` chunk shape of a tile's data (zarr, h5py, dask, ...), or ``None``."""
    chunks = getattr(data, "chunksize", None)  # dask: ``chunks`` holds per-block sizes
    if chunks is None:
        chunks = getattr(data, "chunks", None)
    try:
        chunks = tuple(int(c) for c in chunks)
    except (TypeError, ValueError):
        return None
    return chunks if len(chunks) == 3 and all(c > 0 for c in chunks) else None


def _region_shape(tiles: Sequence[Tile], grid: Tuple[int, int, int], ds: Tuple[int, int, int],
                  flat: bool) -> Tuple[int, int, int]:
    """``(z, y, x)`` size of the output regions fused at once, in output voxels.

    A region is read in one go from every tile it overlaps, and a compressed source
    chunk is decoded in full by every region that touches it. So regions are made at
    least as large as the (downsampled) source chunks in y and x, preferably twice as
    large, which keeps the number of times each chunk is decoded close to one. Along
    z, 3D regions are cut into slabs of whole source chunks so that a region holds
    about ``_REGION_VOXELS`` voxels whatever the depth. Projections keep the whole z
    range (it is reduced while reading).
    """
    base = _REGION_2D if flat else _REGION_3D
    src = _source_chunks(tiles[0].data) if tiles else None
    cz = 1
    ry, rx = base
    if src is not None:
        cz, cy, cx = (max(1, -(-c // d)) for c, d in zip(src, ds))
        for mult in (2, 1):
            ry = max(base[0], min(mult * cy, _REGION_MAX))
            rx = max(base[1], min(mult * cx, _REGION_MAX))
            if flat or cz * ry * rx <= _REGION_VOXELS:
                break
    ry, rx = min(ry, grid[1]), min(rx, grid[2])
    if flat:
        return grid[0], ry, rx
    rz = max(1, _REGION_VOXELS // (ry * rx))
    if cz > 1:  # whole source chunks along z
        rz = max(cz, rz - rz % cz)
    return min(rz, grid[0]), ry, rx


def _overlap(p: _Placed, region, zrange: Tuple[int, int]):
    """Output-grid ranges ``(k0, k1, j0, j1, i0, i1)`` where tile ``p`` meets ``region``
    (``y0, y1, x0, x1``) within ``zrange`` (``z0, z1``), or ``None``."""
    y0, y1, x0, x1 = region
    oz, oy, ox = p.offset
    sz, sy, sx = p.size
    k0, k1 = max(zrange[0], oz), min(zrange[1], oz + sz)
    j0, j1 = max(y0, oy), min(y1, oy + sy)
    i0, i1 = max(x0, ox), min(x1, ox + sx)
    if k1 <= k0 or j1 <= j0 or i1 <= i0:
        return None
    return k0, k1, j0, j1, i0, i1


def _target(p: _Placed, region, zrange: Tuple[int, int], flat: bool):
    """Canvas slices of ``region`` covered by tile ``p`` (``None`` if they do not overlap)."""
    ov = _overlap(p, region, zrange)
    if ov is None:
        return None
    k0, k1, j0, j1, i0, i1 = ov
    yx = (slice(j0 - region[0], j1 - region[0]), slice(i0 - region[2], i1 - region[2]))
    return yx if flat else (slice(k0 - zrange[0], k1 - zrange[0]),) + yx


def _layout_blocks(index: int, channel: str, placed: Sequence[_Placed], yx: Tuple[int, int], depth: int,
                   region: Tuple[int, int, int], blend: str, axis: int) -> List[tuple]:
    """The regions of one channel that hold tiles of the layout ``placed``, see :class:`_RegionSeries`.

    ``region`` is the ``(z, y, x)`` size of a region. Looks up the regions of every tile, so
    the cost follows the tiles and not the regions of the whole output.
    """
    cz, cy, cx = region
    cells: Dict[Tuple[int, int, int], List[int]] = {}
    for i, p in enumerate(placed):
        ranges = []
        for offset, size, stop, step in zip(p.offset, p.size, (depth,) + yx, region):
            lo, hi = max(offset, 0), min(offset + size, stop)
            ranges.append(range(lo // step, (hi - 1) // step + 1) if hi > lo else range(0))
        for k in ranges[0]:
            for j in ranges[1]:
                for n in ranges[2]:
                    cells.setdefault((j, n, k), []).append(i)
    midline = (axis, _midline_ranges(placed, axis)) if blend == "midline" else None
    transitions = (axis, _sigmoid_transitions(placed, axis)) if blend == "sigmoid" else None
    return [(index, channel,
             (j * cy, min((j + 1) * cy, yx[0]), n * cx, min((n + 1) * cx, yx[1])),
             (k * cz, min((k + 1) * cz, depth)),
             midline, transitions, cells[(j, n, k)])
            for j, n, k in sorted(cells)]


def _read_block(job) -> np.ndarray:
    """Read and downsample the part of one tile inside one region (runs in workers).

    Returns float32 ``(h, w)`` for projections, ``(d, h, w)`` otherwise.
    """
    p, region, zrange, ds, projection = job
    k0, k1, j0, j1, i0, i1 = _overlap(p, region, zrange)
    oz, oy, ox = p.offset
    dz, dy, dx = ds
    src_y = slice((j0 - oy) * dy, (j1 - oy) * dy)
    src_x = slice((i0 - ox) * dx, (i1 - ox) * dx)
    data = p.tile.data

    if projection is not None:  # reduce over z at full resolution first, then downsample in yx
        if k0 == oz and k1 == oz + p.size[0]:
            z_start, z_stop = 0, p.tile.shape[0]  # whole tile, including trailing planes
        else:  # z crop
            z_start, z_stop = (k0 - oz) * dz, (k1 - oz) * dz
        reduced = None
        for z in range(z_start, z_stop, _SLAB_PLANES):
            raw = np.asarray(data[z : min(z + _SLAB_PLANES, z_stop), src_y, src_x])
            if projection == "max":
                part = raw.max(axis=0)
                reduced = part if reduced is None else np.maximum(reduced, part)
            else:
                part = raw.sum(axis=0, dtype=np.float32)
                reduced = part if reduced is None else reduced + part
        return _downsample(reduced[None], (1, dy, dx))[0]

    out = np.empty((k1 - k0, j1 - j0, i1 - i0), np.float32)
    step = max(1, _SLAB_PLANES // dz)
    for a in range(k0, k1, step):
        b = min(a + step, k1)
        raw = np.asarray(data[(a - oz) * dz : (b - oz) * dz, src_y, src_x])
        out[a - k0 : b - k0] = _downsample(raw, ds)
    return out


def _merge_groups(canvases, midline, region, flat: bool, shape, z0: int = 0) -> np.ndarray:
    """Merge per-group canvases: each group wins on its own side of the cut, others fill gaps."""
    if len(canvases) == 1:
        return next(iter(canvases.values()))[0]
    y0, _, x0, _ = region
    axis, ranges = midline  # axis is an index into (z, y, x)
    start = (y0, x0)[axis - 1] if axis > 0 else z0
    arr_axis = (axis - 1) if flat else axis  # canvas axis of the midline axis
    length = shape[arr_axis]
    pos = start + np.arange(length) + 0.5
    bshape = [1] * len(shape)
    bshape[arr_axis] = length

    result = np.zeros(shape, np.float32)
    filled = np.zeros(shape, bool)
    for g, (lo, hi) in ranges.items():
        if g not in canvases:
            continue
        acc, cov = canvases[g]
        m = ((pos >= lo) & (pos < hi)).reshape(bshape) & cov
        result[m] = acc[m]
        filled |= m
    for acc, cov in canvases.values():
        m = ~filled & cov
        result[m] = acc[m]
        filled |= m
    return result


# Sigmoid blending: every voxel is a weighted mean of the tiles that cover it. A
# tile's weight is d**steepness, with d the distance from the voxel to the tile's
# border (multiplied over the axes), so where two tiles overlap the share of one
# is 1 / (1 + (d_other / d_self)**steepness) = sigmoid(steepness * ln(d_self / d_other)):
# 0 at its border, 1/2 in the middle of the overlap and 1 at the other tile's
# border, whatever the overlap width. Groups are then merged the same way across
# each overlap with their neighbour along the midline axis, so the switch happens
# where ``midline`` cuts, but smoothly. Weights are kept as logarithms and summed
# with a running maximum (a streaming softmax), so tiny weights never underflow.

def _log_border_distance(start: int, size: int, lo: int, hi: int) -> np.ndarray:
    """ln of the distance from the centres of output voxels ``lo..hi`` to the nearer
    border of the interval ``[start, start + size)`` (all inside it)."""
    pos = np.arange(lo, hi, dtype=np.float64) + 0.5 - start
    return np.log(np.minimum(pos, size - pos))


def _tile_log_weight(p: _Placed, region, zrange: Tuple[int, int], flat: bool, steepness: float) -> np.ndarray:
    """Log weight of tile ``p`` on its part of ``region`` (shape of :func:`_read_block`)."""
    k0, k1, j0, j1, i0, i1 = _overlap(p, region, zrange)
    ly = _log_border_distance(p.offset[1], p.size[1], j0, j1)[:, None]
    lx = _log_border_distance(p.offset[2], p.size[2], i0, i1)[None, :]
    lw = ly + lx
    if not flat:
        lw = _log_border_distance(p.offset[0], p.size[0], k0, k1)[:, None, None] + lw
    return (steepness * lw).astype(np.float32)


def _softmax_add(state, sel, values: np.ndarray, log_weight: np.ndarray, mask=None) -> None:
    """Add ``values`` with weights ``exp(log_weight)`` to the running weighted sum
    ``state = (acc, wsum, log_max)`` at ``sel`` (a tuple of slices), in place.

    Only voxels where ``mask`` is true are updated (all of them when ``None``).
    ``log_weight`` may be any array broadcastable to the selection.

    Equivalent to ``new = max(old, lw); acc = acc * exp(old - new) + exp(lw - new) * v``
    (and likewise for ``wsum``), but one of the two exponentials is always
    ``exp(0) = 1``, so a single ``exp(-|lw - old|)`` is computed and the updates are
    done in place on the voxels where each case applies. The result is identical.
    """
    acc, wsum, log_max = (a[sel] for a in state)
    d = log_weight - log_max  # +inf where nothing was added yet (log_max = -inf)
    up = d > 0  # the new value carries the largest weight: rescale what is there
    e = np.negative(np.abs(d, out=d), out=d)
    np.exp(e, out=e)
    keep = ~up
    if mask is not None:
        up &= mask
        keep &= mask
    ev = np.multiply(e, values)
    np.multiply(acc, e, out=acc, where=up)
    np.add(acc, values, out=acc, where=up)
    np.add(acc, ev, out=acc, where=keep)
    np.multiply(wsum, e, out=wsum, where=up)
    np.add(wsum, np.float32(1), out=wsum, where=up)
    np.add(wsum, e, out=wsum, where=keep)
    np.copyto(log_max, np.broadcast_to(log_weight, log_max.shape), where=up)


def _runs(flags: np.ndarray) -> List[slice]:
    """Slices of the runs of ``True`` in a 1D boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], flags, [False])).astype(np.int8)))
    return [slice(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _sigmoid_add(state, sl, blk: np.ndarray, log_weight: np.ndarray, previous: List[tuple]) -> None:
    """Add tile block ``blk`` at canvas slices ``sl`` to a group's softmax ``state``.

    Where no tile was added before (most of a tile, away from the overlaps), the
    weighted sum is simply initialised with the tile; the weighted update of
    :func:`_softmax_add` only runs on the boxes where the tile meets the tiles in
    ``previous`` (their canvas slices, to which ``sl`` is then appended). The
    result is identical to running :func:`_softmax_add` on the whole block.
    """
    overlaps = []
    for q in previous:
        inter = [slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(sl, q)]
        if all(i.stop > i.start for i in inter):
            overlaps.append(tuple(slice(i.start - a.start, i.stop - a.start) for i, a in zip(inter, sl)))
    previous.append(sl)
    acc, wsum, log_max = (a[sl] for a in state)
    if not overlaps:
        acc[...] = blk
        wsum[...] = 1
        log_max[...] = log_weight
        return
    todo = np.zeros(blk.shape, bool)
    for o in overlaps:
        todo[o] = True
    fresh = ~todo
    np.copyto(acc, blk, where=fresh)
    np.copyto(wsum, np.float32(1), where=fresh)
    np.copyto(log_max, log_weight, where=fresh)
    for o in overlaps:  # boxes may overlap each other: every voxel is updated once
        mask = todo[o]
        if mask.any():
            _softmax_add(state, tuple(slice(a.start + b.start, a.start + b.stop) for a, b in zip(sl, o)),
                         blk[o], log_weight[o], mask=mask)
            todo[o] = False


def _new_softmax_state(shape):
    return np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.full(shape, -np.inf, np.float32)


def _group_log_weight(transitions, pos: np.ndarray, steepness: float) -> np.ndarray:
    """Log weight of a group at voxel centres ``pos`` along the midline axis: across
    each overlap with a neighbour, ``steepness * ln`` of the relative distance from the
    side where the group comes in; 0 (full weight) outside its overlaps."""
    lw = np.zeros(pos.shape, np.float64)
    for start, stop, direction in transitions:
        d = pos - start if direction > 0 else stop - pos
        lw += np.log(np.clip(d, 0.5, stop - start - 0.5) / (stop - start - 0.5))
    return steepness * lw


def _merge_sigmoid(canvases, transitions, region, flat: bool, shape, steepness: float,
                   z0: int = 0) -> np.ndarray:
    """Weighted mean of every group, then of the groups across their overlaps."""
    means = {}
    for g, (acc, wsum, _) in canvases.items():
        cov = wsum > 0
        mean = np.zeros(shape, np.float32)
        np.divide(acc, wsum, out=mean, where=cov)
        means[g] = (mean, cov)
    if len(means) == 1:
        return next(iter(means.values()))[0]

    y0, _, x0, _ = region
    axis, per_group = transitions  # axis is an index into (z, y, x)
    start = (y0, x0)[axis - 1] if axis > 0 else z0
    arr_axis = (axis - 1) if flat else axis
    length = shape[arr_axis]
    pos = start + np.arange(length) + 0.5
    bshape = [1] * len(shape)
    bshape[arr_axis] = length
    # Where a single group covers a voxel its weighted mean is its own value; the
    # weighted merge only runs on the slabs (along the midline axis) where groups meet.
    count = np.zeros(shape, np.uint8)
    result = np.zeros(shape, np.float32)
    for mean, cov in means.values():
        count += cov
        np.copyto(result, mean, where=cov)
    shared = count > 1
    other = tuple(i for i in range(len(shape)) if i != arr_axis)
    for run in _runs(shared.any(axis=other)):
        sel = [slice(None)] * len(shape)
        sel[arr_axis] = run
        sel = tuple(sel)
        state = _new_softmax_state(result[sel].shape)
        everything = (slice(None),) * len(shape)
        for g, (mean, cov) in means.items():
            lw = _group_log_weight(per_group[g], pos[run], steepness).astype(np.float32)
            bshape[arr_axis] = run.stop - run.start
            _softmax_add(state, everything, mean[sel], lw.reshape(bshape), mask=cov[sel])
        acc, wsum, _ = state
        merged = np.zeros(acc.shape, np.float32)
        np.divide(acc, wsum, out=merged, where=wsum > 0)
        np.copyto(result[sel], merged, where=shared[sel])
    return result


def _fuse_region(r: _Region, arr, ds, projection, blend, dtype,
                 steepness: float = SIGMOID_STEEPNESS) -> np.ndarray:
    """Read the tiles of region ``r``, merge them, write the region into ``arr`` and
    return a small sample of its non-zero values (for the display window).

    Runs in the workers: regions are aligned with the chunks of ``arr``, so two
    regions never write the same chunk and need no lock.
    """
    flat = projection is not None
    y0, y1, x0, x1 = r.bounds
    z0, z1 = r.zrange
    shape = (y1 - y0, x1 - x0) if flat else (z1 - z0, y1 - y0, x1 - x0)
    sigmoid = blend == "sigmoid"
    canvases: Dict[Optional[str], tuple] = {}
    filled: Dict[Optional[str], List[tuple]] = {}  # sigmoid: canvas slices of the tiles added so far
    for p in r.placed:
        key = p.tile.group if blend in ("midline", "sigmoid") else None
        if key not in canvases:
            canvases[key] = (_new_softmax_state(shape) if sigmoid
                             else (np.zeros(shape, np.float32), np.zeros(shape, bool)))
        sl = _target(p, r.bounds, r.zrange, flat)
        blk = _read_block((p, r.bounds, r.zrange, ds, projection))
        if sigmoid:
            _sigmoid_add(canvases[key], sl, blk, _tile_log_weight(p, r.bounds, r.zrange, flat, steepness),
                         filled.setdefault(key, []))
        else:
            acc, cov = canvases[key]
            _combine(acc[sl], cov[sl], blk, blend)
        del blk

    if sigmoid:
        merged = _merge_sigmoid(canvases, r.transitions, r.bounds, flat, shape, steepness, z0=z0)
    else:
        merged = _merge_groups(canvases, r.midline, r.bounds, flat, shape, z0=z0)
    del canvases
    block = _cast(merged, dtype)
    index = (r.channel_index,) if r.time_index is None else (r.time_index, r.channel_index)
    if flat:
        arr[index + (slice(y0, y1), slice(x0, x1))] = block
    else:
        arr[index + (slice(z0, z1), slice(y0, y1), slice(x0, x1))] = block
    values = block.ravel()[:: max(1, block.size // r.samples)]
    return values[values != 0]


def _fuse_level0(regions, arr, *, ds, projection, blend, dtype, steepness: float,
                 workers: int, executor: str, progress: bool) -> Dict[int, np.ndarray]:
    """Fuse and write every region of level 0; return a sample of values per channel index.

    The unit of parallel work is one whole region: a worker reads the tiles that
    overlap it, merges them and writes the result. Regions match the chunks of
    ``arr`` exactly, so concurrent writes never touch the same chunk. At most
    ``2 * workers`` regions are in flight at a time, which bounds memory.
    """
    from tqdm.auto import tqdm

    pool = None
    if workers > 1:
        pool = (ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor)(max_workers=workers)
    window = 2 * workers
    samples: Dict[int, List[np.ndarray]] = {}
    args = (arr, ds, projection, blend, dtype, steepness)

    def collect(r, values):
        samples.setdefault(r.channel_index, []).append(values)
        bar.update(1)

    try:
        with tqdm(total=len(regions), desc="fusing regions", unit="region", disable=not progress) as bar:
            if pool is None:
                for r in regions:
                    collect(r, _fuse_region(r, *args))
            else:
                in_flight = deque()
                for r in regions:
                    in_flight.append((r, pool.submit(_fuse_region, r, *args)))
                    while len(in_flight) >= window:
                        done_r, fut = in_flight.popleft()
                        collect(done_r, fut.result())
                while in_flight:
                    done_r, fut = in_flight.popleft()
                    collect(done_r, fut.result())
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    return {c: np.concatenate(v) for c, v in samples.items()}


class _RegionSeries:
    """The regions of every fused timepoint, built when they are read, since there can
    be many timepoints.

    ``blocks`` holds ``(channel index, channel, bounds, zrange, midline, transitions,
    indices)`` for one timepoint, where ``indices`` are the tiles of the channel's layout
    that overlap the region. It is a list when every timepoint has the same layout, or a
    function of the timepoint when each has its own. Regions in which a timepoint has no
    tile to read are left out: they keep the fill value (0).
    """

    def __init__(self, blocks: Union[Sequence[tuple], Callable[[int], Sequence[tuple]]], plan: "FusionPlan"):
        self.blocks = blocks
        self.plan = plan
        self.stacked = "t" in plan.axes
        self.samples = max(1, _WINDOW_SAMPLES // len(plan.timepoints))
        self.total = sum(any(plan.frames[tp][b[1]][i] is not None for i in b[6])
                         for tp in plan.timepoints for b in self._blocks(tp))

    def _blocks(self, timepoint: int) -> Sequence[tuple]:
        return self.blocks(timepoint) if callable(self.blocks) else self.blocks

    def __len__(self) -> int:
        return self.total

    def __iter__(self) -> Iterator[_Region]:
        for ti, tp in enumerate(self.plan.timepoints):
            frame = self.plan.frames[tp]
            for ci, channel, bounds, zrange, midline, transitions, indices in self._blocks(tp):
                inside = [frame[channel][i] for i in indices if frame[channel][i] is not None]
                if inside:
                    yield _Region(ci, bounds, inside, midline, transitions, zrange,
                                  ti if self.stacked else None, self.samples)


def _display_windows(samples: Dict[int, np.ndarray], n_channels: int, dtype) -> List[Tuple[float, float]]:
    """``(start, end)`` display window per channel from the fused-value samples."""
    windows = []
    for c in range(n_channels):
        values = samples.get(c)
        if values is None or values.size == 0:
            windows.append((0.0, 1.0))
            continue
        low, high = (float(v) for v in np.percentile(values, DISPLAY_PERCENTILES))
        windows.append((low, high if high > low else low + 1.0))
    return windows


def _write_display_windows(path, windows: Sequence[Tuple[float, float]]) -> None:
    """Store the display windows in the OMERO channel metadata (NGFF 0.4 and 0.5)."""
    root = zarr.open_group(str(path), mode="r+")
    attrs = root.attrs.asdict()
    nested = "ome" in attrs  # NGFF 0.5 keeps everything under "ome"
    omero = (attrs["ome"] if nested else attrs).get("omero")
    if not omero:
        return
    for channel, (start, end) in zip(omero.get("channels", []), windows):
        channel.setdefault("window", {}).update({"start": start, "end": end})
    root.attrs.update({"ome": attrs["ome"]} if nested else {"omero": omero})


def _write_provenance(path, plan: "FusionPlan") -> None:
    """Record which acquired timepoints the output's ``t`` axis holds and the reference."""
    zarr.open_group(str(path), mode="r+").attrs[PROVENANCE_KEY] = {
        "timepoints": list(plan.timepoints),
        "reference_timepoint": plan.reference_timepoint,
    }


def display_windows(path) -> List[Tuple[float, float]]:
    """Display window ``(start, end)`` of every channel of an OME-Zarr image, from its
    OMERO metadata (written by :func:`fuse_tiles` from the fused values)."""
    attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    omero = attrs.get("ome", attrs).get("omero") or {}
    return [(float(c["window"]["start"]), float(c["window"]["end"]))
            for c in omero.get("channels", []) if "window" in c]


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

@dataclass
class FusionPlan:
    """Level-0 geometry of a fusion, computed without reading any pixel data."""

    tiles: List[Tile]  # tiles that are read, at every timepoint (skipped tiles excluded)
    placed: Dict[str, List[_Placed]]  # per output channel, offsets relative to the output grid
    layout: Dict[str, List[_Placed]]  # like ``placed`` but including skipped tiles: the geometry of every timepoint,
    # or with reference_timepoint="all" the tiles of all fused timepoints (see ``layouts``)
    channels: List[str]
    ds: Tuple[int, int, int]
    blend: str
    dtype: np.dtype
    axes: str  # "czyx" or "cyx", with a leading "t" for several timepoints
    grid_shape: Tuple[int, int, int]  # (z, y, x) of the 3D output grid
    shape: Tuple[int, ...]  # level 0, in ``axes`` order
    scale: Tuple[float, ...]  # level-0 voxel size of the spatial axes
    origin: Tuple[float, float, float]  # sample position of output voxel [0, 0, 0]
    n_skipped: int = 0  # tiles left blank
    fields: List[str] = dataclasses.field(default_factory=list)  # fields of the tiles (skipped ones included)
    timepoints: List[int] = dataclasses.field(default_factory=list)  # fused timepoints, in output order
    reference_timepoint: Union[int, str] = 0  # timepoint whose tiles set the geometry, or ALL_TIMEPOINTS
    # per timepoint and channel, the tiles aligned with ``layout``; None where a tile is skipped or missing
    frames: Dict[int, Dict[str, List[Optional[_Placed]]]] = dataclasses.field(default_factory=dict)
    n_missing: int = 0  # tiles of ``layout`` that a timepoint does not have
    n_available: int = 1  # timepoints of the tiles the plan was made from
    # reference_timepoint="all": the layout of every timepoint, aligned with ``frames``; empty otherwise
    layouts: Dict[int, Dict[str, List[_Placed]]] = dataclasses.field(default_factory=dict)

    @property
    def per_timepoint(self) -> bool:
        """Whether every timepoint is placed at its own positions."""
        return self.reference_timepoint == ALL_TIMEPOINTS

    @property
    def nbytes(self) -> int:
        """Size of level 0 in bytes."""
        return math.prod(self.shape) * self.dtype.itemsize

    def summary(self) -> str:
        """One line such as ``2 ch x 260 x 1100 x 1500 px (czyx), 1.71 GB, voxel 12 x 11.7 x 11.7 micrometer`` (MB below 0.1 GB);
        ``5 t x 2 ch x ...`` for several timepoints, followed by the reference timepoint
        or by ``own positions per timepoint``."""
        dims = " x ".join(str(v) for v in self.shape[-len(self.scale):])
        vox = " x ".join(f"{v:.3g}" for v in self.scale)
        size = f"{self.nbytes / 1e9:.2f} GB" if self.nbytes >= 1e8 else f"{self.nbytes / 1e6:.1f} MB"
        blank = f", {self.n_skipped} tile(s) left blank" if self.n_skipped else ""
        missing = f", {self.n_missing} tile(s) missing at some timepoints (blank)" if self.n_missing else ""
        time = f"{len(self.timepoints)} t x " if "t" in self.axes else ""
        reference = ""
        if self.n_available > 1:
            reference = (", own positions per timepoint" if self.per_timepoint
                         else f", reference timepoint {self.reference_timepoint}")
        fields = f"; ERROR: {len(self.fields)} separate fields ({', '.join(self.fields)}), cannot be fused" \
            if len(self.fields) > 1 else ""
        return (f"{time}{len(self.channels)} ch x {dims} px ({self.axes}), {size}, voxel {vox} {self.tiles[0].units}"
                f"{reference}{blank}{missing}{fields}")

    @property
    def multiple_fields(self) -> bool:
        """Whether the fused tiles come from more than one field (such a plan is not fused)."""
        return len(self.fields) > 1

    @property
    def error(self) -> Optional[str]:
        """Why the plan cannot be fused (tiles from separate fields), or ``None``."""
        return multiple_fields_message(self.fields) if self.multiple_fields else None


def _select_timepoints(tiles: Sequence[Tile], timepoints, reference_timepoint) -> Tuple[List[int], Union[int, str], List[int]]:
    """``(fused timepoints, reference timepoint or ALL_TIMEPOINTS, available timepoints)``, validated."""
    available = sorted({t.timepoint for t in tiles})
    if timepoints is None:
        selected = available
    else:
        selected = sorted({int(tp) for tp in timepoints})
        missing = [tp for tp in selected if tp not in available]
        if missing:
            raise ValueError(f"Timepoints {missing} not found; available: {available}.")
        if not selected:
            raise ValueError("No timepoint selected.")
    if reference_timepoint == ALL_TIMEPOINTS:
        return selected, ALL_TIMEPOINTS, available
    if isinstance(reference_timepoint, str) and not reference_timepoint.strip().lstrip("-").isdigit():
        raise ValueError(f"reference_timepoint must be a timepoint or {ALL_TIMEPOINTS!r}, got {reference_timepoint!r}.")
    reference = available[0] if reference_timepoint is None else int(reference_timepoint)
    if reference not in available:
        raise ValueError(f"Reference timepoint {reference} not found; available: {available}.")
    return selected, reference, available


def _build_frames(tiles: Sequence[Tile], layout: Dict[str, List[_Placed]], timepoints: Sequence[int],
                  reference: int, skipped: set) -> Tuple[dict, int, int]:
    """Place the tiles of every timepoint with the geometry of the reference timepoint.

    Returns ``(frames, n_skipped, n_missing)``, see :attr:`FusionPlan.frames`. A tile that
    has another shape, voxel size or unit than its reference tile cannot be placed the
    same way and raises ``ValueError``.
    """
    keys = series_keys(tiles)
    at = {(k, t.timepoint): t for k, t in zip(keys, tiles)}
    key_of = {id(t): k for k, t in zip(keys, tiles)}
    known = {k for k, t in zip(keys, tiles) if t.timepoint == reference}
    wanted = set(timepoints)
    ignored = sum(t.timepoint in wanted and k not in known for k, t in zip(keys, tiles))
    if ignored:
        logger.warning("%d tile(s) have no counterpart at reference timepoint %d: they are not fused.",
                       ignored, reference)

    frames: Dict[int, Dict[str, List[Optional[_Placed]]]] = {}
    n_skipped = n_missing = 0
    for tp in timepoints:
        frames[tp] = {}
        for channel, placed in layout.items():
            row: List[Optional[_Placed]] = []
            for p in placed:
                t = at.get((key_of[id(p.tile)], tp))
                if t is None:
                    n_missing += 1
                    row.append(None)
                    continue
                if (t.shape != p.tile.shape or t.units != p.tile.units
                        or not np.allclose(t.voxel_size, p.tile.voxel_size, rtol=1e-3)):
                    raise ValueError(
                        f"Tile {t.attrs.get('stack')!r} of channel {t.channel!r} at timepoint {tp} (shape {t.shape}, "
                        f"voxel {t.voxel_size} {t.units}) does not match reference timepoint {reference} "
                        f"(shape {p.tile.shape}, voxel {p.tile.voxel_size} {p.tile.units}): leave this "
                        "timepoint out with timepoints=..."
                    )
                if id(t) in skipped:
                    n_skipped += 1
                    row.append(None)
                    continue
                row.append(_Placed(t, p.offset, p.size))
            frames[tp][channel] = row
    if n_missing:
        logger.warning("%d tile(s) are missing at some timepoints: they stay blank there.", n_missing)
    return frames, n_skipped, n_missing


def _build_own_frames(tiles: Sequence[Tile], placed: Sequence[_Placed], names: Sequence[str],
                      timepoints: Sequence[int], skipped: set) -> Tuple[dict, dict, int, int]:
    """Split ``placed`` into the tiles of every timepoint, each at its own position.

    Returns ``(layouts, frames, n_skipped, n_missing)``, see :attr:`FusionPlan.layouts`. A
    position that a timepoint does not have is missing there; one cropped away is not.
    """
    layouts = {tp: {n: [] for n in names} for tp in timepoints}
    frames = {tp: {n: [] for n in names} for tp in timepoints}
    n_skipped = 0
    for p in placed:
        tp, channel = p.tile.timepoint, p.tile.channel
        layouts[tp][channel].append(p)
        blank = id(p.tile) in skipped
        n_skipped += blank
        frames[tp][channel].append(None if blank else p)
    keys = series_keys(tiles)
    present = {(k, t.timepoint) for k, t in zip(keys, tiles)}
    n_missing = sum((k, tp) not in present for k in set(keys) for tp in timepoints)
    if n_missing:
        logger.warning("%d tile(s) are missing at some timepoints: they stay blank there.", n_missing)
    return layouts, frames, n_skipped, n_missing


def plan_fusion(
    tiles: Sequence[Tile],
    *,
    ds: Sequence[int] = (1, 1, 1),
    projection: Optional[str] = None,
    blend: str = "sigmoid",
    channels: Optional[Sequence[str]] = None,
    crop: Optional[dict] = None,
    skip: Optional[Sequence[Tile]] = None,
    out_dtype=None,
    midline_axis: str = "x",
    timepoints: Optional[Sequence[int]] = None,
    reference_timepoint: Union[int, str, None] = None,
) -> FusionPlan:
    """Validate fusion arguments and compute the level-0 geometry (see :func:`fuse_tiles`)."""
    tiles = list(tiles)
    if not tiles:
        raise ValueError("No tiles to fuse.")
    if blend not in BLENDS:
        raise ValueError(f"blend must be one of {BLENDS}, got {blend!r}.")
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}, got {projection!r}.")
    ds = tuple(int(v) for v in ds)
    if len(ds) != 3 or min(ds) < 1:
        raise ValueError(f"ds must be three positive integers (z, y, x), got {ds}.")
    if midline_axis not in ("x", "y", "z"):
        raise ValueError("midline_axis must be 'x', 'y' or 'z'.")
    flat = projection is not None
    if flat and midline_axis == "z":
        raise ValueError("midline_axis='z' is not available for projections.")

    names = list(dict.fromkeys(t.channel for t in tiles))
    if channels is not None:
        missing = [c for c in channels if c not in names]
        if missing:
            raise ValueError(f"Channels {missing} not found; available: {names}.")
        names = list(channels)
        if not names:
            raise ValueError("No channel selected.")
    tiles = [t for n in names for t in tiles if t.channel == n]
    selected, reference, available = _select_timepoints(tiles, timepoints, reference_timepoint)

    per_timepoint = reference == ALL_TIMEPOINTS
    # With own positions the grid fits all available timepoints, so parts fused separately line up.
    placed, grid, out_vox, origin = _place_tiles(tiles if per_timepoint else
                                                 [t for t in tiles if t.timepoint == reference], ds)
    if per_timepoint:
        placed = [p for p in placed if p.tile.timepoint in selected]
    if crop:  # shift the grid to the crop and drop the tiles outside it
        bounds = _crop_bounds(crop, grid)
        start = tuple(b[0] for b in bounds)
        grid = tuple(b[1] - b[0] for b in bounds)
        origin = tuple(o + s * v for o, s, v in zip(origin, start, out_vox))
        placed = [_Placed(p.tile, tuple(o - s for o, s in zip(p.offset, start)), p.size) for p in placed]
        placed = [p for p in placed if _overlap(p, (0, grid[1], 0, grid[2]), (0, grid[0])) is not None]
        if not placed:
            raise ValueError("No tile intersects the crop.")
    layout = {n: [p for p in placed if p.tile.channel == n] for n in names}
    fields = fields_of([p.tile for p in placed])
    # skipped tiles keep their place in the geometry (the output does not shrink) but are not read
    skipped = {id(t) for t in skip or ()}
    layouts = {}
    if per_timepoint:
        layouts, frames, n_skipped, n_missing = _build_own_frames(tiles, placed, names, selected, skipped)
    else:
        frames, n_skipped, n_missing = _build_frames(tiles, layout, selected, reference, skipped)
    read = [p.tile for tp in selected for n in names for p in frames[tp][n] if p is not None]
    if not read:
        raise ValueError("Every tile is skipped: nothing to fuse.")
    placed_by_channel = {n: [p for p in layout[n] if id(p.tile) not in skipped] for n in names}

    if blend == "midline" and all(t.group is None for t in tiles):
        logger.warning("blend='midline' but tiles have no group; falling back to 'max'.")
        blend = "max"

    if out_dtype is None:
        out_dtype = np.float32 if blend == "sum" else np.result_type(*{t.dtype for t in tiles})
    out_dtype = np.dtype(out_dtype)

    spatial = grid[1:] if flat else grid
    leading = (len(selected), len(names)) if len(selected) > 1 else (len(names),)
    return FusionPlan(
        tiles=read,
        placed=placed_by_channel,
        layout=layout,
        channels=names,
        ds=ds,
        blend=blend,
        dtype=out_dtype,
        axes=("t" if len(selected) > 1 else "") + ("cyx" if flat else "czyx"),
        grid_shape=grid,
        shape=leading + tuple(spatial),
        scale=tuple(out_vox[1:] if flat else out_vox),
        origin=origin,
        n_skipped=n_skipped,
        fields=fields,
        timepoints=selected,
        reference_timepoint=reference,
        frames=frames,
        n_missing=n_missing,
        n_available=len(available),
        layouts=layouts,
    )


def _spatial_factor(value, flat: bool):
    """Pyramid downscale factor for the output axes (``(z, y, x)`` given; z dropped for projections)."""
    if value is None:
        return 2
    if isinstance(value, (list, tuple)):
        value = tuple(int(v) for v in value)
        if len(value) == 1:
            return value[0]
        if flat and len(value) == 3:
            return value[1:]
    return value


def _check_write_options(options: dict) -> None:
    fields = {f.name for f in dataclasses.fields(ZarrWriteConfig)}
    unknown = set(options) - fields
    if unknown:
        raise TypeError(f"Unknown OME-Zarr option(s) {sorted(unknown)}; valid ones are {sorted(fields - _RESERVED_WRITE_OPTIONS)}.")
    reserved = set(options) & _RESERVED_WRITE_OPTIONS
    if reserved:
        raise TypeError(f"Option(s) {sorted(reserved)} are set by fuse_tiles and cannot be passed.")


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def fuse_tiles(
    tiles: Sequence[Tile],
    out: str | Path,
    *,
    ds: Sequence[int] = (1, 1, 1),
    projection: Optional[str] = None,
    blend: str = "sigmoid",
    channels: Optional[Sequence[str]] = None,
    channel_names: Optional[Sequence[str]] = None,
    channel_colors: Optional[Sequence[str]] = None,
    crop: Optional[dict] = None,
    skip: Optional[Sequence[Tile]] = None,
    levels: int = 1,
    downscale_factor=2,
    workers: int = 1,
    executor: str = "auto",
    out_dtype=None,
    midline_axis: str = "x",
    sigmoid_steepness: float = SIGMOID_STEEPNESS,
    timepoints: Optional[Sequence[int]] = None,
    reference_timepoint: Union[int, str, None] = None,
    overwrite: bool = False,
    name: Optional[str] = None,
    temp_dir: str | Path | None = None,
    temp_compression: bool = False,
    progress: bool = True,
    **write_options,
):
    """Fuse ``tiles`` into an OME-Zarr image and return it as a ``ZarrManager``.

    Parameters
    ----------
    tiles
        Tiles to fuse (all must share voxel size and unit).
    out
        Output ``.ome.zarr`` path.
    ds
        ``(z, y, x)`` integer downsampling factors applied while reading the tiles
        (each output voxel is the mean of its block).
    projection
        ``None`` for a 3D volume, ``"max"``/``"sum"`` for a 2D projection along z.
    blend
        How tiles that overlap are merged:

        * ``"sigmoid"`` (default) -- smooth transition: across an overlap, the
          share of each tile follows a sigmoid of the log ratio of the distances
          to the two tile borders, from 0 at a tile's border through 1/2 in the
          middle of the overlap to 1 at the other tile's border, so no seam is
          visible. The tiles of each ``group`` are fused separately, then the
          groups are merged the same way across their overlap along
          ``midline_axis`` (the switch is centred where ``"midline"`` cuts);
        * ``"max"`` -- keep the brighter value;
        * ``"sum"`` -- add the values, so overlaps show up as brighter bands;
        * ``"midline"`` -- ``max`` within each tile ``group``; where groups
          overlap, the image switches sharply from one group to the next at the
          middle of their overlap, along ``midline_axis``.
    channels
        Tile channels to fuse, in output order. Default: all, in tile order.
    channel_names, channel_colors
        Names and colours (hex or matplotlib names) written for the output
        channels. Default: the tile channel names and the ZarrSmith palette.
    crop
        Part of the output to write, in output voxels (after ``ds``), e.g.
        ``{"y": slice(100, 900), "z": slice(0, 50)}``. For projections, a ``z``
        crop limits the planes that are projected.
    skip
        Tiles to leave blank: they are not read, but the output keeps the size and
        position it has with them, so their area stays 0 (where no other tile
        covers it). With ``blend="midline"``/``"sigmoid"`` they still count for
        the position of the switch between groups.
    levels
        Number of pyramid levels.
    downscale_factor
        Factor between pyramid levels: one integer or ``(z, y, x)``.
    workers
        Number of regions fused in parallel (threads or processes, see ``executor``).
    executor
        ``"process"``, ``"thread"`` or ``"auto"`` (processes when ``workers > 1`` and
        every tile's data is picklable, e.g. :class:`~zarrsmith.mosaic_manager.H5Array`;
        threads otherwise). h5py serialises reads inside one process, so processes
        are what actually speeds up reading from HDF5 tiles.
    out_dtype
        Output dtype. Default: input dtype for ``sigmoid``/``max``/``midline``,
        ``float32`` for ``sum``.
    midline_axis
        ``"x"``, ``"y"`` or ``"z"``: axis along which groups are separated
        (``midline`` and ``sigmoid``).
    sigmoid_steepness
        Steepness of the ``sigmoid`` transition (> 0). The share of a tile
        goes from 10% to 90% over the central ``1 - 2 / (1 + 9 ** (1 / s))``
        of the overlap: about 27% for the default 4, 14% for 8, 50% for 2.
    timepoints
        Timepoints to fuse (``Tile.timepoint`` values). Default: all. With more than
        one, the output has a leading ``t`` axis (``tczyx``, or ``tcyx`` for
        projections), in increasing order; with one, it has none.
    reference_timepoint
        Timepoint whose tiles set the geometry (output grid, tile offsets, transitions
        between tiles and groups); it need not be among ``timepoints``. Every timepoint
        is fused with this geometry, so ``crop``, tile selections and the output shape
        hold for all of them, and timepoints fused in separate runs line up. Default:
        the first timepoint of ``tiles``. Tiles are matched across timepoints with
        :func:`~zarrsmith.mosaic_manager.tile.series_keys`. A tile missing at a timepoint
        stays blank there (with a warning), a tile without a counterpart at the reference
        is not fused (with a warning), and a tile with another shape, voxel size or
        unit than its reference tile raises ``ValueError``.

        ``"all"`` places every timepoint at its own positions instead. The output grid
        covers the tiles of all timepoints of ``tiles``, also those not in ``timepoints``,
        so ``crop`` and the output shape hold for all of them and timepoints fused in
        separate runs line up. Tile offsets and transitions are computed for each
        timepoint, so tiles and the overlaps between them follow the reported positions.
        A tile missing at a timepoint stays blank there (with a warning).
    overwrite
        Replace ``out`` if it is an existing Zarr store (with a warning, once the
        fusion has succeeded). Otherwise an existing non-empty ``out`` raises
        ``FileExistsError``.
    name
        Image name stored in the metadata.
    temp_dir
        Folder for the temporary fused level 0 (deleted at the end). Default:
        next to ``out``. A local disk is faster when ``out`` is on a network share.
    temp_compression
        Compress the temporary level 0. Off by default: skipping compression (and
        decoding it again for the pyramid) is noticeably faster, but the temporary
        store then takes the full uncompressed size of level 0 (``plan_fusion(...).nbytes``),
        so ``temp_dir`` needs room for it. Turn it on when disk space is short.
    progress
        Show progress bars.
    **write_options
        Any other :meth:`~zarrsmith.microscope_manager.MicroscopeManager.to_zarr`
        option, as for ``zarrsmith convert``: ``chunks``, ``shards``, ``shard_target_mb``,
        ``shard_exclude_axes``, ``zarr_format``, ``ngff_version``, ``compressor``,
        ``compressor_level``, ``drop_singleton``, ``num_workers``, ``storage_options``.
        ``chunks`` is given in the output axis order (``c, z, y, x``, or ``c, y, x``
        for projections, after a leading ``t`` for several timepoints).

    The display window of every channel (OMERO ``window`` metadata) is set to the
    0.5th-99.9th percentile of the fused values of all timepoints, ignoring blank
    areas, so viewers open the result with a usable contrast; see :func:`display_windows`.

    When ``tiles`` hold several timepoints, the root attribute ``zarrsmith_fusion``
    records the acquired timepoints of the output's ``t`` axis and the reference.

    Use :func:`plan_fusion` with the same arguments to get the output shape and
    size without reading or writing anything.

    ``tiles`` must come from one field (``Tile.field``, set by
    :class:`~zarrsmith.mosaic_manager.MosaicManager`; skipped tiles count): tiles of
    separate fields are not fused, a
    :class:`~zarrsmith.mosaic_manager.fields.MultipleFieldsError` is raised before
    anything is read or written.
    """
    from ..microscope_manager.array_manager import ArrayManager
    from ..microscope_manager.zarr_manager import ZarrManager

    if executor not in ("auto", "process", "thread"):
        raise ValueError("executor must be 'auto', 'process' or 'thread'.")
    if not sigmoid_steepness > 0:
        raise ValueError(f"sigmoid_steepness must be positive, got {sigmoid_steepness!r}.")
    _check_write_options(write_options)
    plan = plan_fusion(tiles, ds=ds, projection=projection, blend=blend, channels=channels,
                       crop=crop, skip=skip, out_dtype=out_dtype, midline_axis=midline_axis,
                       timepoints=timepoints, reference_timepoint=reference_timepoint)
    check_single_field([p.tile for ps in plan.layout.values() for p in ps])
    blend, out_dtype, names, ds = plan.blend, plan.dtype, plan.channels, plan.ds
    flat = projection is not None
    n_ch = len(names)
    for label, values in (("channel_names", channel_names), ("channel_colors", channel_colors)):
        if values is not None and len(values) != n_ch:
            raise ValueError(f"{label} has {len(values)} entries but {n_ch} channels are fused ({names}).")

    chunks = write_options.get("chunks")
    if chunks is None:
        chunks = ((1,) if "t" in plan.axes else ()) + ((1, 512, 512) if flat else (1, 64, 256, 256))
    chunks = tuple(int(c) for c in chunks)
    if len(chunks) != len(plan.axes):
        raise ValueError(f"chunks must have {len(plan.axes)} values for axes {plan.axes!r}, got {chunks}.")
    chunks = tuple(max(1, min(c, s)) for c, s in zip(chunks, plan.shape))
    logger.info("Fusing %d tiles: %s, blend=%s, projection=%s", len(plan.tiles), plan.summary(), blend, projection)

    out = Path(out)
    tmp = Path(temp_dir or out.parent) / f".{out.name}.level0.zarr"
    # An existing output is only replaced at the end, once the fusion has succeeded.
    refusal = output_refusal(out, overwrite=overwrite, sources=[tmp])
    if refusal is not None:
        raise FileExistsError(refusal)
    if tmp.exists():
        shutil.rmtree(tmp)

    try:
        # ----- 1. level 0, region by region, into a plain temporary store -----
        depth = plan.grid_shape[0]
        cz, cy, cx = _region_shape(plan.tiles, plan.grid_shape, ds, flat)
        level0 = zarr.create_array(str(tmp), shape=plan.shape, dtype=out_dtype,
                                   chunks=(1,) * (len(plan.axes) - len(plan.scale)) + (() if flat else (cz,)) + (cy, cx),
                                   fill_value=0, overwrite=True,
                                   compressors=_TEMP_COMPRESSORS if temp_compression else None)
        axis = "zyx".index(midline_axis)
        region = (depth if flat else cz, cy, cx)

        def layout_blocks(layout: Dict[str, List[_Placed]]) -> List[tuple]:
            return [b for ci, n in enumerate(names)
                    for b in _layout_blocks(ci, n, layout[n], tuple(plan.shape[-2:]), depth, region, blend, axis)]

        regions = _RegionSeries((lambda tp: layout_blocks(plan.layouts[tp])) if plan.per_timepoint
                                else layout_blocks(plan.layout), plan)
        logger.info("Fusing %d regions of up to %s voxels (z, y, x)", len(regions),
                    ("-", cy, cx) if flat else (cz, cy, cx))

        auto = executor == "auto"
        if auto:
            picklable = all(getattr(t.data, "picklable", False) for t in plan.tiles)
            executor = "process" if workers > 1 and picklable else "thread"
        options = dict(ds=ds, projection=projection, blend=blend, dtype=out_dtype,
                       steepness=float(sigmoid_steepness), workers=workers, progress=progress)
        try:
            samples = _fuse_level0(regions, level0, executor=executor, **options)
        except (BrokenProcessPool, RuntimeError) as e:
            if not (auto and executor == "process"):
                raise
            # e.g. worker processes cannot be spawned from this interpreter (some GUI
            # launchers); every region is rewritten as a whole, so redoing them is safe.
            logger.warning("Worker processes failed (%s); retrying with threads.", e)
            samples = _fuse_level0(regions, level0, executor="thread", **options)
        windows = _display_windows(samples, n_ch, out_dtype)

        # ----- 2. pyramid + OME-Zarr through the standard ZarrSmith writer -----
        metadata = build_metadata(
            size=[plan.shape],
            scales=[plan.scale],
            units=(plan.tiles[0].units,) * len(plan.scale),
            channels=[ChannelInfo(
                (channel_names or names)[i],
                None if channel_colors is None else channel_colors[i],
            ) for i in range(n_ch)],
            dtype=str(out_dtype),
            axes=plan.axes,
            chunksize=[chunks],
            name=name or "fused mosaic",
            data_type="intensity",
        )
        dataset = ArrayManager(da.from_zarr(str(tmp)).rechunk(chunks), metadata)
        dataset.build_pyramid(num_levels=max(1, int(levels)),
                              downscale_factor=_spatial_factor(downscale_factor, flat))
        dataset.to_zarr(str(out), overwrite=overwrite, progress=progress, **write_options)
        # Sensible display range for viewers (the writer's default is the whole dtype range).
        _write_display_windows(out, windows)
        logger.info("Display window per channel: %s", windows)
        if plan.n_available > 1:
            _write_provenance(out, plan)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return ZarrManager(str(out), mode="r")
