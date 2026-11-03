"""Content-based registration of mosaic tiles: correct stage positions from the image data.

Registration runs in three steps, on one **reference channel** at one **reference timepoint**;
the result is applied to every channel and timepoint:

1. **Pairs.** Tiles whose boxes share a face (within the search range) are paired, inside
   one field. Only the overlap of a pair is read, widened by the search range ``max_shift``.
2. **Measure.** The read per pair is bounded (``budget_mb``), since raw tiles have no pyramid
   and downsampling does not reduce what is read. A window that does not fit is sampled in
   contiguous, chunk-aligned **slabs** along z (scattered planes would decode the same
   chunks again and again). Small probes first locate the depths where both tiles have
   signal. Each slab is matched on its own, with phase correlation proposing shifts and
   normalized cross-correlation (NCC) on the real overlap choosing between them, and
   reading stops as soon as two slabs agree. A last pass matches a small part of the overlap
   at full resolution. Results are saved after every pair, so an interrupted run resumes
   without reading anything again.
3. **Solve.** Weighted least squares over the links of every **block** (the tiles of one group
   in one field), per axis. Links that disagree with the solution are dropped and the fit is
   repeated. Tiles without a reliable link follow the linear trend of the corrections of
   their block, or stay at the stage position when there is no trend. Thresholds can be
   changed and the fit repeated (:meth:`Registration.solve`) without reading any pixel.

Moving tiles is exact when all of them share one orientation, which also covers a stage
rotated against the camera and a wrong pixel size. Blocks of one field can differ, however:
the planes of the left and right illumination sides of a light-sheet microscope can be
tilted differently. A z offset between neighbours then comes from the tilt inside the tiles
and must not add up over the mosaic, or the blocks drift apart at their seam. So in a field
with several blocks, tiles keep their stage height and the tilt of their block, fitted on
its links, is removed inside them (:class:`TiltedArray`). The blocks are then placed against
each other with one offset, from all the links between them. In z that offset is measured
per channel, as the channels can differ between blocks.

Applying a registration only changes the tiles in memory, see
:meth:`MosaicManager.apply_registration
<zarrsmith.mosaic_manager.MosaicManager.apply_registration>`. Fusion then places the tiles
as usual: on whole output voxels, with no interpolation.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..utils.chromatic_shift import _subpixel_peak
from .fusion import _Placed, _read_block, _source_chunks
from .tiff_array import PlaneStackArray
from .tile import Tile, series_keys

logger = logging.getLogger("zarrsmith")

#: Version of the registration file. Version 1 had no fields and no tile centres in its links.
FILE_VERSION = 2

#: Default search range, as a fraction of the tile size per axis (z, y, x): stage errors of
#: up to 3% on each of two tiles give a relative error of up to 6% in y/x.
MAX_SHIFT = (0.02, 0.06, 0.06)
#: Default limit of raw data read per pair, from its two tiles together.
BUDGET_MB = 1024.0
#: Default lowest NCC for a measurement to be trusted. On real data, wrong shifts scored up
#: to 0.03 and right ones from 0.17.
MIN_NCC = 0.1
#: Default largest disagreement with the solution, in working voxels of the link.
MAX_RESIDUAL = 2.0

#: Tiles with fewer planes are matched in y/x only, on their maximum projection.
_MIN_Z_PLANES = 16
#: Voxels of the array matched per tile and slab (32 MB per float32 array): bounds the memory of a worker.
_WORK_VOXELS = 2**23
#: Chunks assumed for data that does not report them (e.g. HDF5 tiles opened lazily).
_DEFAULT_CHUNKS = (64, 64, 64)
#: Share of the budget of a pair spent on locating signal.
_SCOUT_SHARE = 0.1
#: Lowest :func:`_structure` score that counts as signal; pure noise scores about 1.
_MIN_STRUCTURE = 2.0
#: Phase correlation peaks checked with NCC.
_N_PEAKS = 8
#: Most steps of the NCC climb from the best peak.
_MAX_CLIMB = 8
#: Share of the smaller array that must overlap for a shift to be considered.
_MIN_OVERLAP = 0.1
#: Slabs agree when their shifts are this close, in working voxels.
_AGREEMENT = 2.0
#: Scale (working voxels) of the background removed before matching.
_HIGHPASS_SIGMA = 8.0
#: Weak pull of every tile towards its stage position, relative to a link of NCC 1.
_STAGE_PULL = 1e-3
#: Fewest linked tiles, and links, a block needs for a linear trend, and for a tilt.
_MIN_TREND_TILES = 4
_MIN_TILT_LINKS = 3
#: Links between two blocks measured in every other channel, for its z offset between them.
_CHANNEL_SEAMS = 4
#: Planes of the part of an overlap matched at full resolution.
_FINE_PLANES = 128
#: Attempts at measuring a pair, and seconds before the second one (doubled after that):
#: on a network drive a read now and then fails for no lasting reason.
_READ_ATTEMPTS = 4
_RETRY_SECONDS = 15.0


# ---------------------------------------------------------------------------
# tilted planes
# ---------------------------------------------------------------------------

class TiltedArray:
    """Lazy ``(z, y, x)`` view of a tile with the tilt of its planes removed.

    Every column ``(y, x)`` moves along z by a whole number of planes, the rounded
    ``-(tilt_y * u_y + tilt_x * u_x) / voxel_z``, with ``u`` its distance to the tile centre in
    sample units. Voxel values are not interpolated. Planes are added so that every column
    fits: :attr:`lead` before plane 0 of the tile, and voxels without data read as 0.
    The data is read only when sliced, and whole planes or chunks are read as for ``data``.
    """

    def __init__(self, data, voxel_size, tilt):
        self.data = data
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.tilt = tuple(float(v) for v in tilt)
        self.picklable = getattr(data, "picklable", False)
        self.dtype = np.dtype(data.dtype)
        self.ndim = 3
        nz, ny, nx = (int(n) for n in data.shape)
        corners = self._columns(slice(0, ny), slice(0, nx))
        low, high = int(corners.min()), int(corners.max())
        self.lead = -low  # planes before plane 0 of the tile
        self.shape = (nz + high - low, ny, nx)
        chunks = getattr(data, "chunks", None)
        if chunks is not None:
            self.chunks = chunks

    def _columns(self, ys: slice, xs: slice) -> np.ndarray:
        """Planes every column of the tile moves, as integers."""
        vz, vy, vx = self.voxel_size
        ny, nx = self.data.shape[1:]
        rows = (np.arange(ys.start, ys.stop)[:, None] - (ny - 1) / 2) * vy
        columns = (np.arange(xs.start, xs.stop)[None, :] - (nx - 1) / 2) * vx
        return np.rint(-(self.tilt[0] * rows + self.tilt[1] * columns) / vz).astype(int)

    def __getitem__(self, key) -> np.ndarray:
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (3 - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (3 - len(key))
        selection, dropped = [], []
        for axis, (k, n) in enumerate(zip(key, self.shape)):
            if isinstance(k, (int, np.integer)):
                k = int(k) + (n if k < 0 else 0)
                selection.append(slice(k, k + 1))
                dropped.append(axis)
            else:
                start, stop, step = k.indices(n)
                if step != 1:
                    raise IndexError("TiltedArray slices cannot have a step.")
                selection.append(slice(start, max(start, stop)))
        zs, ys, xs = selection
        out = np.zeros((zs.stop - zs.start, ys.stop - ys.start, xs.stop - xs.start), self.dtype)
        if out.size:
            moves = self._columns(ys, xs)
            depth = self.data.shape[0]
            for move in np.unique(moves):
                first = zs.start - self.lead - int(move)  # plane of the tile at the top of the selection
                lo, hi = max(first, 0), min(first + out.shape[0], depth)
                if hi <= lo:
                    continue
                mask = moves == move
                rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
                j0, j1, i0, i1 = rows[0], rows[-1] + 1, columns[0], columns[-1] + 1
                block = np.asarray(self.data[lo:hi, ys.start + j0:ys.start + j1, xs.start + i0:xs.start + i1])
                inside = mask[j0:j1, i0:i1]
                out[lo - first:hi - first, j0:j1, i0:i1][:, inside] = block[:, inside]
        return np.squeeze(out, axis=tuple(dropped)) if dropped else out

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"TiltedArray({self.data!r}, tilt={self.tilt}, shape={self.shape})"


# ---------------------------------------------------------------------------
# which tiles, which pairs
# ---------------------------------------------------------------------------

def position_id(key: tuple) -> str:
    """Name of a position from its :func:`~zarrsmith.mosaic_manager.tile.series_keys` key:
    ``"group|stack|rank"``, shared by its tiles in every channel and timepoint."""
    _, group, stack, rank = key
    return f"{'' if group is None else group}|{'' if stack is None else stack}|{rank}"


def _group_of(position: str) -> str:
    """Group in a :func:`position_id`."""
    return position.split("|", 1)[0]


def _max_shift_fractions(max_shift) -> Tuple[float, float, float]:
    """``(z, y, x)`` search range from one fraction of the tile size or three."""
    values = tuple(float(v) for v in np.ravel(max_shift))
    values = values * 3 if len(values) == 1 else values
    if len(values) != 3 or min(values) < 0:
        raise ValueError(f"max_shift must be one or three (z, y, x) fractions >= 0, got {max_shift!r}.")
    return values


def _reference_tiles(tiles: Sequence[Tile], channel: Optional[str], timepoint: Optional[int]):
    """``({position id: tile}, channel, timepoint)`` of the tiles registration reads."""
    tiles = list(tiles)
    if not tiles:
        raise ValueError("No tiles to register.")
    channels = list(dict.fromkeys(t.channel for t in tiles))
    timepoints = sorted({t.timepoint for t in tiles})
    channel = channels[0] if channel is None else channel
    timepoint = timepoints[0] if timepoint is None else int(timepoint)
    if channel not in channels:
        raise ValueError(f"Reference channel {channel!r} not found; available: {channels}.")
    if timepoint not in timepoints:
        raise ValueError(f"Reference timepoint {timepoint} not found; available: {timepoints}.")
    refs = {position_id(k): t for k, t in zip(series_keys(tiles), tiles)
            if t.channel == channel and t.timepoint == timepoint}
    if not refs:
        raise ValueError(f"No tile of channel {channel!r} at timepoint {timepoint}.")
    vox = next(iter(refs.values())).voxel_size
    if any(not np.allclose(t.voxel_size, vox, rtol=1e-3) for t in refs.values()):
        raise ValueError("Tiles must share one voxel size to be registered.")
    return refs, channel, timepoint


def _find_pairs(refs: Dict[str, Tile], fractions) -> List[Tuple[str, str]]:
    """Positions that share a face, within the search range, inside one field.

    A pair overlapping in less than half a tile on several axes meets at a corner: it
    holds little data and adds nothing to the faces that connect the same tiles.
    """
    ids = list(refs)
    if len(ids) < 2:
        return []
    lo = np.array([refs[i].extent[0] for i in ids])
    hi = np.array([refs[i].extent[1] for i in ids])
    size = hi - lo
    fields = np.array([str(refs[i].field) for i in ids])
    margin = size * np.array(fractions)
    pairs = []
    for i in range(len(ids) - 1):
        ov = np.minimum(hi[i], hi[i + 1:]) - np.maximum(lo[i], lo[i + 1:])
        reach = margin[i] + margin[i + 1:]
        side = np.minimum(size[i], size[i + 1:])
        keep = (np.all(ov + reach > 0, axis=1) & (np.sum(ov / side < 0.5, axis=1) <= 1)
                & (fields[i + 1:] == fields[i]))
        pairs += [(ids[i], ids[i + 1 + k]) for k in np.flatnonzero(keep)]
    return pairs


def _side_budget(budget_mb: Optional[float]) -> float:
    """Bytes that may be read from each tile of a pair: half the budget (no limit without one)."""
    if budget_mb is None:
        return math.inf
    return budget_mb * 1e6 / 2


# ---------------------------------------------------------------------------
# reading parts of a tile
# ---------------------------------------------------------------------------

def _box(a: Tile, b: Tile, shift: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` sample coordinates of the overlap of two tiles, widened by ``shift``."""
    lo = np.maximum(a.extent[0], b.extent[0]) - shift
    hi = np.minimum(a.extent[1], b.extent[1]) + shift
    return lo, hi


def _crop(t: Tile, lo, hi, ds) -> List[Tuple[int, int]]:
    """``(start, stop)`` per axis, in voxels of ``ds``, of the part of ``t`` inside a box."""
    out = []
    for o, v, n, d, a, b in zip(t.origin, t.voxel_size, t.shape, ds, lo, hi):
        start = max(0, int(math.floor((a - o) / v / d)))
        out.append((start, max(start, min(n // d, int(math.ceil((b - o) / v / d))))))
    return out


def _read_cost(t: Tile, crop, ds) -> float:
    """Bytes read for a ``crop``: whole planes for data stored per plane, whole chunks otherwise."""
    (z0, z1), (y0, y1), (x0, x1) = crop
    z = (z1 - z0) * ds[0]
    if isinstance(t.data, PlaneStackArray):
        return z * t.shape[1] * t.shape[2] * t.dtype.itemsize
    _, cy, cx = _source_chunks(t.data) or _DEFAULT_CHUNKS

    def spanned(a, b, d, c):
        return (math.ceil(b * d / c) - math.floor(a * d / c)) * c

    return z * spanned(y0, y1, ds[1], cy) * spanned(x0, x1, ds[2], cx) * t.dtype.itemsize


def _read(t: Tile, crop, ds, flat: bool) -> Tuple[np.ndarray, np.ndarray]:
    """``(array, sample position of its first voxel)`` of a crop, downsampled by ``ds``.

    The array has three axes also when ``flat``, where it is the maximum projection of
    the whole depth, with a z axis of size 1.
    """
    (z0, z1), (y0, y1), (x0, x1) = crop
    placed = _Placed(t, (0, 0, 0), tuple(n // d for n, d in zip(t.shape, ds)))
    block = _read_block((placed, (y0, y1, x0, x1), (z0, z1), ds, "max" if flat else None))
    origin = np.array(t.origin) + np.array([z0, y0, x0]) * np.array(ds) * np.array(t.voxel_size)
    return (block[None] if flat else block), origin


def _working_ds(shape, voxel, flat: bool) -> Tuple[int, int, int]:
    """Downsampling that brings a ``shape`` to at most ``_WORK_VOXELS``, halving the
    axis with the finest voxels first, so that the result gets closer to isotropic."""
    ds = [1, 1, 1]
    while math.prod(max(1, n // d) for n, d in zip(shape, ds)) > _WORK_VOXELS:
        axes = [i for i in range(3) if shape[i] // ds[i] > 16 and not (flat and i == 0)]
        if not axes:
            break
        i = min(axes, key=lambda k: voxel[k] * ds[k])
        ds[i] *= 2
    return tuple(ds)


# ---------------------------------------------------------------------------
# matching two arrays
# ---------------------------------------------------------------------------

def _highpass(arr: np.ndarray) -> np.ndarray:
    """Remove the slow background, which correlates at any shift (illumination, flat field)."""
    sigma = [_HIGHPASS_SIGMA if n > 1 else 0 for n in arr.shape]
    return arr - ndi.gaussian_filter(arr, sigma, mode="nearest")


def _taper(arr: np.ndarray) -> np.ndarray:
    """Fade the borders to 0, so that they do not produce a peak at no shift."""
    out = arr - arr.mean()
    for axis, n in enumerate(out.shape):
        if n < 4:
            continue
        edge = max(1, n // 8)
        ramp = np.ones(n, np.float32)
        fade = 0.5 - 0.5 * np.cos(np.pi * (np.arange(edge, dtype=np.float32) + 0.5) / edge)
        ramp[:edge], ramp[-edge:] = fade, fade[::-1]
        shape = [1] * out.ndim
        shape[axis] = n
        out = out * ramp.reshape(shape)
    return out


def _ncc_at(a: np.ndarray, b: np.ndarray, s: Sequence[int]) -> Tuple[float, int]:
    """``(NCC, voxels compared)`` of ``a[i + s]`` with ``b[i]``."""
    sa, sb = [], []
    for n, m, d in zip(a.shape, b.shape, s):
        start = max(0, -d)
        stop = min(m, n - d)
        if stop <= start:
            return 0.0, 0
        sb.append(slice(start, stop))
        sa.append(slice(start + d, stop + d))
    x, y = a[tuple(sa)], b[tuple(sb)]
    x, y = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(np.sum(x * x)) * float(np.sum(y * y)))
    return (float(np.sum(x * y)) / denom if denom > 0 else 0.0), x.size


def _match(a: np.ndarray, b: np.ndarray, expected, reach) -> Optional[dict]:
    """Best shift ``s`` such that ``a[i + s]`` shows what ``b[i]`` shows.

    ``expected`` is the ``s`` of the stage positions (not rounded) and ``reach`` how far,
    per axis, ``s`` may be from it, both in voxels. Returns ``{"s": subvoxel shift,
    "ncc": NCC at that shift, "ncc_stage": NCC at the stage positions}``, or ``None``
    when no allowed shift overlaps enough.
    """
    fa, fb = _highpass(a), _highpass(b)
    pad = [max(n, m) + int(math.ceil(r)) + int(math.ceil(abs(e))) for n, m, r, e in zip(a.shape, b.shape, reach, expected)]
    shape = [sfft.next_fast_len(p, real=True) if p > 1 else 1 for p in pad]
    cross = sfft.rfftn(_taper(fa), shape) * np.conj(sfft.rfftn(_taper(fb), shape))
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = sfft.irfftn(cross, shape)

    lows = [int(math.ceil(e - r)) for e, r in zip(expected, reach)]
    highs = [int(math.floor(e + r)) for e, r in zip(expected, reach)]
    if any(h < lo for h, lo in zip(highs, lows)):
        return None
    grid = np.ix_(*[np.arange(lo, h + 1) % n for lo, h, n in zip(lows, highs, shape)])
    allowed = corr[grid]
    peaks = (allowed == ndi.maximum_filter(allowed, size=3, mode="nearest")) & (allowed > 0)
    found = np.argwhere(peaks)
    if len(found) == 0:
        found = np.array([np.unravel_index(int(np.argmax(allowed)), allowed.shape)])
    order = np.argsort(allowed[tuple(found.T)])[::-1][:_N_PEAKS]
    candidates = [tuple(int(i) + lo for i, lo in zip(index, lows)) for index in found[order]]
    # the stage positions compete too, so a match is never worse than not moving at all
    candidates.append(tuple(min(h, max(lo, int(round(e)))) for e, lo, h in zip(expected, lows, highs)))

    enough = _MIN_OVERLAP * min(a.size, b.size)

    def score(s):
        ncc, voxels = _ncc_at(fa, fb, s)
        return ncc if voxels >= enough else -math.inf

    best = max(candidates, key=score)
    ncc = score(best)
    if ncc == -math.inf:
        return None

    # phase correlation only proposes: climb the NCC from the best proposal
    for _ in range(_MAX_CLIMB):
        steps = [tuple(v + (d if k == axis else 0) for k, v in enumerate(best))
                 for axis in range(3) if highs[axis] > lows[axis] for d in (-1, 1)]
        steps = [c for c in steps if all(lo <= v <= h for v, lo, h in zip(c, lows, highs))]
        top = max(steps, key=score, default=best)
        if score(top) <= ncc:
            break
        best, ncc = top, score(top)

    refined = []
    for axis, value in enumerate(best):
        if not lows[axis] < value < highs[axis]:
            refined.append(float(value))
            continue
        curve = np.array([score(tuple(v + (d if k == axis else 0) for k, v in enumerate(best))) for d in (-1, 0, 1)])
        position, at_edge = _subpixel_peak(curve) if np.all(np.isfinite(curve)) else (1.0, True)
        refined.append(float(value) if at_edge else value + position - 1)
    stage, _ = _ncc_at(fa, fb, [int(round(e)) for e in expected])
    return {"s": refined, "ncc": ncc, "ncc_stage": stage}


def _structure(block: np.ndarray) -> float:
    """How far a block is from pure noise: about 1 for noise, higher with structure.

    The noise comes from the differences of adjacent voxels, which structure hardly
    affects. Averaging 2 x 2 x 2 voxels divides the spread of white noise by sqrt(8) and
    leaves structure, so the score is the spread of the averages over what noise alone
    would give. The histogram alone cannot tell: textured tissue and noise can look the same.
    """
    values = np.asarray(block, np.float32)
    steps = np.diff(values, axis=-1)
    noise = 1.4826 * float(np.median(np.abs(steps - np.median(steps)))) / math.sqrt(2)
    pool = tuple(s - s % 2 for s in values.shape)
    if min(pool) < 2:
        return math.inf  # too thin to tell: read it
    cropped = values[:pool[0], :pool[1], :pool[2]]
    averaged = cropped.reshape(pool[0] // 2, 2, pool[1] // 2, 2, pool[2] // 2, 2).mean(axis=(1, 3, 5))
    return float(averaged.std()) * math.sqrt(8) / max(noise, 1e-6)


# ---------------------------------------------------------------------------
# one pair
# ---------------------------------------------------------------------------

@dataclass
class _PairJob:
    a: str
    b: str
    tile_a: Tile
    tile_b: Tile
    fractions: Tuple[float, float, float]
    side_budget: float
    min_ncc: float
    ds: Optional[Tuple[int, int, int]] = None
    channel: Optional[str] = None  # set for the z offset of another channel than the reference


def _slab_depths(job: _PairJob, lo, hi, reach_planes: int, budget: float) -> Tuple[int, List[Tuple[int, int]]]:
    """``(planes per slab, candidate slabs as z ranges of tile a)`` covering the box in depth."""
    a = job.tile_a
    (z0, z1), (y0, y1), (x0, x1) = _crop(a, lo, hi, (1, 1, 1))
    one_plane = _read_cost(a, ((0, 1), (y0, y1), (x0, x1)), (1, 1, 1))
    planes = z1 - z0
    if planes * one_plane <= budget:
        return planes, [(z0, z1)]
    chunk = (_source_chunks(a.data) or _DEFAULT_CHUNKS)[0]
    available = int(budget // one_plane)
    # thick enough for the z search, as thin as the chunks allow otherwise
    wanted = max(4 * chunk, 3 * reach_planes)
    thickness = min(wanted, available)
    if chunk > 1:
        thickness = max(chunk, thickness - thickness % chunk)
    start = z0 + (-z0 % chunk)
    return thickness, [(z, z + thickness) for z in range(start, z1 - thickness + 1, thickness)]


def _scout(job: _PairJob, slabs, lo, hi, budget: float) -> Tuple[List[Tuple[int, int]], List[float], float]:
    """Order ``slabs`` by the signal both tiles show there, from small probes.

    Returns ``(slabs, their scores, bytes read)``. Data stored per plane is not probed, since
    a probe reads whole planes: its slabs are taken from the middle of the depth outwards.
    """
    a, b = job.tile_a, job.tile_b
    centre = (slabs[0][0] + slabs[-1][1]) / 2
    if isinstance(a.data, PlaneStackArray) or isinstance(b.data, PlaneStackArray):
        ordered = sorted(slabs, key=lambda s: abs((s[0] + s[1]) / 2 - centre))
        return ordered, [math.inf] * len(ordered), 0.0
    cz, cy, cx = _source_chunks(a.data) or _DEFAULT_CHUNKS
    block_bytes = cz * cy * cx * a.dtype.itemsize
    # every depth gets a probe, or signal only found at some depths could be missed
    per_depth = min(4, max(1, int(budget * _SCOUT_SHARE // (block_bytes * len(slabs)))))

    # along the longer side of the overlap, at the middle of the shorter one
    _, (ay0, ay1), (ax0, ax1) = _crop(a, lo, hi, (1, 1, 1))
    along_y = (ay1 - ay0) >= (ax1 - ax0)
    span = (ay0, ay1) if along_y else (ax0, ax1)
    other = (ax0 + ax1) // 2 if along_y else (ay0 + ay1) // 2
    spots = np.linspace(span[0], span[1], per_depth + 2)[1:-1].astype(int)

    spent = 0.0
    scores = {}
    for z0, z1 in slabs:
        z = (z0 + z1) // 2
        values = []
        for spot in spots:
            y, x = (spot, other) if along_y else (other, spot)
            world = np.array(a.origin) + np.array([z, y, x]) * np.array(a.voxel_size)
            pair = []
            for t in (a, b):
                start = np.rint((world - np.array(t.origin)) / np.array(t.voxel_size)).astype(int)
                start = [max(0, min(int(v), n - c)) for v, n, c in zip(start, t.shape, (cz, cy, cx))]
                pair.append(_structure(t.data[start[0]:start[0] + cz, start[1]:start[1] + cy, start[2]:start[2] + cx]))
                spent += block_bytes
            values.append(min(pair))
        scores[(z0, z1)] = float(np.mean(values))

    ordered = sorted(slabs, key=scores.get, reverse=True)
    return ordered, [scores[s] for s in ordered], spent


def _agreeing(estimates: List[dict], min_ncc: float) -> List[dict]:
    """The largest group of trusted estimates with close shifts; the best estimate breaks ties."""
    trusted = [e for e in estimates if e["ncc"] >= min_ncc]
    best: List[dict] = []
    for e in trusted:
        group = [o for o in trusted
                 if max(abs(p - q) / v for p, q, v in zip(o["shift"], e["shift"], e["voxel"])) <= _AGREEMENT]
        if len(group) > len(best) or (len(group) == len(best) and group
                                      and max(o["ncc"] for o in group) > max(o["ncc"] for o in best)):
            best = group
    return best


def _measure_pair(job: _PairJob) -> dict:
    """Measure the shift of one pair (runs in the workers), trying again after a failed
    read. Returns one entry of ``links``."""
    for attempt in range(_READ_ATTEMPTS):
        try:
            return _measure_once(job)
        except OSError as e:
            if attempt == _READ_ATTEMPTS - 1:
                raise
            logger.warning("Read failed for pair %s - %s, trying again: %s", job.a, job.b, e)
            time.sleep(_RETRY_SECONDS * 2**attempt)


def _measure_once(job: _PairJob) -> dict:
    a, b = job.tile_a, job.tile_b
    voxel = np.array(a.voxel_size)
    size = np.minimum(np.array(a.shape) * voxel, np.array(b.shape) * voxel)
    flat = min(a.shape[0], b.shape[0]) < _MIN_Z_PLANES
    limit = size * np.array(job.fractions)
    if flat:
        limit[0] = 0.0
    lo, hi = _box(a, b, limit)
    if flat:
        lo[0], hi[0] = min(a.origin[0], b.origin[0]), max(a.extent[1][0], b.extent[1][0])
    centre = [(np.array(t.origin) + (np.array(t.shape) - 1) / 2 * voxel).tolist() for t in (a, b)]
    nominal = _box(a, b, np.zeros(3))
    link: Dict[str, Any] = {"a": job.a, "b": job.b, "size": [float(v) for v in size], "flat": flat,
                            "centres": centre, "overlap": ((nominal[0] + nominal[1]) / 2).tolist(),
                            "estimates": [], "read_mb": 0.0, "slabs_tried": 0, "slabs_available": 1, "signal": True}
    if job.channel is not None:
        link["channel"] = job.channel
    spent = 0.0

    if flat:
        slabs = [(0, a.shape[0])]
        scores = [math.inf]
    else:
        reach_planes = int(math.ceil(limit[0] / voxel[0]))
        _, slabs = _slab_depths(job, lo, hi, reach_planes, job.side_budget)
        link["slabs_available"] = len(slabs)
        scores = [math.inf]
        if len(slabs) > 1:
            slabs, scores, spent = _scout(job, slabs, lo, hi, job.side_budget)
            if scores[0] < _MIN_STRUCTURE:
                link["signal"] = False
                link["read_mb"] = spent / 1e6
                return link
        else:
            scores = scores * len(slabs)

    for (z0, z1), score in zip(slabs, scores):
        if score < _MIN_STRUCTURE:
            break  # slabs are ordered: the others have no signal either
        slab_lo, slab_hi = lo.copy(), hi.copy()
        if not flat:
            slab_lo[0] = a.origin[0] + z0 * voxel[0]
            slab_hi[0] = a.origin[0] + z1 * voxel[0]
        full_a, full_b = _crop(a, slab_lo, slab_hi, (1, 1, 1)), _crop(b, slab_lo, slab_hi, (1, 1, 1))
        if flat:
            full_a[0], full_b[0] = (0, a.shape[0]), (0, b.shape[0])
        cost = max(_read_cost(a, full_a, (1, 1, 1)), _read_cost(b, full_b, (1, 1, 1)))
        if link["slabs_tried"] and spent / 2 + cost > job.side_budget:
            break  # spent counts both tiles, the budget one
        widest = [max(p[1] - p[0], q[1] - q[0]) for p, q in zip(full_a, full_b)]
        ds = job.ds or _working_ds(widest, voxel, flat)
        if flat:
            ds = (1,) + tuple(ds[1:])
        crop_a, crop_b = _crop(a, slab_lo, slab_hi, ds), _crop(b, slab_lo, slab_hi, ds)
        if flat:
            crop_a[0], crop_b[0] = (0, a.shape[0]), (0, b.shape[0])
        if min(p[1] - p[0] for p in crop_a + crop_b) < 1:
            break
        arr_a, wa = _read(a, crop_a, ds, flat)
        arr_b, wb = _read(b, crop_b, ds, flat)
        spent += _read_cost(a, crop_a, ds) + _read_cost(b, crop_b, ds)
        link["slabs_tried"] += 1
        work = voxel * np.array(ds)
        expected = (wb - wa) / work
        reach = limit / work
        if flat:
            expected[0], reach[0], wa[0], wb[0] = 0.0, 0.0, 0.0, 0.0
        found = _match(arr_a, arr_b, expected, reach)
        if found is None:
            continue
        # moving b by this makes b[i] land on a[i + s]
        shift = (wa - wb) + np.array(found["s"]) * work
        link["estimates"].append({"shift": [float(v) for v in shift], "voxel": [float(v) for v in work],
                                  "ncc": found["ncc"], "ncc_stage": found["ncc_stage"],
                                  "z": [int(z0), int(z1)], "structure": None if math.isinf(score) else float(score)})
        if len(_agreeing(link["estimates"], job.min_ncc)) >= 2:
            break
    if job.channel is None and job.ds is None:
        spent += _match_finely(job, link, nominal, voxel, flat, spent)
    link["read_mb"] = spent / 1e6
    return link


def _match_finely(job: _PairJob, link: dict, nominal, voxel, flat: bool, spent: float) -> float:
    """Match a small part of the overlap at full resolution, near the shift already found.

    Only when that shift was measured on downsampled images, and if the budget allows it.
    Sets ``link["fine"]`` and returns the bytes read.
    """
    group = _agreeing(link["estimates"], job.min_ncc)
    if not group or all(v <= u * 1.001 for v, u in zip(group[0]["voxel"][1:], voxel[1:])):
        return 0.0
    a, b = job.tile_a, job.tile_b
    best = max(group, key=lambda e: e["ncc"])
    coarse = np.array(best["voxel"])
    lo, hi = nominal[0].copy(), nominal[1].copy()
    planes = min(_FINE_PLANES, best["z"][1] - best["z"][0])
    middle = a.origin[0] + (best["z"][0] + best["z"][1]) / 2 * voxel[0]
    if not flat:
        lo[0], hi[0] = middle - planes / 2 * voxel[0], middle + planes / 2 * voxel[0]
    else:
        lo[0], hi[0] = min(a.origin[0], b.origin[0]), max(a.extent[1][0], b.extent[1][0])
    # the whole width across the seam, as much of the length as fits, from its middle
    along = 1 if hi[1] - lo[1] >= hi[2] - lo[2] else 2
    across = 3 - along
    wide = (hi[across] - lo[across]) / voxel[across]
    length = min((hi[along] - lo[along]) / voxel[along], _WORK_VOXELS / max(planes if not flat else 1, 1) / max(wide, 1))
    centre = (lo[along] + hi[along]) / 2
    lo[along], hi[along] = centre - length / 2 * voxel[along], centre + length / 2 * voxel[along]
    ds = (1, 1, 1)
    crop_a, crop_b = _crop(a, lo, hi, ds), _crop(b, lo, hi, ds)
    if flat:
        crop_a[0], crop_b[0] = (0, a.shape[0]), (0, b.shape[0])
    if min(p[1] - p[0] for p in crop_a + crop_b) < 1:
        return 0.0
    cost = _read_cost(a, crop_a, ds) + _read_cost(b, crop_b, ds)
    if (spent + cost) / 2 > job.side_budget:
        return 0.0
    arr_a, wa = _read(a, crop_a, ds, flat)
    arr_b, wb = _read(b, crop_b, ds, flat)
    if flat:
        wa[0], wb[0] = 0.0, 0.0
    expected = (np.array(best["shift"]) - (wa - wb)) / voxel
    reach = _AGREEMENT * coarse / voxel
    if flat:
        expected[0], reach[0] = 0.0, 0.0
    found = _match(arr_a, arr_b, expected, reach)
    if found is not None and found["ncc"] >= job.min_ncc:
        shift = (wa - wb) + np.array(found["s"]) * voxel
        link["fine"] = {"shift": [float(v) for v in shift], "ncc": found["ncc"]}
    return cost


# ---------------------------------------------------------------------------
# solving
# ---------------------------------------------------------------------------

def _link_value(link: dict, min_ncc: float) -> Tuple[Optional[dict], str]:
    """``(combined measurement, "")`` of a link, or ``(None, why it cannot be used)``."""
    if not link["signal"]:
        return None, "no signal"
    estimates = link["estimates"]
    if not estimates:
        return None, "no measurement"
    group = _agreeing(estimates, min_ncc)
    if not group:
        return None, f"low NCC ({max(e['ncc'] for e in estimates):.2f})"
    if len(group) == 1 and len(estimates) > 1 and sum(e["ncc"] >= min_ncc for e in estimates) > 1:
        return None, "slabs disagree"
    weights = np.array([e["ncc"] for e in group])
    shift = np.average([e["shift"] for e in group], axis=0, weights=weights)
    fine = link.get("fine")
    if fine is not None and fine["ncc"] >= min_ncc:
        shift = np.array(fine["shift"])
    return {"shift": shift, "voxel": np.array(group[0]["voxel"]), "ncc": float(np.mean(weights)),
            "ncc_stage": float(np.mean([e["ncc_stage"] for e in group]))}, ""


def _least_squares(ids: Sequence[str], used: List[dict], targets: Optional[np.ndarray] = None) -> np.ndarray:
    """Corrections ``(n, 3)`` from links ``x[b] - x[a] = shift``, weighted by their NCC, with a
    weak pull to ``targets`` (default 0), which also places tiles without links."""
    index = {name: i for i, name in enumerate(ids)}
    n, m = len(ids), len(used)
    targets = np.zeros((n, 3)) if targets is None else targets
    out = targets.copy()
    if not m:
        return out
    rows, cols, values = [], [], []
    for k, u in enumerate(used):
        rows += [k, k]
        cols += [index[u["a"]], index[u["b"]]]
        values += [-1.0, 1.0]
    design = sparse.csr_matrix((values, (rows, cols)), shape=(m, n))
    weights = sparse.diags([u["ncc"] for u in used])
    normal = (design.T @ weights @ design + _STAGE_PULL * sparse.identity(n)).tocsc()
    wanted = np.array([u["shift"] for u in used]) - design @ targets
    for axis in range(3):
        out[:, axis] += spsolve(normal, design.T @ (weights @ wanted[:, axis]))
    return out


def _fit_block(ids: Sequence[str], used: List[dict], stage: np.ndarray, max_residual: float, outcome: dict,
               trend: bool) -> np.ndarray:
    """Corrections of one block, dropping from ``used`` the link furthest from the solution until
    all are within ``max_residual`` working voxels. With ``trend``, tiles without a link follow
    the linear trend of the others, if there are enough of them to fit it."""
    index = {name: i for i, name in enumerate(ids)}
    corrections = _least_squares(ids, used)

    def residuals(values):
        return [float(np.max(np.abs(values[index[u["b"]]] - values[index[u["a"]]] - u["shift"]) / u["voxel"])) for u in used]

    while used:
        distances = residuals(corrections)
        worst = int(np.argmax(distances))
        if distances[worst] <= max_residual:
            break
        dropped = used.pop(worst)
        outcome[dropped["key"]] = {"status": "rejected", "reason": f"disagrees with the others ({distances[worst]:.1f} voxels)"}
        corrections = _least_squares(ids, used)
    linked = sorted({index[name] for u in used for name in (u["a"], u["b"])})
    if trend and len(linked) >= _MIN_TREND_TILES:
        design = np.c_[stage[:, 1:], np.ones(len(ids))]
        if np.linalg.matrix_rank(design[linked]) == 3:
            coefficients = np.linalg.lstsq(design[linked], corrections[linked], rcond=None)[0]
            corrections = _least_squares(ids, used, design @ coefficients)
    return corrections


def _fit_tilt(used: List[dict], max_residual: float) -> Tuple[float, float]:
    """``(z per sample unit of y, of x)`` of the planes of a block, from its links: neighbours
    meet with a z offset equal to the tilt times the distance between their centres."""
    moved = [u for u in used if not u["flat"]]
    while len(moved) >= _MIN_TILT_LINKS:
        distance = np.array([np.subtract(u["centres"][1], u["centres"][0])[1:] for u in moved])
        offset = np.array([u["shift"][0] for u in moved])
        weight = np.sqrt([u["ncc"] for u in moved])
        tilt = np.linalg.lstsq(distance * weight[:, None], offset * weight, rcond=None)[0]
        away = np.abs(distance @ tilt - offset) / np.array([u["voxel"][0] for u in moved])
        if away.max() <= max_residual:
            return float(tilt[0]), float(tilt[1])
        moved.pop(int(np.argmax(away)))
    return 0.0, 0.0


def _without_tilt(value: dict, tilt_a, tilt_b) -> float:
    """z offset of a link once both tiles have their tilt removed, at the middle of their overlap."""
    a, b = (np.subtract(value["overlap"], c)[1:] for c in value["centres"])
    return float(value["shift"][0] - np.dot(tilt_a, a) + np.dot(tilt_b, b))


def _place_blocks(blocks: List[str], across: List[dict], max_residual: float, outcome: dict):
    """``({block: offset (z, y, x)}, links used)``, relative to the first block, from the median of
    the links between each two blocks. A block without links stays where it is."""
    offsets = {blocks[0]: np.zeros(3)}
    used: List[dict] = []
    todo = list(blocks[1:])
    while todo:
        step = None
        for block in todo:
            for known in offsets:
                seam = [u for u in across if {u["block_a"], u["block_b"]} == {known, block}]
                if seam:
                    step = (block, known, seam)
                    break
            if step:
                break
        if step is None:
            for block in todo:
                offsets[block] = np.zeros(3)
            break
        block, known, seam = step
        todo.remove(block)
        rest = np.array([u["residual"] * (1 if u["block_b"] == block else -1) for u in seam])
        middle = np.median(rest, axis=0)
        offsets[block] = offsets[known] + middle
        for u, r in zip(seam, rest):
            away = float(np.max(np.abs(r - middle) / u["voxel"]))
            if away > max_residual:
                outcome[u["key"]] = {"status": "rejected", "reason": f"disagrees with the others ({away:.1f} voxels)"}
            else:
                outcome[u["key"]] = {"status": "used", "ncc": u["ncc"], "ncc_stage": u["ncc_stage"], "residual": away}
                used.append(u)
    return offsets, used


def _solve(positions: Dict[str, List[float]], fields: Dict[str, str], links: List[dict], channel_links: Dict[str, List[dict]],
           reference: str, *, min_ncc: float, max_shift, max_residual: float):
    """Fit a whole registration: ``(shifts, outcome, tilts, z offsets, notes)``, see :class:`Registration`."""
    ids = list(positions)
    limits = _max_shift_fractions(max_shift)
    block_of = {name: f"{fields.get(name, '')}|{_group_of(name)}" for name in ids}
    outcome: Dict[Tuple[str, str], dict] = {}

    def usable(link, record=True):
        value, why = _link_value(link, min_ncc)
        if value is not None and np.any(np.abs(value["shift"]) > np.array(link["size"]) * limits + value["voxel"]):
            value, why = None, "shift above max_shift"
        if value is None:
            if record:
                outcome[(link["a"], link["b"])] = {"status": "rejected", "reason": why}
            return None
        return {"a": link["a"], "b": link["b"], "key": (link["a"], link["b"]), "flat": link["flat"],
                "centres": link.get("centres"), "overlap": link.get("overlap"),
                "block_a": block_of[link["a"]], "block_b": block_of[link["b"]], **value}

    values = [v for v in (usable(link) for link in links) if v is not None]
    corrections = {name: np.zeros(3) for name in ids}
    tilts: Dict[str, Tuple[float, float]] = {}
    z_offsets: Dict[str, Dict[str, float]] = {}
    notes: List[str] = []
    for field in sorted({fields.get(name, "") for name in ids}):
        members = [name for name in ids if fields.get(name, "") == field]
        blocks = sorted({block_of[name] for name in members})
        chosen = [v for v in values if v["a"] in set(members)]
        several = len(blocks) > 1
        if several and any(not v["flat"] and v["centres"] is None for v in chosen):
            raise ValueError("This registration was made by an earlier version, which cannot place the groups of "
                             "a mosaic against each other: run the registration again with a new file.")
        for block in blocks:
            inside = [v for v in chosen if v["block_a"] == v["block_b"] == block]
            tilt = _fit_tilt(inside, max_residual) if several else (0.0, 0.0)
            if tilt != (0.0, 0.0):
                tilts[block] = tilt
                for v in inside:  # what remains of the z offsets once the tilt is taken out of the tiles
                    v["shift"] = v["shift"].copy()
                    v["shift"][0] = _without_tilt(v, tilt, tilt)
            names = [name for name in members if block_of[name] == block]
            stage = np.array([positions[name] for name in names])
            fitted = _fit_block(names, inside, stage, max_residual, outcome, trend=bool(fields))
            for name, c in zip(names, fitted):
                corrections[name] = c
            for u in inside:
                found = corrections[u["b"]] - corrections[u["a"]]
                outcome[u["key"]] = {"status": "used", "ncc": u["ncc"], "ncc_stage": u["ncc_stage"],
                                     "residual": float(np.max(np.abs(found - u["shift"]) / u["voxel"]))}
        if not several:
            continue

        def seam_links(items):
            seam = [v for v in items if v["block_a"] != v["block_b"]]
            for v in seam:
                v["shift"] = v["shift"].copy()
                if not v["flat"]:
                    v["shift"][0] = _without_tilt(v, tilts.get(v["block_a"], (0.0, 0.0)), tilts.get(v["block_b"], (0.0, 0.0)))
                v["residual"] = v["shift"] - (corrections[v["b"]] - corrections[v["a"]])
            return seam

        offsets, _ = _place_blocks(blocks, seam_links(chosen), max_residual, outcome)
        for name in members:  # y and x hold for all channels; z is kept per channel, below
            corrections[name] = corrections[name] + offsets[block_of[name]] * [0, 1, 1]
        for block in blocks:
            z_offsets.setdefault(block, {})[reference] = float(offsets[block][0])
        for channel, measured in channel_links.items():
            items = [v for v in (usable(link, record=False) for link in measured) if v is not None and v["a"] in set(members)]
            seam = seam_links(items)
            for u in seam:  # y and x of the other block are in the corrections by now
                u["residual"] = u["residual"] * [1, 0, 0]
            found, _ = _place_blocks(blocks, seam, math.inf, {})
            missing = [block for block in blocks[1:] if not any(block in (u["block_a"], u["block_b"]) for u in seam)]
            for block in blocks:
                z_offsets[block][channel] = float(offsets[block][0] if block in missing else found[block][0])
            if missing:
                notes.append(f"Channel {channel!r} has no trusted match between blocks {', '.join(missing)}: "
                             f"it uses the z offset of {reference!r}.")
    return corrections, outcome, tilts, z_offsets, notes


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class Registration:
    """Measurements of a registration and the corrections solved from them.

    Created by :func:`register_tiles`, or read back with :meth:`load`. ``shifts`` maps every
    position (:func:`position_id`) to its ``(z, y, x)`` correction in sample units, and
    :meth:`placement` gives what a tile of a position gets in one channel. In a field with
    several blocks (the tiles of one group), ``tilts`` has the ``(z per unit of y, of x)`` of the
    planes of a block and ``z_offsets`` the z offset of a block per channel; a block is named
    ``"field|group"``. Nothing changes on the tiles until it is given to
    :meth:`MosaicManager.apply_registration <zarrsmith.mosaic_manager.MosaicManager.apply_registration>`.
    """

    positions: Dict[str, List[float]]  # stage position of every position, at the reference timepoint
    voxel_size: Tuple[float, float, float]
    units: str
    reference_channel: str
    reference_timepoint: int
    settings: Dict[str, Any]
    links: List[dict] = _dc_field(default_factory=list)
    fields: Dict[str, str] = _dc_field(default_factory=dict)  # field of every position
    channel_links: Dict[str, List[dict]] = _dc_field(default_factory=dict)  # measured in the other channels
    shifts: Dict[str, List[float]] = _dc_field(default_factory=dict)
    tilts: Dict[str, Tuple[float, float]] = _dc_field(default_factory=dict)
    z_offsets: Dict[str, Dict[str, float]] = _dc_field(default_factory=dict)
    notes: List[str] = _dc_field(default_factory=list)
    outcome: Dict[Tuple[str, str], dict] = _dc_field(default_factory=dict)

    @property
    def n_used(self) -> int:
        """Links used in the solution."""
        return sum(o["status"] == "used" for o in self.outcome.values())

    def _block(self, position: str) -> str:
        return f"{self.fields.get(position, '')}|{_group_of(position)}"

    def placement(self, position: str, channel: str) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]]]:
        """``(shift (z, y, x), tilt or None)`` of the tile of ``position`` in ``channel``; the
        shift is ``None`` for a position this registration does not know. A channel it did
        not measure gets the z offset of the reference channel."""
        shift = self.shifts.get(position)
        if shift is None:
            return None, None
        block = self._block(position)
        offsets = self.z_offsets.get(block, {})
        z = offsets.get(channel, offsets.get(self.reference_channel, 0.0))
        return np.array(shift) + [z, 0, 0], self.tilts.get(block)

    def solve(self, *, min_ncc: Optional[float] = None, max_shift=None, max_residual: Optional[float] = None) -> "Registration":
        """Fit the corrections from the saved measurements; reads no pixel data.

        Parameters left out keep their value. ``max_shift`` can only reject more than was
        searched when measuring. Call :meth:`save` to keep the result.
        """
        for key, value in (("min_ncc", min_ncc), ("max_residual", max_residual),
                           ("max_shift", None if max_shift is None else list(_max_shift_fractions(max_shift)))):
            if value is not None:
                self.settings[key] = value
        corrections, self.outcome, self.tilts, self.z_offsets, self.notes = _solve(
            self.positions, self.fields, self.links, self.channel_links, self.reference_channel,
            min_ncc=self.settings["min_ncc"], max_shift=self.settings["max_shift"],
            max_residual=self.settings["max_residual"])
        self.shifts = {name: [float(v) for v in corrections[name]] for name in self.positions}
        return self

    def summary(self) -> str:
        """Report: links used and rejected (with the reasons), size of the corrections, reads."""
        lines = [f"{len(self.positions)} positions, {len(self.links)} links, {self.n_used} used "
                 f"(reference channel {self.reference_channel!r}, timepoint {self.reference_timepoint})"]
        measured = self.links + [link for items in self.channel_links.values() for link in items]
        read = sum(link["read_mb"] for link in measured)
        lines.append(f"Read {read / 1e3:.1f} GB in {sum(link['slabs_tried'] for link in measured)} slab(s)")
        if self.shifts:
            values = np.abs(np.array(list(self.shifts.values()))) / np.array(self.voxel_size)
            lines.append("Corrections, voxels (z, y, x): median "
                         + ", ".join(f"{v:.1f}" for v in np.median(values, axis=0)) + "; largest "
                         + ", ".join(f"{v:.1f}" for v in np.max(values, axis=0)))
        used = [o for o in self.outcome.values() if o["status"] == "used"]
        if used:
            lines.append(f"NCC of the used links: median {np.median([o['ncc'] for o in used]):.3f} "
                         f"(at the stage positions {np.median([o['ncc_stage'] for o in used]):.3f}); largest "
                         f"disagreement {max(o['residual'] for o in used):.2f} working voxels")
        reasons: Dict[str, List[str]] = {}
        for (a, b), o in self.outcome.items():
            if o["status"] != "used":
                reasons.setdefault(o["reason"].split(" (")[0], []).append(f"{a} - {b}")
        for reason, names in reasons.items():
            lines.append(f"Rejected, {reason}: {len(names)} link(s): " + ", ".join(names[:6])
                         + (", ..." if len(names) > 6 else ""))
        linked = {name for key, o in self.outcome.items() if o["status"] == "used" for name in key}
        alone = [name for name in self.positions if name not in linked]
        if alone:
            lines.append(f"{len(alone)} position(s) without a used link follow the trend of their block, or stay at "
                         "the stage position: " + ", ".join(alone[:6]) + (", ..." if len(alone) > 6 else ""))
        for block, (tilt_y, tilt_x) in self.tilts.items():
            lines.append(f"Tilt of block {block}: {tilt_y * 100:+.2f}% along y, {tilt_x * 100:+.2f}% along x")
        for block, offsets in self.z_offsets.items():
            if any(offsets.values()):
                lines.append(f"z offset of block {block}, planes: " + ", ".join(
                    f"{channel} {value / self.voxel_size[0]:+.1f}" for channel, value in offsets.items()))
        lines += self.notes
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Write the registration as JSON; a crash while writing leaves the previous file."""
        path = Path(path)
        content = {"version": FILE_VERSION, "positions": self.positions, "voxel_size": list(self.voxel_size),
                   "units": self.units, "reference_channel": self.reference_channel,
                   "reference_timepoint": self.reference_timepoint, "settings": self.settings,
                   "fields": self.fields, "links": self.links, "channel_links": self.channel_links,
                   "shifts": self.shifts, "tilts": self.tilts, "z_offsets": self.z_offsets}
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(content, indent=1), encoding="utf-8")
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str | Path) -> "Registration":
        """Read a file written by :meth:`save`, also one of an unfinished run (without ``shifts``)."""
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        if content.get("version") not in (1, FILE_VERSION):
            raise ValueError(f"{path} has registration file version {content.get('version')!r}, "
                             f"expected {FILE_VERSION}.")
        reg = cls(positions=content["positions"], voxel_size=tuple(content["voxel_size"]), units=content["units"],
                  reference_channel=content["reference_channel"], reference_timepoint=content["reference_timepoint"],
                  settings=content["settings"], links=content["links"], fields=content.get("fields", {}),
                  channel_links=content.get("channel_links", {}), shifts=content["shifts"])
        if reg.shifts:
            reg.solve()
        return reg


# ---------------------------------------------------------------------------
# planning and running
# ---------------------------------------------------------------------------

@dataclass
class RegistrationPlan:
    """What a registration would read, computed without reading any pixel data."""

    positions: int
    pairs: List[Tuple[str, str]]
    reference_channel: str
    reference_timepoint: int
    fractions: Tuple[float, float, float]
    budget_mb: Optional[float]
    full_gb: float  # both tiles of every pair, whole windows
    limit_gb: float  # the same, within the budget
    channel_seams: int = 0  # links between blocks also measured in the other channels

    def summary(self) -> str:
        """One line with the number of pairs and the data read at most."""
        budget = "no budget" if self.budget_mb is None else f"budget {self.budget_mb:.0f} MB per pair"
        seams = f", with {self.channel_seams} seams in other channels" if self.channel_seams else ""
        return (f"{self.positions} positions, {len(self.pairs)} pairs in channel {self.reference_channel!r} at "
                f"timepoint {self.reference_timepoint}: reads at most {self.limit_gb:.1f} GB{seams} ({budget}; "
                f"{self.full_gb:.1f} GB for whole overlaps)")


def plan_registration(
    tiles: Sequence[Tile],
    *,
    reference_channel: Optional[str] = None,
    reference_timepoint: Optional[int] = None,
    max_shift=MAX_SHIFT,
    budget_mb: Optional[float] = BUDGET_MB,
) -> RegistrationPlan:
    """Pairs and the most data :func:`register_tiles` would read with these arguments; reads no pixels."""
    fractions = _max_shift_fractions(max_shift)
    refs, channel, timepoint = _reference_tiles(tiles, reference_channel, reference_timepoint)
    pairs = _find_pairs(refs, fractions)
    full = limit = 0.0
    for pair in pairs:
        a, b = refs[pair[0]], refs[pair[1]]
        size = np.minimum(np.array(a.shape) * a.voxel_size, np.array(b.shape) * b.voxel_size)
        lo, hi = _box(a, b, size * np.array(fractions))
        sides = [_read_cost(t, _crop(t, lo, hi, (1, 1, 1)), (1, 1, 1)) for t in (a, b)]
        full += sum(sides)
        limit += sum(min(s, _side_budget(budget_mb)) for s in sides)
    # the z offset between blocks is also measured in every other channel, on a few seams
    others = len({t.channel for t in tiles}) - 1
    across: Dict[tuple, List[tuple]] = {}
    for pair in pairs:
        a, b = refs[pair[0]], refs[pair[1]]
        if _group_of(pair[0]) != _group_of(pair[1]) and min(a.shape[0], b.shape[0]) >= _MIN_Z_PLANES:
            across.setdefault((a.field, *sorted((_group_of(pair[0]), _group_of(pair[1])))), []).append(pair)
    seams = others * sum(min(_CHANNEL_SEAMS, len(v)) for v in across.values())
    for pair in [p for v in across.values() for p in v[:_CHANNEL_SEAMS]]:
        a, b = refs[pair[0]], refs[pair[1]]
        size = np.minimum(np.array(a.shape) * a.voxel_size, np.array(b.shape) * b.voxel_size)
        lo, hi = _box(a, b, size * np.array(fractions))
        sides = [_read_cost(t, _crop(t, lo, hi, (1, 1, 1)), (1, 1, 1)) for t in (a, b)]
        limit += others * sum(min(s, _side_budget(budget_mb)) for s in sides)
    return RegistrationPlan(len(refs), pairs, channel, timepoint, fractions, budget_mb, full / 1e9, limit / 1e9, seams)


def _map_pairs(jobs: Sequence[_PairJob], workers: int, executor: str, on_result, progress: bool) -> None:
    """Run :func:`_measure_pair` on ``jobs`` and give every result to ``on_result``, in the
    calling process. At most ``2 * workers`` pairs are in flight, which bounds memory."""
    from tqdm.auto import tqdm

    pool = None
    if workers > 1:
        pool = (ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor)(max_workers=workers)
    try:
        with tqdm(total=len(jobs), desc="registering pairs", unit="pair", disable=not progress) as bar:
            if pool is None:
                for job in jobs:
                    on_result(_measure_pair(job))
                    bar.update(1)
                return
            in_flight: deque = deque()
            for job in jobs:
                in_flight.append(pool.submit(_measure_pair, job))
                while len(in_flight) >= 2 * workers:
                    on_result(in_flight.popleft().result())
                    bar.update(1)
            while in_flight:
                on_result(in_flight.popleft().result())
                bar.update(1)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)


def register_tiles(
    tiles: Sequence[Tile],
    save_to: str | Path,
    *,
    reference_channel: Optional[str] = None,
    reference_timepoint: Optional[int] = None,
    max_shift=MAX_SHIFT,
    budget_mb: Optional[float] = BUDGET_MB,
    min_ncc: float = MIN_NCC,
    max_residual: float = MAX_RESIDUAL,
    retry_rejected: bool = False,
    ds: Optional[Sequence[int]] = None,
    workers: int = 1,
    executor: str = "auto",
    progress: bool = True,
) -> Registration:
    """Measure and solve the corrections of ``tiles``; the tiles themselves do not change.

    Parameters
    ----------
    tiles
        Tiles to register. Those of the reference channel and timepoint are read, and in a
        field with several groups also a few seams between them in the other channels of
        ``tiles``. Tiles left out (e.g. a failed stack) stay at their stage position.
    save_to
        Registration file, written after every pair. If it exists and has the same
        settings the run continues from it: finished pairs are not read again.
    reference_channel
        Channel to match, ideally one with structure in the overlaps. Default: the first.
    reference_timepoint
        Timepoint to match. Default: the first.
    max_shift
        Search range between two tiles: a fraction of the tile size, one value or
        ``(z, y, x)``. Reads grow with it. The z axis is not searched for tiles of less
        than 16 planes, which are matched on their maximum projection in y/x only.
    budget_mb
        Most raw data read per pair, from its two tiles together (``None``: whole
        overlaps). Reading stops earlier when two slabs of a pair agree. A very small budget
        can be exceeded: every depth gets a probe, and every pair reads at least one slab.
    min_ncc
        Lowest NCC (on the background-subtracted images) at which a measurement is trusted.
    max_residual
        Links further than this from the solution, in working voxels, are dropped.
    retry_rejected
        With an existing ``save_to``: measure again the pairs that were rejected, for
        instance with a larger ``budget_mb``.
    ds
        ``(z, y, x)`` downsampling of the images matched. Default: chosen per pair so
        that the images matched stay small; it does not change what is read.
    workers, executor
        Pairs measured in parallel, as in :func:`~zarrsmith.mosaic_manager.fuse_tiles`.
    progress
        Show a progress bar.

    Returns
    -------
    Registration
        Apply it with :meth:`MosaicManager.apply_registration
        <zarrsmith.mosaic_manager.MosaicManager.apply_registration>`.
    """
    if executor not in ("auto", "process", "thread"):
        raise ValueError("executor must be 'auto', 'process' or 'thread'.")
    fractions = _max_shift_fractions(max_shift)
    if any("stage_origin" in t.attrs for t in tiles):
        raise ValueError("Tiles must be at their stage positions to be registered: "
                         "call mosaic.clear_registration() first.")
    refs, channel, timepoint = _reference_tiles(tiles, reference_channel, reference_timepoint)
    pairs = _find_pairs(refs, fractions)
    first = next(iter(refs.values()))
    settings = {"max_shift": list(fractions), "min_ncc": float(min_ncc), "max_residual": float(max_residual),
                "budget_mb": budget_mb, "ds": None if ds is None else [int(v) for v in ds]}
    positions = {name: list(t.origin) for name, t in refs.items()}
    identity = {"positions": positions, "reference_channel": channel, "reference_timepoint": timepoint}

    save_to = Path(save_to)
    fields = {name: str(t.field) for name, t in refs.items()}
    reg = Registration(voxel_size=first.voxel_size, units=first.units, settings=settings, fields=fields, **identity)
    if save_to.exists():
        old = Registration.load(save_to)
        same = {k: getattr(old, k) for k in identity} == identity and \
            all(old.settings.get(k) == settings[k] for k in ("max_shift", "ds"))
        if not same:
            raise ValueError(f"{save_to} holds another registration (other positions, reference or settings): "
                             "choose another file, or delete it to start again.")
        reg.links, reg.channel_links = old.links, old.channel_links
    if retry_rejected and reg.links:
        reg.solve()
        rejected = {key for key, o in reg.outcome.items() if o["status"] != "used"}
        reg.links = [link for link in reg.links if (link["a"], link["b"]) not in rejected]
    done = {(link["a"], link["b"]) for link in reg.links}

    jobs = [_PairJob(a, b, refs[a], refs[b], fractions, _side_budget(budget_mb), float(min_ncc),
                     None if ds is None else tuple(int(v) for v in ds))
            for a, b in pairs if (a, b) not in done]
    logger.info("Registering %d pairs (%d done) of %d positions, channel %r, timepoint %d",
                len(jobs), len(done), len(refs), channel, timepoint)

    def on_result(link):
        channel_of = link.get("channel")
        (reg.links if channel_of is None else reg.channel_links.setdefault(channel_of, [])).append(link)
        reg.save(save_to)

    reg.save(save_to)
    auto = executor == "auto"
    if auto:
        picklable = all(getattr(t.data, "picklable", False) for t in refs.values())
        executor = "process" if workers > 1 and picklable else "thread"
    try:
        _map_pairs(jobs, workers, executor, on_result, progress)
    except (BrokenProcessPool, RuntimeError) as e:
        if not (auto and executor == "process"):
            raise
        logger.warning("Worker processes failed (%s); retrying with threads.", e)
        finished = {(link["a"], link["b"]) for link in reg.links}
        _map_pairs([j for j in jobs if (j.a, j.b) not in finished], workers, "thread", on_result, progress)
    reg.solve()

    # the channels can differ in z between blocks: measure that on the best seams of each two blocks
    keys = {(position_id(k), t.channel): t for k, t in zip(series_keys(tiles), tiles) if t.timepoint == timepoint}
    seams: Dict[tuple, List[dict]] = {}
    for link in reg.links:
        outcome = reg.outcome[(link["a"], link["b"])]
        if reg._block(link["a"]) != reg._block(link["b"]) and not link["flat"] and outcome["status"] == "used":
            seams.setdefault(tuple(sorted((reg._block(link["a"]), reg._block(link["b"])))), []).append(link)
    best = [link for items in seams.values()
            for link in sorted(items, key=lambda k: reg.outcome[(k["a"], k["b"])]["ncc"], reverse=True)[:_CHANNEL_SEAMS]]
    others = [c for c in dict.fromkeys(t.channel for t in tiles) if c != channel]
    known = {c: {(link["a"], link["b"]) for link in reg.channel_links.get(c, [])} for c in others}
    jobs = [_PairJob(link["a"], link["b"], keys[(link["a"], c)], keys[(link["b"], c)], fractions,
                     _side_budget(budget_mb), float(min_ncc), None if ds is None else tuple(int(v) for v in ds), c)
            for c in others for link in best
            if (link["a"], link["b"]) not in known[c] and (link["a"], c) in keys and (link["b"], c) in keys]
    if jobs:
        logger.info("Measuring the z offset between blocks in %d other channel(s): %d seams", len(others), len(jobs))
        _map_pairs(jobs, workers, executor if executor != "auto" else "thread", on_result, progress)
        reg.solve()
    reg.save(save_to)
    logger.info("%s", reg.summary())
    return reg
