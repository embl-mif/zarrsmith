"""Content-based registration of mosaic tiles: correct stage positions from the image data.

Registration is translation-only and runs in three steps, on one **reference channel** at
one **reference timepoint**; the result is applied to every channel and timepoint:

1. **Pairs.** Tiles whose boxes share a face (within the search range) are paired, inside
   one field. Only the overlap of a pair is read, widened by the search range ``max_shift``.
2. **Measure.** The read per tile is bounded (``budget_mb``), since raw tiles have no pyramid
   and downsampling does not reduce what is read. A window that does not fit is sampled in
   contiguous, chunk-aligned **slabs** along z (scattered planes would decode the same
   chunks again and again). Small probes first locate the depths where both tiles have
   signal. Each slab is matched on its own, with phase correlation proposing shifts and
   normalized cross-correlation (NCC) on the real overlap choosing between them, and
   reading stops as soon as two slabs agree. Results are saved after every pair, so an
   interrupted run resumes without reading anything again.
3. **Solve.** Weighted least squares over all links, per axis, with a weak pull towards the
   stage position, so that tiles without a reliable link stay close to it. Links that
   disagree with the solution are dropped and the fit is repeated. Thresholds can be changed
   and the fit repeated (:meth:`Registration.solve`) without reading any pixel.

Applying a registration only updates :attr:`Tile.origin
<zarrsmith.mosaic_manager.tile.Tile.origin>` in memory, see
:meth:`MosaicManager.apply_registration
<zarrsmith.mosaic_manager.MosaicManager.apply_registration>`. Fusion then places the tiles
as usual: on whole output voxels, with no interpolation.
"""
from __future__ import annotations

import json
import logging
import math
import os
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

#: Version of the registration file.
FILE_VERSION = 1

#: Default search range, as a fraction of the tile size per axis (z, y, x): stage errors of
#: up to 3% on each of two tiles give a relative error of up to 6% in y/x.
MAX_SHIFT = (0.02, 0.06, 0.06)
#: Default limit of raw data read per tile, split between its neighbours.
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


# ---------------------------------------------------------------------------
# which tiles, which pairs
# ---------------------------------------------------------------------------

def position_id(key: tuple) -> str:
    """Name of a position from its :func:`~zarrsmith.mosaic_manager.tile.series_keys` key:
    ``"group|stack|rank"``, shared by its tiles in every channel and timepoint."""
    _, group, stack, rank = key
    return f"{'' if group is None else group}|{'' if stack is None else stack}|{rank}"


def _max_shift_fractions(max_shift) -> Tuple[float, float, float]:
    """``(z, y, x)`` search range from one fraction of the tile size or three."""
    values = tuple(float(v) for v in (max_shift if np.ndim(max_shift) else (max_shift,) * 3))
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


def _neighbours(pairs) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for a, b in pairs:
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    return counts


def _side_budget(pair, counts, budget_mb: Optional[float]) -> float:
    """Bytes that may be read from each tile of ``pair``: the budget of the tile with the
    most neighbours, split between them (no limit without a budget)."""
    if budget_mb is None:
        return math.inf
    return budget_mb * 1e6 / max(counts[pair[0]], counts[pair[1]])


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
    """Measure the shift of one pair (runs in the workers). Returns one entry of ``links``."""
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
    link: Dict[str, Any] = {"a": job.a, "b": job.b, "size": [float(v) for v in size], "flat": flat,
                            "estimates": [], "read_mb": 0.0, "slabs_tried": 0, "slabs_available": 1, "signal": True}
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
    link["read_mb"] = spent / 1e6
    return link


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
    return {"shift": np.average([e["shift"] for e in group], axis=0, weights=weights),
            "voxel": np.array(group[0]["voxel"]),
            "ncc": float(np.mean(weights)),
            "ncc_stage": float(np.mean([e["ncc_stage"] for e in group]))}, ""


def _least_squares(ids: Sequence[str], used: List[dict]) -> np.ndarray:
    """Corrections ``(n, 3)`` from links ``x[b] - x[a] = shift``, weighted by their NCC, with a
    weak pull to 0, which keeps every group of connected tiles centred on its stage positions."""
    index = {name: i for i, name in enumerate(ids)}
    n, m = len(ids), len(used)
    rows, cols, values = [], [], []
    for k, u in enumerate(used):
        rows += [k, k]
        cols += [index[u["a"]], index[u["b"]]]
        values += [-1.0, 1.0]
    design = sparse.csr_matrix((values, (rows, cols)), shape=(m, n))
    weights = sparse.diags([u["ncc"] for u in used]) if m else sparse.csr_matrix((0, 0))
    normal = (design.T @ weights @ design + _STAGE_PULL * sparse.identity(n)).tocsc()
    out = np.zeros((n, 3))
    if m:
        target = np.array([u["shift"] for u in used])
        for axis in range(3):
            out[:, axis] = spsolve(normal, design.T @ (weights @ target[:, axis]))
    return out


def _solve(ids: Sequence[str], links: List[dict], *, min_ncc: float, max_shift, max_residual: float):
    """``(corrections, per-link outcome)``: the fit over the usable links, dropping the link
    furthest from the solution until all are within ``max_residual`` working voxels."""
    used, outcome = [], {}
    limits = _max_shift_fractions(max_shift)
    for link in links:
        key = (link["a"], link["b"])
        value, why = _link_value(link, min_ncc)
        if value is not None and np.any(np.abs(value["shift"]) > np.array(link["size"]) * limits + value["voxel"]):
            value, why = None, "shift above max_shift"
        if value is None:
            outcome[key] = {"status": "rejected", "reason": why}
        else:
            used.append({"a": link["a"], "b": link["b"], "key": key, **value})
    index = {name: i for i, name in enumerate(ids)}
    corrections = _least_squares(ids, used)
    while used:
        worst, worst_value = None, 0.0
        for i, u in enumerate(used):
            found = corrections[index[u["b"]]] - corrections[index[u["a"]]]
            value = float(np.max(np.abs(found - u["shift"]) / u["voxel"]))
            if value > worst_value:
                worst, worst_value = i, value
        if worst is None or worst_value <= max_residual:
            break
        dropped = used.pop(worst)
        outcome[dropped["key"]] = {"status": "rejected", "reason": f"disagrees with the others ({worst_value:.1f} voxels)"}
        corrections = _least_squares(ids, used)
    for u in used:
        found = corrections[index[u["b"]]] - corrections[index[u["a"]]]
        outcome[u["key"]] = {"status": "used", "ncc": u["ncc"], "ncc_stage": u["ncc_stage"],
                             "residual": float(np.max(np.abs(found - u["shift"]) / u["voxel"]))}
    return corrections, outcome


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class Registration:
    """Measurements of a registration and the corrections solved from them.

    Created by :func:`register_tiles`, or read back with :meth:`load`. ``shifts`` maps every
    position (:func:`position_id`) to its ``(z, y, x)`` correction in sample units. Nothing
    changes on the tiles until it is given to
    :meth:`MosaicManager.apply_registration <zarrsmith.mosaic_manager.MosaicManager.apply_registration>`.
    """

    positions: Dict[str, List[float]]  # stage position of every position, at the reference timepoint
    voxel_size: Tuple[float, float, float]
    units: str
    reference_channel: str
    reference_timepoint: int
    settings: Dict[str, Any]
    links: List[dict] = _dc_field(default_factory=list)
    shifts: Dict[str, List[float]] = _dc_field(default_factory=dict)
    outcome: Dict[Tuple[str, str], dict] = _dc_field(default_factory=dict)

    @property
    def n_used(self) -> int:
        """Links used in the solution."""
        return sum(o["status"] == "used" for o in self.outcome.values())

    def solve(self, *, min_ncc: Optional[float] = None, max_shift=None, max_residual: Optional[float] = None) -> "Registration":
        """Fit the corrections from the saved measurements; reads no pixel data.

        Parameters left out keep their value. ``max_shift`` can only reject more than was
        searched when measuring. Call :meth:`save` to keep the result.
        """
        for key, value in (("min_ncc", min_ncc), ("max_residual", max_residual),
                           ("max_shift", None if max_shift is None else list(_max_shift_fractions(max_shift)))):
            if value is not None:
                self.settings[key] = value
        ids = list(self.positions)
        corrections, self.outcome = _solve(ids, self.links, min_ncc=self.settings["min_ncc"],
                                           max_shift=self.settings["max_shift"],
                                           max_residual=self.settings["max_residual"])
        self.shifts = {name: [float(v) for v in c] for name, c in zip(ids, corrections)}
        return self

    def summary(self) -> str:
        """Report: links used and rejected (with the reasons), size of the corrections, reads."""
        lines = [f"{len(self.positions)} positions, {len(self.links)} links, {self.n_used} used "
                 f"(reference channel {self.reference_channel!r}, timepoint {self.reference_timepoint})"]
        read = sum(link["read_mb"] for link in self.links)
        lines.append(f"Read {read / 1e3:.1f} GB in {sum(link['slabs_tried'] for link in self.links)} slab(s)")
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
            lines.append(f"{len(alone)} position(s) without a used link stay at the stage position: "
                         + ", ".join(alone[:6]) + (", ..." if len(alone) > 6 else ""))
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Write the registration as JSON; a crash while writing leaves the previous file."""
        path = Path(path)
        content = {"version": FILE_VERSION, "positions": self.positions, "voxel_size": list(self.voxel_size),
                   "units": self.units, "reference_channel": self.reference_channel,
                   "reference_timepoint": self.reference_timepoint, "settings": self.settings,
                   "links": self.links, "shifts": self.shifts}
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(content, indent=1), encoding="utf-8")
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str | Path) -> "Registration":
        """Read a file written by :meth:`save`, also one of an unfinished run (without ``shifts``)."""
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        if content.get("version") != FILE_VERSION:
            raise ValueError(f"{path} has registration file version {content.get('version')!r}, "
                             f"expected {FILE_VERSION}.")
        reg = cls(positions=content["positions"], voxel_size=tuple(content["voxel_size"]), units=content["units"],
                  reference_channel=content["reference_channel"], reference_timepoint=content["reference_timepoint"],
                  settings=content["settings"], links=content["links"], shifts=content["shifts"])
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

    def summary(self) -> str:
        """One line with the number of pairs and the data read at most."""
        budget = "no budget" if self.budget_mb is None else f"budget {self.budget_mb:.0f} MB per tile"
        return (f"{self.positions} positions, {len(self.pairs)} pairs in channel {self.reference_channel!r} at "
                f"timepoint {self.reference_timepoint}: reads at most {self.limit_gb:.1f} GB ({budget}; "
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
    counts = _neighbours(pairs)
    full = limit = 0.0
    for pair in pairs:
        a, b = refs[pair[0]], refs[pair[1]]
        size = np.minimum(np.array(a.shape) * a.voxel_size, np.array(b.shape) * b.voxel_size)
        lo, hi = _box(a, b, size * np.array(fractions))
        sides = [_read_cost(t, _crop(t, lo, hi, (1, 1, 1)), (1, 1, 1)) for t in (a, b)]
        full += sum(sides)
        limit += sum(min(s, _side_budget(pair, counts, budget_mb)) for s in sides)
    return RegistrationPlan(len(refs), pairs, channel, timepoint, fractions, budget_mb, full / 1e9, limit / 1e9)


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
        Tiles to register. Only those of the reference channel and timepoint are read. Tiles
        left out (e.g. a failed stack) stay at their stage position.
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
        Most raw data read per tile, split between its neighbours (``None``: whole
        overlaps). Reading stops earlier when two slabs of a pair agree. The budget of the
        first slab of a pair can be exceeded when no thinner slab is possible.
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
    refs, channel, timepoint = _reference_tiles(tiles, reference_channel, reference_timepoint)
    pairs = _find_pairs(refs, fractions)
    counts = _neighbours(pairs)
    first = next(iter(refs.values()))
    settings = {"max_shift": list(fractions), "min_ncc": float(min_ncc), "max_residual": float(max_residual),
                "budget_mb": budget_mb, "ds": None if ds is None else [int(v) for v in ds]}
    positions = {name: list(t.attrs.get("stage_origin", t.origin)) for name, t in refs.items()}
    identity = {"positions": positions, "reference_channel": channel, "reference_timepoint": timepoint}

    save_to = Path(save_to)
    reg = Registration(voxel_size=first.voxel_size, units=first.units, settings=settings, **identity)
    if save_to.exists():
        old = Registration.load(save_to)
        same = {k: getattr(old, k) for k in identity} == identity and \
            all(old.settings.get(k) == settings[k] for k in ("max_shift", "ds"))
        if not same:
            raise ValueError(f"{save_to} holds another registration (other positions, reference or settings): "
                             "choose another file, or delete it to start again.")
        reg.links = old.links
    if retry_rejected and reg.links:
        reg.solve()
        rejected = {key for key, o in reg.outcome.items() if o["status"] != "used"}
        reg.links = [link for link in reg.links if (link["a"], link["b"]) not in rejected]
    done = {(link["a"], link["b"]) for link in reg.links}

    jobs = [_PairJob(a, b, refs[a], refs[b], fractions, _side_budget((a, b), counts, budget_mb), float(min_ncc),
                     None if ds is None else tuple(int(v) for v in ds))
            for a, b in pairs if (a, b) not in done]
    logger.info("Registering %d pairs (%d done) of %d positions, channel %r, timepoint %d",
                len(jobs), len(done), len(refs), channel, timepoint)

    def on_result(link):
        reg.links.append(link)
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
    reg.save(save_to)
    logger.info("%s", reg.summary())
    return reg
