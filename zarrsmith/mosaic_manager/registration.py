"""Image-based registration of the opposing camera views of a MuVi SPIM.

The two cameras of a MuVi SPIM image the same sample position, but through two
detection paths (objective, tube lens, camera) that never match exactly: placed at the
stage position, the views are offset by a few micrometres and slightly rotated with
respect to each other, so the fused image jumps where one camera hands over to the other.

The fusion switches from one camera to the other around the middle of their overlap
along z (``blend="midline"``/``"sigmoid"``), so only the alignment there matters:
:func:`estimate_camera_registration` measures the transform between two views in a z
band centred on that handover plane, and :func:`register_cameras` applies it to the
tiles of the other cameras, so the fusion and its transforms file use it.

How the transform is measured:

1. both views are read downsampled in the band and placed in the stage frame (the moving
   view mirrored back along ``Tile.flip``, then resampled with the current estimate);
2. the band is cut into overlapping 3D blocks; in every block where both views hold
   structure, the residual shift is measured by phase correlation of band-passed images
   and kept when the blocks correlate well once shifted;
3. a rigid transform (translation and rotation, ``model="rigid"``) or a translation is
   fitted to the block shifts, weighted by their correlation, iteratively discarding
   outliers; the steps repeat on the corrected views.

How it is applied: a translation moves the tile (``Tile.origin``), the data is not
touched. A rotation cannot be expressed by a tile position: the tile's data is replaced
by an :class:`AffineArray`, a lazy view of it resampled on the reference camera's grid,
which every reader of ``Tile.data`` (fusion, napari) uses unchanged.

Transforms are 4 x 4 matrices on ``(z, y, x, 1)`` positions in the tile frame (the frame
of ``Tile.origin``, the stage frame for Luxendo tiles), mapping a position in the
reference view to the matching position in the moving view placed at the stage.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from .tile import Tile

logger = logging.getLogger("zarrsmith")

MODELS = ("rigid", "translation")

#: Planes read at once while building the downsampled views (bounds memory).
_READ_PLANES = 32


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def voxel_to_frame(origin, voxel_size, shape, flip) -> np.ndarray:
    """4 x 4 matrix mapping a raw voxel index ``(z, y, x, 1)`` of a tile's data to its
    position in the tile frame, with the flips (as the transforms file writes it)."""
    m = np.eye(4)
    for a in range(3):
        v = float(voxel_size[a])
        m[a, a] = -v if flip[a] else v
        m[a, 3] = float(origin[a]) + ((int(shape[a]) - 1) * v if flip[a] else 0.0)
    return m


def tile_voxel_to_frame(tile: Tile) -> np.ndarray:
    """:func:`voxel_to_frame` of a tile as it is placed now."""
    return voxel_to_frame(tile.origin, tile.voxel_size, tile.shape, tile.flip)


def rotation_angle(matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """Angle (degrees) and unit ``(z, y, x)`` axis of the rotation part of ``matrix``."""
    r = np.asarray(matrix, float)[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    n = np.linalg.norm(axis)
    return angle, (axis / n if n > 0 else np.array([1.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------

def placed_view(tile: Tile, factors: Sequence[int] = (1, 4, 4), z_range: Optional[Tuple[float, float]] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """The data of ``tile`` flipped back along ``Tile.flip`` (as fused) and block-mean
    downsampled by ``factors`` (float32), and the 4 x 4 matrix from its voxels (block
    centres) to the tile frame.

    ``z_range`` (tile frame units) limits the planes read to those inside it. Trailing
    voxels that do not fill a block are dropped.
    """
    from .fusion import _flip, _source_slice

    f = np.array([int(v) for v in factors])
    vox, origin, n = np.array(tile.voxel_size), np.array(tile.origin), np.array(tile.shape)
    lo, hi = 0, n[0] // f[0]  # downsampled planes, counted in the placed (flipped-back) frame
    if z_range is not None:
        lo = max(lo, int(np.ceil((z_range[0] - origin[0]) / (vox[0] * f[0]))))
        hi = min(hi, int(np.floor((z_range[1] - origin[0]) / (vox[0] * f[0]))))
    shape = (hi - lo, n[1] // f[1], n[2] // f[2])
    if min(shape) < 1:
        raise ValueError(f"Nothing to read from tile of shape {tile.shape} (downsampling {tuple(f)}, "
                         f"z range {z_range}).")
    out = np.empty(shape, np.float32)
    ys = _source_slice(0, shape[1] * f[1], n[1], tile.flip[1])
    xs = _source_slice(0, shape[2] * f[2], n[2], tile.flip[2])
    step = max(1, _READ_PLANES // f[0])
    for j in range(lo, hi, step):
        k = min(j + step, hi)
        zs = _source_slice(j * f[0], k * f[0], n[0], tile.flip[0])
        raw = _flip(np.asarray(tile.data[zs, ys, xs]), tile.flip).astype(np.float32)
        out[j - lo:k - lo] = raw.reshape(k - j, f[0], shape[1], f[1], shape[2], f[2]).mean((1, 3, 5))
    grid = np.diag(np.r_[vox * f, 1.0])
    grid[:3, 3] = origin + vox * (f - 1) / 2 + np.array([lo, 0, 0]) * vox * f
    return out, grid


def handover_z(reference: Tile, moving: Tile) -> float:
    """Position along z where the fusion hands over from one view to the other: the
    middle of the overlap of their z extents (as ``blend="midline"`` cuts)."""
    (r0, r1), (m0, m1) = ([t.extent[0][0], t.extent[1][0]] for t in (reference, moving))
    return (max(r0, m0) + min(r1, m1)) / 2


# ---------------------------------------------------------------------------
# block matching
# ---------------------------------------------------------------------------

@dataclass
class BlockMatches:
    """Shifts measured between two views, block by block.

    ``centres`` are block centres in the tile frame (``(n, 3)``, ``z, y, x``), ``shifts``
    the displacement from a structure in the reference view to the same structure in the
    (current) moving view, tile frame units, ``ncc`` the normalised correlation of the
    blocks once shifted.
    """

    centres: np.ndarray
    shifts: np.ndarray
    ncc: np.ndarray

    def __len__(self) -> int:
        return len(self.centres)

    def subset(self, keep: np.ndarray) -> "BlockMatches":
        return BlockMatches(self.centres[keep], self.shifts[keep], self.ncc[keep])


def _bandpass(vol: np.ndarray, sigma_vox: np.ndarray) -> np.ndarray:
    """Difference of Gaussians: keeps structures around the given size (e.g. nuclei),
    drops background and haze, which differ between the two cameras."""
    return ndi.gaussian_filter(vol, sigma_vox * 0.5) - ndi.gaussian_filter(vol, sigma_vox * 2.0)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def match_blocks(reference: np.ndarray, moving: np.ndarray, grid: np.ndarray, block_um: Sequence[float],
                 *, structure_um: float = 8.0, min_content: float = 0.25, max_shift_um: float = 30.0,
                 workers: int = 8) -> BlockMatches:
    """Shifts between two views on the same grid (``grid``: voxel -> tile frame, diagonal).

    The views are band-passed (structures around ``structure_um``) and cut into blocks
    of about ``block_um`` (``z, y, x``) with 50% overlap. Blocks where either view's
    structure (band-pass standard deviation) is below ``min_content`` times its 99th
    percentile over blocks (blank, or blurred out) are skipped, as are shifts larger
    than ``max_shift_um``.
    """
    from concurrent.futures import ThreadPoolExecutor

    from skimage.registration import phase_cross_correlation

    vox = np.diag(grid)[:3]
    sigma = np.maximum(structure_um / vox / 2.355, 0.5)
    ref_bp, mov_bp = _bandpass(reference, sigma), _bandpass(moving, sigma)
    size = np.minimum(np.maximum(np.rint(np.asarray(block_um, float) / vox).astype(int), 8), reference.shape)
    step = np.maximum(size // 2, 1)
    starts = [range(0, max(n - s, 0) + 1, st) for n, s, st in zip(reference.shape, size, step)]
    blocks = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]

    def slices(b):
        return tuple(slice(a, a + s) for a, s in zip(b, size))

    ref_std = np.array([ref_bp[slices(b)].std() for b in blocks])
    mov_std = np.array([mov_bp[slices(b)].std() for b in blocks])
    content = np.minimum(ref_std / max(np.percentile(ref_std, 99) * min_content, 1e-12),
                         mov_std / max(np.percentile(mov_std, 99) * min_content, 1e-12))
    candidates = [i for i in range(len(blocks)) if content[i] >= 1]
    window = np.ones(tuple(size), np.float32)
    for a, s in enumerate(size):  # Hann taper: no wrap-around edges in the FFT
        shape = [1, 1, 1]
        shape[a] = s
        window = window * np.hanning(s + 2)[1:-1].reshape(shape).astype(np.float32)

    def one(i):
        a, b = ref_bp[slices(blocks[i])], mov_bp[slices(blocks[i])]
        shift, _, _ = phase_cross_correlation(a * window, b * window, upsample_factor=10, normalization=None)
        # b shifted by ``shift`` matches a: a structure at x in a is at x - shift in b
        d = -np.asarray(shift, float)
        if np.any(np.abs(d * vox) > max_shift_um):
            return None
        return d, _ncc(a, ndi.shift(b, -d, order=1, mode="nearest", prefilter=False))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        found = list(pool.map(one, candidates))
    keep = [(i, r) for i, r in zip(candidates, found) if r is not None]
    if not keep:
        return BlockMatches(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    centres = (np.array([blocks[i] for i, _ in keep], float) + (size - 1) / 2) * vox + grid[:3, 3]
    return BlockMatches(centres, np.array([r[0] for _, r in keep]) * vox, np.array([r[1] for _, r in keep]))


# ---------------------------------------------------------------------------
# model fitting
# ---------------------------------------------------------------------------

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def _fit(src: np.ndarray, dst: np.ndarray, w: np.ndarray, model: str) -> np.ndarray:
    """4 x 4 transform taking ``src`` to ``dst``: weighted median translation, or weighted
    least-squares rigid transform (Kabsch)."""
    m = np.eye(4)
    if model == "translation":
        m[:3, 3] = [_weighted_median(dst[:, a] - src[:, a], w) for a in range(3)]
        return m
    cs, cd = (w[:, None] * src).sum(0) / w.sum(), (w[:, None] * dst).sum(0) / w.sum()
    h = ((src - cs) * w[:, None]).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    m[:3, :3] = r
    m[:3, 3] = cd - r @ cs
    return m


def fit_transform(matches: BlockMatches, model: str = "rigid", outlier_um: float = 2.0,
                  iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ``model`` to block matches: the 4 x 4 transform taking a reference position to
    the matching moving position, and the boolean mask of the inliers.

    Matches are weighted by their correlation; those whose residual exceeds
    ``max(outlier_um, 3 x median absolute residual)`` are dropped, iteratively.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}.")
    needed = 3 if model == "rigid" else 1
    src, dst = matches.centres, matches.centres + matches.shifts
    w = np.clip(matches.ncc, 1e-3, None)
    if len(src) < needed:
        raise ValueError(f"Only {len(src)} block matches: too few to fit a {model} transform.")
    inliers = np.ones(len(src), bool)
    m = np.eye(4)
    for _ in range(iterations):
        m = _fit(src[inliers], dst[inliers], w[inliers], model)
        res = np.linalg.norm((m[:3, :3] @ src.T).T + m[:3, 3] - dst, axis=1)
        limit = max(outlier_um, 3 * 1.4826 * float(np.median(res[inliers])))
        new = res <= limit
        if new.sum() < needed or np.array_equal(new, inliers):
            break
        inliers = new
    return m, inliers


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------

@dataclass
class CameraRegistration:
    """Result of :func:`estimate_camera_registration`.

    ``matrix`` maps a position in the reference view to the matching position in the
    moving view placed at the stage (tile frame, ``z, y, x``). ``initial`` are the block
    matches at the stage placement, ``matches`` those measured once registered (their
    shifts are what is left), ``z_range`` the band that was matched.
    """

    matrix: np.ndarray
    model: str
    reference: str
    moving: str
    z_range: Tuple[float, float]
    initial: BlockMatches
    matches: BlockMatches
    factors: Tuple[int, int, int]
    channel: Optional[str] = None  # channel the transform was measured on (None: several combined)

    @property
    def centre(self) -> np.ndarray:
        """Centre of the matched blocks (tile frame)."""
        return self.initial.centres.mean(0) if len(self.initial) else np.zeros(3)

    @property
    def shift(self) -> np.ndarray:
        """Displacement of the centre of the matched blocks, reference -> moving view."""
        c = self.centre
        return self.matrix[:3, :3] @ c + self.matrix[:3, 3] - c

    @property
    def rotation(self) -> Tuple[float, np.ndarray]:
        """Angle (degrees) and ``(z, y, x)`` axis of the rotation, see :func:`rotation_angle`."""
        return rotation_angle(self.matrix)

    def misalignment(self, registered: bool = True) -> float:
        """Median norm of the block shifts once registered (or at the stage placement)."""
        m = self.matches if registered else self.initial
        return float(np.median(np.linalg.norm(m.shifts, axis=1))) if len(m) else float("nan")

    def summary(self) -> str:
        angle, axis = self.rotation
        rot = f", rotation {angle:.2f} deg about (z, y, x) {np.round(axis, 2).tolist()}" if self.model == "rigid" else ""
        on = f", {self.channel}" if self.channel else ""
        return (f"{self.moving} -> {self.reference} ({self.model}{on}): shift (z, y, x) "
                f"{np.round(self.shift, 2).tolist()}{rot}; from {len(self.initial)} blocks in z "
                f"{self.z_range[0]:.0f}..{self.z_range[1]:.0f}, median block misalignment "
                f"{self.misalignment(False):.2f} (stage) -> {self.misalignment():.2f} (registered)")

    def to_dict(self) -> Dict:
        angle, axis = self.rotation
        return {
            "model": self.model,
            "channel": self.channel,
            "reference_camera": self.reference,
            "moving_camera": self.moving,
            "matrix": self.matrix.tolist(),
            "shift": [float(v) for v in self.shift],
            "centre": [float(v) for v in self.centre],
            "rotation_deg": angle,
            "rotation_axis": [float(v) for v in axis],
            "z_range": [float(v) for v in self.z_range],
            "blocks": len(self.initial),
            "misalignment_stage": self.misalignment(False),
            "misalignment_registered": self.misalignment(),
            "downsampling": list(self.factors),
        }


def _views(tiles: Sequence[Tile], factors, z_range) -> Tuple[np.ndarray, np.ndarray]:
    """Downsampled placed views of tiles of one camera, summed over channels after
    normalising each to its 99.9th percentile (one transform holds for all channels)."""
    total = grid = None
    for t in tiles:
        v, grid = placed_view(t, factors, z_range)
        lo, hi = np.percentile(v[:, ::4, ::4], (1, 99.9))
        v = (v - lo) / max(hi - lo, 1e-6)
        total = v if total is None else total + v
    return total, grid


def estimate_camera_registration(
    reference: Sequence[Tile] | Tile,
    moving: Sequence[Tile] | Tile,
    *,
    model: str = "rigid",
    band_um: float = 200.0,
    factors: Sequence[int] = (1, 4, 4),
    block_um: Sequence[float] = (40.0, 150.0, 150.0),
    structure_um: float = 8.0,
    min_content: float = 0.25,
    min_ncc: float = 0.3,
    max_shift_um: float = 30.0,
    outlier_um: float = 2.0,
    rounds: int = 3,
    workers: int = 8,
) -> CameraRegistration:
    """Measure the transform between two camera views of one stack, around the plane
    where the fusion hands over from one camera to the other (:func:`handover_z`).

    Parameters
    ----------
    reference, moving
        Tiles of the reference and of the moving camera (one tile each, or one per
        channel: the channels are combined, as one transform holds for all of them).
    model
        ``"rigid"`` (default: translation and rotation) or ``"translation"``.
    band_um
        Thickness of the z band, centred on the handover plane, that is matched.
    factors
        ``(z, y, x)`` downsampling of the views that are matched.
    block_um
        Size of the matched blocks (``z, y, x``), in tile units.
    structure_um
        Size of the structures matched (band-pass), e.g. nuclei.
    min_content
        Blocks with less structure than this fraction of the 99th percentile over blocks,
        in either view, are skipped.
    min_ncc
        Block matches whose correlation, once shifted, is lower are discarded.
    max_shift_um
        Largest shift accepted for a block.
    outlier_um
        Residual (after the fit) above which a block is an outlier (or 3 median absolute
        deviations, if larger).
    rounds
        Measure-and-fit rounds, each on the views corrected by the previous ones.
    workers
        Blocks matched in parallel.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}.")
    ref_tiles = [reference] if isinstance(reference, Tile) else list(reference)
    mov_tiles = [moving] if isinstance(moving, Tile) else list(moving)
    r0, m0 = ref_tiles[0], mov_tiles[0]
    if not np.allclose(r0.voxel_size, m0.voxel_size, rtol=1e-3):
        raise ValueError(f"Views have different voxel sizes: {r0.voxel_size} and {m0.voxel_size}.")
    factors = tuple(int(v) for v in factors)
    zc = handover_z(r0, m0)
    z_range = (zc - band_um / 2, zc + band_um / 2)
    margin = max_shift_um + 0.05 * band_um  # the moving band, a little thicker (shift, tilt)
    logger.info("Camera registration %s -> %s (%s): reading z %.0f..%.0f, downsampled %s", m0.camera,
                r0.camera, model, *z_range, factors)
    ref_view, ref_grid = _views(ref_tiles, factors, z_range)
    mov_view, mov_grid = _views(mov_tiles, factors, (z_range[0] - margin, z_range[1] + margin))

    def measure(matrix):
        to_mov = np.linalg.inv(mov_grid) @ matrix @ ref_grid  # reference voxel -> moving voxel
        warped = ndi.affine_transform(mov_view, to_mov[:3, :3], offset=to_mov[:3, 3], output_shape=ref_view.shape,
                                      order=1, mode="constant", cval=0.0, prefilter=False)
        found = match_blocks(ref_view, warped, ref_grid, block_um, structure_um=structure_um,
                             min_content=min_content, max_shift_um=max_shift_um, workers=workers)
        return found.subset(found.ncc >= min_ncc)

    matrix = np.eye(4)
    initial = matches = measure(matrix)
    for rnd in range(max(1, rounds)):
        if len(matches) < (3 if model == "rigid" else 1):
            raise ValueError(f"Only {len(matches)} blocks match between the {r0.camera} and {m0.camera} views in z "
                             f"{z_range[0]:.0f}..{z_range[1]:.0f}: cannot register them (try a thicker band, lower "
                             "min_ncc / min_content, or larger blocks).")
        # matches: reference position -> position in the corrected view; compose
        update, _ = fit_transform(matches, model=model, outlier_um=outlier_um)
        matrix = matrix @ update
        matches = measure(matrix)
        logger.info("Round %d: %d blocks, median misalignment %.2f", rnd + 1, len(matches),
                    float(np.median(np.linalg.norm(matches.shifts, axis=1))) if len(matches) else float("nan"))
    channels = {t.channel for t in ref_tiles + mov_tiles}
    result = CameraRegistration(matrix=matrix, model=model, reference=str(r0.camera), moving=str(m0.camera),
                                z_range=z_range, initial=initial, matches=matches, factors=factors,
                                channel=channels.pop() if len(channels) == 1 else None)
    logger.info("Camera registration: %s", result.summary())
    return result


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------

class AffineArray:
    """Lazy view of a tile's data resampled on another grid through an affine transform.

    Behaves like a read-only ``(z, y, x)`` array of ``shape``: reading a block reads only
    the source voxels it maps to and interpolates them (``order`` 1, linear; 0 keeps the
    values, for labels). Voxels mapped outside the source take the value of the nearest
    source voxel, so no dark seam appears along the borders. Picklable when the source
    is, so fusion can read it in worker processes.

    ``matrix`` (4 x 4) maps an index of this array to a (fractional) index of ``source``.
    """

    def __init__(self, source, matrix: np.ndarray, shape: Sequence[int], order: int = 1):
        self.source = source
        self.matrix = np.asarray(matrix, float)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(source.dtype)
        self.ndim = 3
        self.order = int(order)
        self.picklable = bool(getattr(source, "picklable", False))

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"AffineArray({self.source!r}, shape={self.shape})"

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (3 - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (3 - len(key))
        starts, sizes, squeeze = [], [], []
        for a, k in enumerate(key):
            if isinstance(k, (int, np.integer)):
                k = int(k) + (self.shape[a] if k < 0 else 0)
                k = slice(k, k + 1)
                squeeze.append(a)
            start, stop, step = k.indices(self.shape[a])
            if step != 1:
                raise IndexError("AffineArray supports only contiguous slices.")
            starts.append(start)
            sizes.append(max(stop - start, 0))
        if min(sizes) == 0:
            return np.zeros(sizes, self.dtype).squeeze(tuple(squeeze))
        out = self._read(np.array(starts, float), tuple(sizes))
        return out.squeeze(tuple(squeeze)) if squeeze else out

    def _read(self, start: np.ndarray, size: Tuple[int, int, int]) -> np.ndarray:
        corners = np.array([[start[0] + dz * (size[0] - 1), start[1] + dy * (size[1] - 1),
                             start[2] + dx * (size[2] - 1), 1.0]
                            for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)])
        src = (self.matrix @ corners.T).T[:, :3]
        n = np.array(self.source.shape)
        lo = np.clip(np.floor(src.min(0)).astype(int) - 1, 0, n - 1)
        hi = np.clip(np.ceil(src.max(0)).astype(int) + 2, lo + 1, n)
        data = np.asarray(self.source[tuple(slice(a, b) for a, b in zip(lo, hi))])
        offset = self.matrix[:3, :3] @ start + self.matrix[:3, 3] - lo
        out = ndi.affine_transform(data.astype(np.float32) if self.order else data, self.matrix[:3, :3],
                                   offset=offset, output_shape=size, order=self.order, mode="nearest",
                                   prefilter=False)
        if np.issubdtype(self.dtype, np.integer) and self.order:
            info = np.iinfo(self.dtype)
            return np.clip(np.rint(out), info.min, info.max).astype(self.dtype)
        return out.astype(self.dtype, copy=False)


def _is_translation(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.allclose(np.asarray(matrix)[:3, :3], np.eye(3), rtol=0.0, atol=tol))


def registered_tile(tile: Tile, matrix: np.ndarray, grid: Optional[Dict[str, Any]] = None,
                    method: str = "registration", order: int = 1) -> Tile:
    """A copy of ``tile`` (at its stage placement) registered by ``matrix`` (reference
    position -> position in ``tile``, tile frame).

    A translation moves the tile by ``-translation``; the data is not touched. Otherwise
    the data is resampled (:class:`AffineArray`, interpolation ``order``) on ``grid``,
    ``{"origin", "voxel_size", "shape", "flip"}``, usually the reference camera's tile.
    ``attrs["registration"]`` keeps ``matrix`` (and ``grid``), which the transforms file
    records; ``attrs`` also keeps the stage placement (``stage_origin``, ``stage_flip``,
    ``stage_data``) for :meth:`MosaicManager.reset_transforms
    <zarrsmith.mosaic_manager.MosaicManager.reset_transforms>`.
    """
    matrix = np.asarray(matrix, float)
    attrs = dict(tile.attrs)
    attrs.setdefault("stage_origin", tile.origin)
    attrs.setdefault("stage_flip", tile.flip)
    attrs["transform_method"] = method
    if _is_translation(matrix):
        attrs["registration"] = {"matrix": matrix.tolist()}
        return dataclasses.replace(tile, origin=tuple(np.array(tile.origin) - matrix[:3, 3]), attrs=attrs)
    if grid is None:
        raise ValueError("A rotation needs the grid to resample the tile on.")
    grid = {"origin": [float(v) for v in grid["origin"]], "voxel_size": [float(v) for v in grid["voxel_size"]],
            "shape": [int(v) for v in grid["shape"]], "flip": [bool(v) for v in grid["flip"]]}
    attrs.setdefault("stage_data", tile.data)
    attrs["registration"] = {"matrix": matrix.tolist(), "grid": grid}
    to_src = (np.linalg.inv(tile_voxel_to_frame(tile)) @ matrix
              @ voxel_to_frame(grid["origin"], grid["voxel_size"], grid["shape"], grid["flip"]))
    return dataclasses.replace(tile, data=AffineArray(tile.data, to_src, grid["shape"], order=order),
                               origin=tuple(grid["origin"]), voxel_size=tuple(grid["voxel_size"]),
                               flip=tuple(grid["flip"]), attrs=attrs)


def tile_grid(tile: Tile) -> Dict[str, Any]:
    """The grid of ``tile`` as :func:`registered_tile` takes it."""
    return {"origin": list(tile.origin), "voxel_size": list(tile.voxel_size), "shape": list(tile.shape),
            "flip": list(tile.flip)}


#: ``reference_channel`` / ``reference_timepoint`` value that registers every channel / timepoint.
ALL = "all"


def _reference(value, available: Sequence, label: str):
    """The reference among ``available`` (``None``: the first; ``ALL``: ``None``, each its own)."""
    if value is None:
        return available[0]
    if isinstance(value, str) and value.lower() == ALL:
        return None
    if label == "timepoint":
        value = int(value)
    if value not in available:
        raise ValueError(f"Reference {label} {value!r} is not among the {label}s {list(available)}.")
    return value


def register_cameras(
    tiles: Sequence[Tile],
    *,
    reference_camera: Optional[str] = None,
    reference_channel: Optional[str] = None,
    reference_timepoint=None,
    model: str = "rigid",
    transforms: Optional[Dict[tuple, np.ndarray]] = None,
    **options,
) -> Tuple[List[Tile], Dict[tuple, CameraRegistration]]:
    """Register the views of every camera onto a reference camera, stack by stack.

    Returns ``(tiles, results)``: ``tiles`` in the same order, with the tiles of the other
    cameras replaced by :func:`registered_tile` copies (the input tiles are not changed),
    and the :class:`CameraRegistration` measured for every
    ``(stack, timepoint, group, camera, channel)``.

    Parameters
    ----------
    tiles
        Tiles of several cameras (``Tile.camera``), at their stage placement; tiles of
        one camera alone are returned unchanged.
    reference_camera
        Camera the others are registered onto. Default: the first camera of ``tiles``.
    reference_channel
        Channel the transform is measured on (e.g. a nuclear stain), then applied to every
        channel. Default: the first channel of ``tiles``. ``"all"``: every channel is
        registered on its own images (this also corrects a chromatic difference between the
        cameras); a channel that cannot be registered (too little structure) gets the
        transform of the channel with the most matched blocks, with a warning.
    reference_timepoint
        Timepoint the transform is measured at, then applied to every timepoint (as the
        fusion places every timepoint with the geometry of its reference timepoint).
        Default: the first timepoint of ``tiles``. ``"all"``: every timepoint is registered.
    model
        ``"rigid"`` (translation and rotation) or ``"translation"``.
    transforms
        Known 4 x 4 transforms, by ``(stack, timepoint, group, camera, channel)`` or by
        ``(stack, timepoint, group, camera)`` for every channel (``timepoint`` is the one
        measured, see ``reference_timepoint``); those are not measured.
    **options
        Passed to :func:`estimate_camera_registration`.
    """
    cameras = list(dict.fromkeys(t.camera for t in tiles))
    if len(cameras) < 2:
        return list(tiles), {}
    if None in cameras:
        raise ValueError("Some tiles have a camera and others not: they cannot be registered.")
    ref_cam = cameras[0] if reference_camera is None else reference_camera
    if ref_cam not in cameras:
        raise ValueError(f"Reference camera {ref_cam!r} is not among the cameras {cameras}.")
    ref_ch = _reference(reference_channel, list(dict.fromkeys(t.channel for t in tiles)), "channel")
    ref_tp = _reference(reference_timepoint, sorted({t.timepoint for t in tiles}), "timepoint")
    known = dict(transforms or {})
    # position (stack, group) -> timepoint -> camera -> channel -> tile
    views: Dict[tuple, Dict[int, Dict[str, Dict[str, Tile]]]] = {}
    for t in tiles:
        views.setdefault((t.attrs.get("stack"), t.group), {}).setdefault(t.timepoint, {})              .setdefault(t.camera, {})[t.channel] = t

    results: Dict[tuple, CameraRegistration] = {}
    cache: Dict[tuple, Dict[str, Tuple[np.ndarray, str]]] = {}

    def view_transforms(stack, group, tp, cam, refs, movs) -> Dict[str, Tuple[np.ndarray, str]]:
        """``{channel: (matrix, channel measured on)}`` for the moving view ``cam`` at ``tp``."""
        vkey = (stack, tp, group, cam)
        if vkey in cache:
            return cache[vkey]
        wanted = [ref_ch] if ref_ch is not None else [c for c in movs if c in refs]
        found: Dict[str, Tuple[np.ndarray, str]] = {}
        failures = {}
        for channel in wanted:
            known_matrix = known.get(vkey + (channel,), known.get(vkey))
            if known_matrix is not None:
                found[channel] = (np.asarray(known_matrix, float), channel)
                continue
            if channel not in refs or channel not in movs:
                raise ValueError(f"Stack {stack} (t={tp}): the reference channel {channel!r} is missing for "
                                 f"camera {ref_cam!r} or {cam!r}.")
            try:
                reg = estimate_camera_registration(refs[channel], movs[channel], model=model, **options)
            except ValueError as e:
                if ref_ch is not None:
                    raise
                failures[channel] = e
                continue
            results[vkey + (channel,)] = reg
            found[channel] = (reg.matrix, channel)
        if ref_ch is None:  # every channel: those that failed take the best registered one
            if not found:
                raise ValueError(f"Stack {stack} (t={tp}): no channel could be registered ({failures}).")
            best = max((c for c in found if vkey + (c,) in results),
                       key=lambda c: len(results[vkey + (c,)].matches), default=next(iter(found)))
            for channel, e in failures.items():
                logger.warning("Stack %s (t=%s), %s: %s Using the transform of %s.", stack, tp, channel, e, best)
                found[channel] = found[best]
            for channel in movs:  # a channel without a reference view: same fallback
                found.setdefault(channel, found[best])
        cache[vkey] = found
        return found

    out = {id(t): t for t in tiles}
    for (stack, group), by_tp in views.items():
        for tp, cams in by_tp.items():
            src_tp = tp if ref_tp is None else ref_tp
            src = by_tp.get(src_tp)
            if not src or ref_cam not in src or ref_cam not in cams:
                logger.warning("Stack %s (t=%s): no %s camera view at timepoint %s: not registered.",
                               stack, tp, ref_cam, src_tp)
                continue
            for cam, movs in cams.items():
                if cam == ref_cam:
                    continue
                if cam not in src:
                    logger.warning("Stack %s: no %s camera view at timepoint %s: not registered.", stack, cam, src_tp)
                    continue
                found = view_transforms(stack, group, src_tp, cam, src[ref_cam], src[cam])
                refs = cams[ref_cam]
                for channel, t in movs.items():
                    key = channel if ref_ch is None else ref_ch
                    matrix, measured = found[key] if key in found else next(iter(found.values()))
                    out[id(t)] = registered_tile(t, matrix, tile_grid(refs.get(channel, next(iter(refs.values())))))
                    out[id(t)].attrs["registration"].update(channel=measured, timepoint=int(src_tp))
    return [out[id(t)] for t in tiles], results
