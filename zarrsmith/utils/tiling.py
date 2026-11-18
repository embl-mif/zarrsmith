"""Tile an image for parallel processing, and lay out the output it is written to.

A plan cuts the spatial axes (z, y, x) of one pyramid level into overlapping
tiles. Along each axis, tiles start every ``stride = tile_size - overlap``
pixels from the start of the region of interest (ROI). An axis without a tile
size is not split: one tile spans the whole ROI along it.

Each tile owns a *core*: from its start to the next tile's start (to its end for
the last tile of an axis). Cores do not overlap and cover the ROI, and every
core starts on a multiple of the stride from the ROI start. So an output image
of the ROI whose write block (shard on zarr v3, chunk otherwise) is the stride
lets every tile write its core in parallel, without two tiles touching the same
file (:meth:`TilePlan.allocate_kwargs` gives that layout to
:func:`zarrsmith.utils.allocate.allocate_zarr`).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Mapping

from .region import physical_to_pixel_bounds, physical_to_pixels

TILED_AXES = ("z", "y", "x")

# Default inner chunk (zarr v3) per tiled axis: ~4 MiB of uint32 labels in 3D.
DEFAULT_INNER_CHUNK = {"z": 16, "y": 256, "x": 256}


@dataclass(frozen=True)
class Tile:
    """One tile: ``index`` (1-based), its position in the tile grid, and its
    pixel bounds at the plan's level (half-open, per tiled axis)."""

    index: int
    grid: dict[str, int]
    bounds: dict[str, tuple[int, int]]

    @property
    def tile_id(self) -> str:
        return f"tile_{self.index:06d}"


@dataclass
class TilePlan:
    """Tiles of one level of an image; see :func:`plan_tiles`."""

    level: int
    shape: dict[str, int]
    roi: dict[str, tuple[int, int]]
    tile_size: dict[str, int]
    overlap: dict[str, int]
    halo: dict[str, int]
    tiles: list[Tile]
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def axes(self) -> tuple[str, ...]:
        """Tiled axes, in z, y, x order."""
        return tuple(self.roi)

    @property
    def stride(self) -> dict[str, int]:
        """Distance between tile starts; one tile spanning the ROI has the ROI length."""
        return {ax: self.tile_size[ax] - self.overlap[ax] for ax in self.axes}

    @property
    def origin(self) -> dict[str, int]:
        """First pixel of the ROI: the output's pixel 0, in the source level's pixels."""
        return {ax: lo for ax, (lo, _) in self.roi.items()}

    @property
    def output_shape(self) -> dict[str, int]:
        return {ax: hi - lo for ax, (lo, hi) in self.roi.items()}

    def core(self, tile: Tile) -> dict[str, tuple[int, int]]:
        """The part of ``tile`` it writes, in the source level's pixels: up to the
        next tile's start, or to its own end for the last tile of an axis."""
        out = {}
        for ax in self.axes:
            lo, hi = tile.bounds[ax]
            last = tile.grid[ax] == self.counts[ax] - 1
            out[ax] = (lo, hi if last else lo + self.stride[ax])
        return out

    def write_region(self, tile: Tile) -> dict[str, slice]:
        """Where the core of ``tile`` goes in the output (which starts at :attr:`origin`)."""
        return {ax: slice(lo - self.origin[ax], hi - self.origin[ax]) for ax, (lo, hi) in self.core(tile).items()}

    def neighbors(self, tile: Tile) -> dict[str, Tile | None]:
        """The next tile along each tiled axis (``None`` for the last one)."""
        by_grid = {tuple(t.grid[ax] for ax in self.axes): t for t in self.tiles}
        out = {}
        for ax in self.axes:
            key = tuple(tile.grid[a] + (1 if a == ax else 0) for a in self.axes)
            out[ax] = by_grid.get(key)
        return out

    def allocate_kwargs(
        self,
        axes: str,
        *,
        zarr_format: int = 3,
        split: str | None = None,
        crop: Mapping[str, tuple[int | None, int | None]] | None = None,
        inner_chunk: Mapping[str, int] | None = None,
        sizes: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        """Options of :func:`~zarrsmith.utils.allocate.allocate_zarr` for an output of the ROI.

        ``axes`` are the output's axes (e.g. ``"tzyx"`` for labels). The write
        block is the stride on tiled axes, so each tile writes whole blocks.
        ``split`` names untiled axes whose block is one index: by default ``"t"``
        (when the output has it), so jobs can be split per tile *and* per
        timepoint, each writing ``t=slice(k, k + 1)`` and holding one timepoint
        in memory; a job writing every timepoint of its tile writes several
        whole blocks, which is safe too. Other untiled axes (``c``) span the
        whole axis. ``split=""`` keeps every timepoint of a tile in one block:
        fewer, larger files, but only one job per tile.

        On zarr v3 the block is the shard, holding inner chunks of
        ``inner_chunk`` (a divisor of the stride; 1 along t and c); on zarr v2 it
        is the chunk. ``crop`` keeps part of an untiled axis, e.g.
        ``{"t": (0, 5)}`` (in the template's indices); ``sizes`` overrides the
        extent of untiled axes otherwise.
        """
        axes = "".join(axes).lower()
        split = ("t" if "t" in axes else "") if split is None else "".join(split).lower()
        bad = sorted(set(split) - set(axes))
        if bad or set(split) & set(self.axes):
            raise ValueError(
                f"split takes untiled axes of the output ({axes!r}, tiled: {''.join(self.axes)!r}), got {split!r}."
            )
        crop = {str(k).lower(): v for k, v in (crop or {}).items()}
        if set(crop) & set(self.axes):
            raise ValueError("crop is for untiled axes; the tiled axes are cropped by the plan's roi.")
        sizes = {**self.shape, **(sizes or {})}
        for ax, (lo, hi) in crop.items():
            n = int(self.shape.get(ax, sizes.get(ax, 1)))
            lo, hi = 0 if lo is None else int(lo), n if hi is None else int(hi)
            sizes[ax] = hi - lo
        limits = {**DEFAULT_INNER_CHUNK, **(inner_chunk or {})}
        chunks, shards = [], []
        for ax in axes:
            if ax in split:
                block, inner = 1, 1
            elif ax in self.axes:
                block = self.stride[ax]
                if self.counts[ax] == 1:  # one tile: any block covering it works
                    inner = min(limits.get(ax, block), self.output_shape[ax])
                    block = -(-self.output_shape[ax] // inner) * inner
                else:
                    inner = _divisor_at_most(block, limits.get(ax, block))
            else:
                block, inner = int(sizes.get(ax, 1)), 1
            if zarr_format == 2:
                chunks.append(block)
            else:
                chunks.append(inner)
                shards.append(block)
        return {
            "start_level": self.level,
            "crop": {**crop, **self.roi},
            "chunk_size": tuple(chunks),
            "shards": tuple(shards) if zarr_format != 2 else None,
            "zarr_format": zarr_format,
        }

    def load_bounds(self, tile: Tile) -> dict[str, tuple[int, int]]:
        """What a job reads for ``tile``: its bounds plus the halo, clipped to the image."""
        return {ax: (max(lo - self.halo[ax], 0), min(hi + self.halo[ax], self.shape[ax]))
                for ax, (lo, hi) in tile.bounds.items()}

    @property
    def max_load_voxels(self) -> int:
        """Largest number of voxels read for one tile (halo included), over every tiled axis."""
        return max(_volume(self.load_bounds(t)) for t in self.tiles)

    def rows(self) -> list[dict[str, Any]]:
        """One dict per tile: ``tile_index``, ``tile_id``, ``level`` and, per tiled
        axis, ``<ax>_min``/``<ax>_max`` (pixels), ``tile_size_<ax>`` (the requested
        size) and ``tile_overlap_<ax>`` (e.g. for a CSV)."""
        out = []
        for tile in self.tiles:
            row: dict[str, Any] = {"tile_index": tile.index, "tile_id": tile.tile_id, "level": self.level}
            for ax in self.axes:
                row[f"{ax}_min"], row[f"{ax}_max"] = tile.bounds[ax]
            for ax in self.axes:
                row[f"tile_size_{ax}"] = self.tile_size[ax]
                row[f"tile_overlap_{ax}"] = self.overlap[ax]
            out.append(row)
        return out


def _volume(bounds: Mapping[str, tuple[int, int]]) -> int:
    n = 1
    for lo, hi in bounds.values():
        n *= hi - lo
    return n


def _auto_strides(
    lengths: dict[str, int],
    overlaps: dict[str, int],
    halos: dict[str, int],
    budget: float,
    align: Mapping[str, int],
    whole: dict[str, int],
) -> dict[str, int]:
    """Strides along the axes of ``lengths`` (ROI lengths) for tiles of at most
    ``budget`` loaded voxels (tile plus halo on both sides). ``whole`` is what is
    read along an axis kept whole (the ROI plus the halo that lies in the image).

    Every choice of axes kept whole (one tile along them) is tried; the budget
    left is shared evenly by the axes that are cut. Each stride is the largest
    multiple of a power of two within that share (``align`` at most, and at most
    an eighth of the stride, so rounding wastes little), then evened out so
    the tiles of an axis have about the same size. The choice with the least
    area cut between tiles wins (what the overlap, the halo and merging the tiles
    afterwards cost), then the one with the fewest tiles: so tiles are compact,
    not thin slabs. Axes kept whole are left out of the result.
    """
    axes = list(lengths)
    best = None
    error = None
    for keep in itertools.chain.from_iterable(itertools.combinations(axes, k) for k in range(len(axes))):
        cut = [ax for ax in axes if ax not in keep]
        left = float(budget)
        for ax in keep:
            left /= whole[ax]
        share = left ** (1.0 / len(cut))
        strides, counts = {}, {}
        for ax in cut:
            room = int(share) - 2 * halos[ax] - overlaps[ax]  # largest stride that fits
            if room < 1:
                error = (f"max_tile_voxels is too small: along {ax!r} a tile of about {int(share)} pixels "
                         f"cannot hold the overlap ({overlaps[ax]}) and twice the halo ({halos[ax]}).")
                break
            unit = 1
            while unit * 2 <= min(int(align.get(ax, 1)), room // 8):
                unit *= 2
            stride = room - room % unit
            n = max(1, -(-(lengths[ax] - overlaps[ax]) // stride))  # tiles needed at that stride
            even = -(-(lengths[ax] - overlaps[ax]) // n)  # same count, tiles of equal size
            strides[ax] = min(-(-even // unit) * unit, stride)
            counts[ax] = n
        else:
            count, area = 1, 0
            for ax in axes:
                count *= counts.get(ax, 1)
                face = 1
                for other in axes:
                    if other != ax:
                        face *= lengths[other]
                area += (counts.get(ax, 1) - 1) * face  # cut planes across the ROI
            key = (area, count)
            if best is None or key < best[0]:
                best = (key, strides)
    if best is None:
        raise ValueError(error)
    return best[1]


def _divisor_at_most(n: int, limit: int) -> int:
    """Largest divisor of ``n`` not above ``limit``."""
    for d in range(min(int(n), int(limit)), 0, -1):
        if n % d == 0:
            return d
    return 1


def _per_axis(value, axes, name) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        out = {str(k).lower(): v for k, v in value.items() if v is not None}
        unknown = sorted(set(out) - set(axes))
        if unknown:
            raise ValueError(f"{name} has axes {unknown}; the image's spatial axes are {''.join(axes)!r}.")
        return out
    return {ax: value for ax in axes}


def _axis_tiles(start: int, stop: int, size: int, overlap: int, min_size: int) -> list[tuple[int, int]]:
    if size <= overlap:
        raise ValueError(f"The tile size ({size}) must be larger than the overlap ({overlap}).")
    step = size - overlap
    tiles = []
    lo = start
    while True:
        hi = min(lo + size, stop)
        tiles.append((lo, hi))
        if hi >= stop:
            break
        lo += step
    # A last tile too small to process on its own is folded into the one before.
    if len(tiles) > 1 and tiles[-1][1] - tiles[-1][0] < min_size:
        last = tiles.pop()
        tiles[-1] = (tiles[-1][0], last[1])
    return tiles


def plan_tiles(
    shape: Mapping[str, int],
    *,
    tile_size: float | Mapping[str, float] | None = None,
    overlap: float | Mapping[str, float] = 0,
    halo: float | Mapping[str, float] = 0,
    roi: Mapping[str, tuple[float | None, float | None]] | None = None,
    scale: Mapping[str, float] | None = None,
    translation: Mapping[str, float] | None = None,
    physical: bool = False,
    level: int = 0,
    max_tile_voxels: int | None = None,
    resize: float = 1.0,
) -> TilePlan:
    """Cut the spatial axes of one level of an image into overlapping tiles.

    Parameters
    ----------
    shape
        Size of the level by axis name (e.g. ``{"t": 5, "z": 40, "y": 2048,
        "x": 2048}``); z, y and x are tiled, other axes are not.
    tile_size
        Tile size: one value for every tiled axis, or one per axis. An axis
        without a tile size is not split: one tile spans the whole ROI, unless
        ``max_tile_voxels`` cuts it.
    max_tile_voxels
        Most voxels one job may read for a tile, halo included (e.g. what fits in
        GPU memory). Axes without a ``tile_size`` are cut only when a tile would
        exceed it: the ROI stays one tile if it fits. Otherwise there are as few
        tiles as fit, of about equal size: an axis shorter than an even share of
        the budget stays whole (usually z), the others are cut evenly, with strides
        that are multiples of the default inner chunk (16 along z, 256 along y and
        x) when the budget allows, so the output's write blocks stay aligned to
        it. ``tile_size`` given for an axis is kept and counts towards the budget.
        It applies to one timepoint and channel: untiled axes are not counted.
    resize
        Linear factor by which the tool enlarges each tile before processing, so
        the budget is checked against ``voxels * resize ** ndim``. For Cellpose,
        ``30 / diameter`` (it rescales objects to about 30 pixels).
    overlap
        Pixels (or physical length) shared by consecutive tiles along an axis.
        Ignored on an axis with a single tile.
    halo
        Extra context read around each tile (see
        :meth:`zarrsmith.microscope_manager.ZarrManager.read_region`); only
        recorded, and used as the smallest size of a last tile before it is
        folded into the one before it (at least 2 pixels).
    roi
        ``{axis: (start, stop)}`` to tile; the whole level by default.
    scale, translation, physical
        With ``physical=True``, ``roi``, ``tile_size``, ``overlap`` and ``halo``
        are in physical units and converted with the level's ``scale`` and
        ``translation`` (NGFF 0.6: a translation is the first pixel's centre).
        Lengths are rounded up to whole pixels; the ROI takes every pixel it
        overlaps (see :func:`zarrsmith.utils.region.physical_to_pixel_bounds`).
    level
        Pyramid level the pixels refer to (recorded in the plan).
    """
    shape = {str(k).lower(): int(v) for k, v in shape.items()}
    axes = tuple(ax for ax in TILED_AXES if ax in shape)
    if not axes:
        raise ValueError(f"The image has no spatial axis to tile (axes {''.join(shape)!r}).")
    if physical and scale is None:
        raise ValueError("physical=True needs the level's scale.")

    def to_pixels(values, name):
        values = _per_axis(values, axes, name)
        if physical:
            return {ax: physical_to_pixels(float(v), float(scale[ax])) for ax, v in values.items()}
        bad = {ax: v for ax, v in values.items() if float(v) != int(v)}
        if bad:
            raise ValueError(f"{name} in pixels must be whole numbers, got {bad}; use physical=True for units.")
        return {ax: int(v) for ax, v in values.items()}

    roi_in = _per_axis(roi, axes, "roi")
    if physical:
        bounds = physical_to_pixel_bounds(roi_in, scale, translation, shape)
    else:
        bounds = {}
        for ax, (lo, hi) in roi_in.items():
            lo = 0 if lo is None else int(lo)
            hi = shape[ax] if hi is None else int(hi)
            if not 0 <= lo < hi <= shape[ax]:
                raise ValueError(f"roi along {ax!r} [{lo}, {hi}) is empty or outside the image (size {shape[ax]}).")
            bounds[ax] = (lo, hi)
    roi_px = {ax: bounds.get(ax, (0, shape[ax])) for ax in axes}

    sizes = to_pixels(tile_size, "tile_size")
    overlaps = to_pixels(overlap, "overlap")
    halos = to_pixels(halo, "halo")
    for name, values in (("tile_size", sizes), ("overlap", overlaps), ("halo", halos)):
        for ax, v in values.items():
            if v < (1 if name == "tile_size" else 0):
                raise ValueError(f"{name} along {ax!r} must be {'> 0' if name == 'tile_size' else '>= 0'}, got {v}.")

    def build(sizes: dict[str, int]) -> TilePlan:
        per_axis_tiles, size_out, overlap_out = {}, {}, {}
        for ax in axes:
            lo, hi = roi_px[ax]
            if ax not in sizes or sizes[ax] >= hi - lo:
                # Not split: one tile spans the ROI.
                per_axis_tiles[ax] = [(lo, hi)]
                size_out[ax], overlap_out[ax] = hi - lo, 0
                continue
            per_axis_tiles[ax] = _axis_tiles(lo, hi, sizes[ax], overlaps.get(ax, 0), max(halos.get(ax, 0), 2))
            size_out[ax], overlap_out[ax] = sizes[ax], overlaps.get(ax, 0)

        tiles = []
        for index, combo in enumerate(itertools.product(*(enumerate(per_axis_tiles[ax]) for ax in axes)), start=1):
            tiles.append(Tile(
                index=index,
                grid={ax: i for ax, (i, _) in zip(axes, combo)},
                bounds={ax: b for ax, (_, b) in zip(axes, combo)},
            ))
        return TilePlan(
            level=level, shape=shape, roi=roi_px, tile_size=size_out, overlap=overlap_out,
            halo={ax: halos.get(ax, 0) for ax in axes}, tiles=tiles,
            counts={ax: len(per_axis_tiles[ax]) for ax in axes},
        )

    if max_tile_voxels is None:
        return build(sizes)
    if not float(resize) > 0:
        raise ValueError(f"resize must be > 0, got {resize!r}.")
    limit = float(max_tile_voxels) / float(resize) ** len(axes)  # in source voxels
    plan = build(sizes)
    if plan.max_load_voxels <= limit:
        return plan  # fits as asked (one tile on every axis without a tile size)
    free = [ax for ax in axes if ax not in sizes]
    if not free:
        raise ValueError(
            f"Tiles of tile_size {sizes} read up to {plan.max_load_voxels * float(resize) ** len(axes):.0f} "
            f"voxels, more than max_tile_voxels={max_tile_voxels}."
        )
    fixed = 1
    for ax in axes:
        if ax in sizes:
            fixed *= min(sizes[ax], roi_px[ax][1] - roi_px[ax][0]) + 2 * halos.get(ax, 0)
    lengths = {ax: roi_px[ax][1] - roi_px[ax][0] for ax in free}
    ov = {ax: overlaps.get(ax, 0) for ax in free}
    hl = {ax: halos.get(ax, 0) for ax in free}
    whole = {ax: min(roi_px[ax][1] + hl[ax], shape[ax]) - max(roi_px[ax][0] - hl[ax], 0) for ax in free}
    budget = limit / fixed
    for _ in range(50):  # rounding, folded last tiles: shrink until every tile fits
        strides = _auto_strides(lengths, ov, hl, budget, DEFAULT_INNER_CHUNK, whole)
        plan = build({**sizes, **{ax: s + ov[ax] for ax, s in strides.items()}})
        if plan.max_load_voxels <= limit:
            return plan
        budget *= 0.9
    raise ValueError(f"Could not fit tiles within max_tile_voxels={max_tile_voxels}.")
