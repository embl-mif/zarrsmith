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

import numpy as np

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


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _axis_option(lo: int, hi: int, extent: int, n: int, overlap: int, halo: int, align: int) -> tuple[int, int, int]:
    """``(stride, count, load)`` along one axis cut into about ``n`` tiles.

    The stride is the even share of the ROI ``[lo, hi)`` rounded up to a multiple
    of a power of two (``align`` at most, and at most an eighth of the stride, so
    rounding wastes little); rounding up can leave fewer tiles than ``n``. ``count``
    is the number of tiles :func:`_axis_tiles` then makes (a short last tile folded
    into the one before) and ``load`` the largest length a tile reads, halo included
    and clipped to the image (``extent`` long). ``count == 1``: the axis is not split.
    """
    length = hi - lo
    whole = min(hi + halo, extent) - max(lo - halo, 0)
    if n <= 1 or length - overlap <= 1:
        return length, 1, whole
    even = _ceil_div(length - overlap, n)
    unit = 1
    while unit * 2 <= min(int(align), even // 8):
        unit *= 2
    stride = _ceil_div(even, unit) * unit
    size = stride + overlap
    if size >= length:
        return length, 1, whole
    m = _ceil_div(length - size, stride) + 1
    last_start = lo + (m - 1) * stride
    if m > 1 and hi - last_start < max(halo, 2):  # folded into the tile before
        m -= 1
    # Tiles k < m - 1 are `size` long, the last one ends at hi; only tiles near the
    # image borders lose part of their halo, so the longest load is among these.
    near = _ceil_div(2 * halo + 1, stride) + 1
    ks = set(range(min(m, near))) | set(range(max(0, m - near), m)) | {m // 2}
    load = 0
    for k in ks:
        start = lo + k * stride
        end = hi if k == m - 1 else min(start + size, hi)
        load = max(load, min(end + halo, extent) - max(start - halo, 0))
    return stride, m, load


def _axis_choices(lo: int, hi: int, extent: int, overlap: int, halo: int, align: int) -> list[tuple[int, int, int]]:
    """Every ``(count, load, stride)`` reachable along one axis (see :func:`_axis_option`),
    by increasing count, keeping only the counts that read less than all fewer counts."""
    best: dict[int, tuple[int, int]] = {}
    span = max(1, hi - lo - overlap)
    n = 1
    while n <= span:
        stride, count, load = _axis_option(lo, hi, extent, n, overlap, halo, align)
        if count not in best or load < best[count][0]:
            best[count] = (load, stride)
        even = _ceil_div(span, n)
        if even <= 1:
            break
        n = _ceil_div(span, even - 1)  # next n with a smaller even share (about 2 * sqrt(span) of them)
    out, smallest = [], None
    for count in sorted(best):
        load, stride = best[count]
        if smallest is None or load < smallest:
            out.append((count, load, stride))
            smallest = load
    return out


def _auto_strides(
    roi: dict[str, tuple[int, int]],
    extents: dict[str, int],
    overlaps: dict[str, int],
    halos: dict[str, int],
    budget: float,
    align: Mapping[str, int],
) -> dict[str, int]:
    """Strides along the axes of ``roi`` for tiles of at most ``budget`` loaded voxels
    (tile plus halo, clipped to the image of ``extents``). Axes left whole (one tile)
    are left out of the result.

    Of the tile counts per axis that fit, the one with the least area cut between
    tiles wins (what the overlap, the halo and merging the tiles afterwards cost),
    then the one with the fewest tiles: so tiles are compact, not thin slabs. Tiles
    are a grid, so the largest load is the product of the largest load along each
    axis (:func:`_axis_choices`). Every combination of counts is considered: depth
    first, dropping a branch as soon as its cut area exceeds the best found, and
    taking along the last axis the fewest tiles that fit.
    """
    axes = list(roi)
    lengths = {ax: hi - lo for ax, (lo, hi) in roi.items()}
    faces = {ax: int(np.prod([lengths[o] for o in axes if o != ax])) for ax in axes}
    choices = {ax: _axis_choices(*roi[ax], extents[ax], overlaps[ax], halos[ax], align.get(ax, 1)) for ax in axes}
    best: list = [None]  # ((area, count), {axis: choice})

    def search(i: int, area: int, count: int, load: int, chosen: dict) -> None:
        ax = axes[i]
        if i == len(axes) - 1:  # loads fall as counts rise: the first that fits is the fewest tiles
            for c in choices[ax]:
                if load * c[1] <= budget:
                    key = (area + (c[0] - 1) * faces[ax], count * c[0])
                    if best[0] is None or key < best[0][0]:
                        best[0] = (key, {**chosen, ax: c})
                    return
            return
        for c in choices[ax]:
            a = area + (c[0] - 1) * faces[ax]
            if best[0] is not None and a > best[0][0][0]:
                break  # more tiles along this axis only add area
            search(i + 1, a, count * c[0], load * c[1], {**chosen, ax: c})

    search(0, 0, 1, 1, {})
    if best[0] is None:
        smallest = int(np.prod([choices[ax][-1][1] for ax in axes]))
        raise ValueError(
            f"max_tile_voxels is too small: even the smallest tiles read {smallest} voxels "
            f"(overlap {overlaps} and twice the halo {halos} included), more than {budget:.0f}."
        )
    return {ax: c[2] for ax, c in best[0][1].items() if c[0] > 1}


def _divisor_at_most(n: int, limit: int) -> int:
    """Largest divisor of ``n`` not above ``limit``."""
    for d in range(min(int(n), int(limit)), 0, -1):
        if n % d == 0:
            return d
    return 1


def _values_for_given_axes(value, axes, name) -> dict[str, float]:
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
        exceed it: the ROI stays one tile if it fits. Otherwise every number of
        tiles along every axis is considered, and of those that fit, the one with
        the least area cut between tiles wins, then the one with the fewest tiles
        (so a short axis, usually z, stays whole, and tiles are compact). Tiles of
        an axis have about equal size, with strides that are multiples of the
        default inner chunk (16 along z, 256 along y and x) when the budget allows,
        so the output's write blocks stay aligned to it. ``tile_size`` given for an
        axis is kept and counts towards the budget.
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
        values = _values_for_given_axes(values, axes, name)
        if physical:
            return {ax: physical_to_pixels(float(v), float(scale[ax])) for ax, v in values.items()}
        bad = {ax: v for ax, v in values.items() if float(v) != int(v)}
        if bad:
            raise ValueError(f"{name} in pixels must be whole numbers, got {bad}; use physical=True for units.")
        return {ax: int(v) for ax, v in values.items()}

    roi_in = _values_for_given_axes(roi, axes, "roi")
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
    ov = {ax: overlaps.get(ax, 0) for ax in free}
    strides = _auto_strides({ax: roi_px[ax] for ax in free}, {ax: shape[ax] for ax in free}, ov,
                            {ax: halos.get(ax, 0) for ax in free}, limit / fixed, DEFAULT_INNER_CHUNK)
    plan = build({**sizes, **{ax: s + ov[ax] for ax, s in strides.items()}})
    if plan.max_load_voxels > limit:  # the search computes the loads exactly: should not happen
        raise RuntimeError(f"Tiles read {plan.max_load_voxels} voxels, more than the limit {limit:.0f}.")
    return plan


#: Columns of :func:`write_tiles_csv` (and ``zarrsmith plan-tiles``).
TILES_CSV_COLUMNS = [
    "dataset_id", "source_dataset_id", "tile_id", "tile_index", "input_uri",
    "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "resolution_level",
    "tile_overlap_x", "tile_overlap_y", "tile_overlap_z",
    "tile_size_x", "tile_size_y", "tile_size_z",
    "right_dataset_id", "right_tile_index",
    "bottom_dataset_id", "bottom_tile_index",
    "behind_dataset_id", "behind_tile_index",
]

#: Forward neighbour of a tile along each axis, as the tile CSV names it.
NEIGHBOR_DIRECTIONS = {"x": "right", "y": "bottom", "z": "behind"}


def tiles_table(plan: TilePlan, source_id: str, input_uri: str = "") -> list[dict[str, Any]]:
    """One row per tile of ``plan`` (the columns of :data:`TILES_CSV_COLUMNS`).

    ``dataset_id`` is ``<source_id>__<tile_id>``. Bounds are pixels of the plan's level,
    half-open; ``tile_size_<ax>``/``tile_overlap_<ax>`` are the planned values (an axis left
    whole reports its ROI length and overlap 0). Each row also names its forward neighbours:
    ``right_*`` (next tile along x), ``bottom_*`` (along y), ``behind_*`` (along z), blank at
    the end of an axis, so every pair of touching tiles is listed once, from the lower tile.
    Columns of an axis the image does not have are blank.
    """
    def dataset_id(tile: Tile) -> str:
        return f"{source_id}__{tile.tile_id}"

    rows = []
    for tile in plan.tiles:
        row: dict[str, Any] = {key: "" for key in TILES_CSV_COLUMNS}
        row.update(dataset_id=dataset_id(tile), source_dataset_id=source_id, tile_id=tile.tile_id,
                   tile_index=tile.index, input_uri=input_uri, resolution_level=plan.level)
        for ax in plan.axes:
            row[f"{ax}_min"], row[f"{ax}_max"] = tile.bounds[ax]
            row[f"tile_overlap_{ax}"] = plan.overlap[ax]
            row[f"tile_size_{ax}"] = plan.tile_size[ax]
        for ax, neighbor in plan.neighbors(tile).items():
            if neighbor is not None:
                row[f"{NEIGHBOR_DIRECTIONS[ax]}_dataset_id"] = dataset_id(neighbor)
                row[f"{NEIGHBOR_DIRECTIONS[ax]}_tile_index"] = neighbor.index
        rows.append(row)
    return rows


def write_tiles_csv(plan: TilePlan, path, source_id: str, input_uri: str = "") -> int:
    """Write :func:`tiles_table` to the CSV ``path``; returns the number of tiles."""
    import csv

    rows = tiles_table(plan, source_id, input_uri)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TILES_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
