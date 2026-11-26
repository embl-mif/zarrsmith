"""Chunk / shape helpers shared by the pyramid builder, the Zarr writer, the CLI and napari."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import dask.array as da
import numpy as np

from .axes import metadata_axes, recorded_voxel_size, spatial_axes_in_order, spatial_values_for_axes

#: The default number of levels stops when no spatial axis of the coarsest level is above this.
MAX_COARSEST_SIZE = 2048


def shape_tuple(value: Any, ndim: int) -> tuple[int, ...] | None:
    """Return a positive shape tuple of length ``ndim`` from ``value``, or ``None``."""
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        candidate = tuple(int(v) for v in value)
    except TypeError:
        return None
    if len(candidate) != ndim or any(v <= 0 for v in candidate):
        return None
    return candidate


def get_chunks(arr: da.Array) -> tuple[int, ...]:
    """Return one normalized chunk tuple for a dask array."""
    if hasattr(arr, "chunksize") and arr.chunksize is not None:
        return tuple(int(x) for x in arr.chunksize)
    return tuple(int(c[0]) for c in arr.chunks)


def rechunk_to_shape(arr: da.Array, chunks: Sequence[int]) -> da.Array:
    """Rechunk ``arr`` to ``chunks``, clipped so no axis exceeds the array's extent."""
    normalized = tuple(max(1, min(int(c), int(s))) for c, s in zip(chunks, arr.shape))
    if normalized == get_chunks(arr):
        return arr
    return arr.rechunk(normalized)


#: Default (uncompressed) chunk size in MB: the unit a reader fetches. 1-4 MB keeps reads of
#: one plane (viewers) or of a tile with its halo (segmentation jobs) close to what they need,
#: while compressed chunks stay large enough for network filesystems; files stay few, since
#: the chunks are grouped into shards (at most 512 MB, see ``ZarrWriteConfig.shard_target_mb``).
DEFAULT_CHUNK_MB = 4.0
#: Most z planes in a default chunk. A viewer showing one plane reads whole chunks, so this
#: bounds how much more it reads than it shows (32 planes: 32 times). Lifted once the chunk
#: spans the image in y and x, where more planes are the only way to reach the chunk size.
DEFAULT_MAX_CHUNK_PLANES = 32


def _physical_size(axes: str, voxel_size) -> dict[str, float]:
    """Voxel size of each spatial axis of ``axes`` from ``voxel_size`` (a mapping axis -> size,
    or the sizes of the spatial axes in order), 1.0 where it is unknown (``None``) or unusable.
    With an unknown size, chunks are counted in voxels along that axis."""
    spatial = spatial_axes_in_order(axes)
    if voxel_size is None:
        return {ax: 1.0 for ax in spatial}
    if not isinstance(voxel_size, dict):
        voxel_size = dict(zip(spatial, voxel_size))

    def usable(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) and v > 0 else None

    known = {ax: usable(voxel_size.get(ax)) for ax in spatial}
    if any(v is None for v in known.values()):
        # an unknown size cannot be compared with the others: count every axis in voxels
        return {ax: 1.0 for ax in spatial}
    return known


def auto_chunk_shape(axes: str, shape: Sequence[int], dtype, target_mb: float | None = None,
                     voxel_size=None) -> tuple[int, ...]:
    """Chunk shape of about ``target_mb`` (default :data:`DEFAULT_CHUNK_MB`) uncompressed, in ``axes`` order.

    Chunks are 1 along ``t`` and ``c`` and about cubic in physical units: the spatial size
    that is smallest in micrometers (``voxel_size``, a mapping axis -> size or the sizes of
    the spatial axes in order; ``None`` for an unknown size, the chunk is then cubic in voxels)
    is doubled, one at a time, while the chunk stays within ``target_mb``. Z holds at most
    :data:`DEFAULT_MAX_CHUNK_PLANES` planes, so a viewer showing one plane does not read many.
    For uint16 at 4 MB: ``32 x 256 x 256`` (z steps up to 8 times the pixel size, or unknown),
    ``16 x 256 x 512`` (z steps 16 times the pixel size); ``1024 x 2048`` in 2D. An axis
    smaller than the chunk would be takes the whole image, and the others grow instead.
    """
    target_mb = DEFAULT_CHUNK_MB if target_mb is None else float(target_mb)
    target = max(1, int(target_mb * 1024 * 1024) // np.dtype(dtype).itemsize)
    sizes = {ax: max(1, int(s)) for ax, s in zip(axes, shape)}
    spatial = [ax for ax in "zyx" if ax in sizes]
    physical = _physical_size(axes, voxel_size)
    plane = [ax for ax in ("y", "x") if ax in sizes]
    chunk = {ax: 1 for ax in spatial}

    def can_grow(ax):
        if chunk[ax] >= sizes[ax]:
            return False
        if ax == "z" and chunk["z"] >= DEFAULT_MAX_CHUNK_PLANES:
            return all(chunk[p] >= sizes[p] for p in plane)  # y and x span the image
        return True

    while True:
        total = math.prod(chunk.values())
        growable = sorted((ax for ax in spatial if can_grow(ax)),
                          key=lambda ax: (chunk[ax] * physical[ax], "xyz".index(ax)))
        for ax in growable:
            new = min(2 * chunk[ax], sizes[ax])
            if total // chunk[ax] * new <= target:
                chunk[ax] = new
                break
        else:
            break
    return tuple(chunk.get(ax, 1) for ax in axes)


def default_chunk_size(metadata: dict[str, Any], max_size_mb: float | None = None):
    """Default chunk shape of a dataset (``metadata['axes']``, ``['size']``, ``['dtype']`` and,
    when present, ``['scales']``): :func:`auto_chunk_shape` of level 0, about ``max_size_mb``
    (default :data:`DEFAULT_CHUNK_MB`, 4 MB) uncompressed, with the level-0 voxel sizes the
    file recorded (:func:`~zarrsmith.utils.axes.recorded_voxel_size`).
    Returns ``(chunks, chunk size in MB, number of chunks along z, y, x)``.
    """
    axes = metadata_axes(metadata)
    shape = [int(n) for n in metadata["size"][0]]
    dtype = np.dtype(metadata.get("dtype", "uint16"))
    chunks = auto_chunk_shape(axes, shape, dtype, max_size_mb, voxel_size=recorded_voxel_size(metadata))
    size_map = dict(zip(axes, shape))
    chunk = dict(zip(axes, chunks))
    n_chunks = {ax: math.ceil(size_map.get(ax, 1) / chunk.get(ax, 1)) for ax in ("z", "y", "x")}
    return chunks, math.prod(chunks) * dtype.itemsize / 1024 / 1024, n_chunks


def default_num_levels(metadata: dict[str, Any], downscale_factor: Any = 2) -> int:
    """Default number of pyramid levels: until no spatial axis of the coarsest level is above
    MAX_COARSEST_SIZE voxels. ``downscale_factor``: one value, or one per spatial axis (or Z, Y, X)."""
    axes = metadata_axes(metadata)
    size_map = dict(zip(axes, metadata["size"][0]))
    shape = [int(size_map[ax]) for ax in axes if ax in "zyx"]
    if not shape:
        return 1
    factors = spatial_values_for_axes(downscale_factor, axes, name="downscale_factor", allow_float=False)
    n = 1
    while any(s > MAX_COARSEST_SIZE and f > 1 for s, f in zip(shape, factors)):  # factor 1: never smaller
        n += 1
        shape = [max(1, s // int(f)) for s, f in zip(shape, factors)]
    return n


# ---------------------------------------------------------------------------
# Level-0 layout (chunks, shards, write blocks): what ``zarrsmith convert/fuse --dry_run``
# prints and the napari widgets show, with the functions the writer uses.
# ---------------------------------------------------------------------------

def manual_chunks(axes: str, shape: Sequence[int], zyx: Sequence[int]) -> tuple[int, ...]:
    """Chunk shape in ``axes`` order from a ``(z, y, x)`` chunk: 1 along ``t`` and ``c``,
    clipped to the image (z is ignored without a z axis)."""
    wanted = dict(zip("zyx", (int(v) for v in zyx)))
    return tuple(max(1, min(wanted.get(ax, 1), int(s))) for ax, s in zip(axes, shape))


def block_count(shape: Sequence[int], block: Sequence[int]) -> int:
    """Number of ``block``-shaped pieces that tile ``shape``."""
    return int(math.prod(-(-int(s) // int(b)) for s, b in zip(shape, block)))


def level0_layout(axes: str, shape: Sequence[int], dtype, chunks: Sequence[int], *, zarr_format: int = 3,
                  shards: Any = "auto", shard_mb: float | None = None,
                  shard_exclude_axes: Sequence[str] | None = None) -> dict:
    """Level-0 chunks and shards that the OME-Zarr writer makes for ``chunks`` and the sharding
    options (``shards``: ``"auto"``, a shard shape in ``axes`` order, or ``None``; ``shard_mb``
    and ``shard_exclude_axes`` default to the writer's), computed with the writer's functions.

    Returns ``chunks``, ``shards`` (``None`` without shards, or for zarr v2), ``exclude`` (axes
    shards never merge), ``block_bytes`` and ``n_blocks`` (size and number of the write blocks:
    shards, or chunks without shards).
    """
    from .ngff import DEFAULT_SHARD_EXCLUDE_AXES, DEFAULT_SHARD_TARGET_MB, _auto_shard_for_level, _snap_shard_shape

    chunks = tuple(max(1, min(int(c), int(s))) for c, s in zip(chunks, shape))
    exclude = tuple(dict.fromkeys(DEFAULT_SHARD_EXCLUDE_AXES if shard_exclude_axes is None else shard_exclude_axes))
    shard = None
    if int(zarr_format) == 3 and shards is not None:
        if isinstance(shards, str):
            if shards != "auto":
                raise ValueError(f"shards must be 'auto', a shape or None, got {shards!r}.")
            target = float(DEFAULT_SHARD_TARGET_MB if shard_mb is None else shard_mb)
            shard = _auto_shard_for_level(shape, chunks, np.dtype(dtype).itemsize, int(target * 1024 * 1024),
                                          axes=axes, exclude_axes=exclude)
        else:
            shard = _snap_shard_shape(shards, shape, chunks)
        shard = None if shard is None else tuple(min(int(b), int(s)) for b, s in zip(shard, shape))
    block = shard or chunks
    return {"chunks": chunks, "shards": shard, "exclude": exclude,
            "block_bytes": math.prod(int(b) for b in block) * np.dtype(dtype).itemsize,
            "n_blocks": block_count(shape, block)}


def chunk_shard_layout(axes: str, shape: Sequence[int], dtype, *, chunk_auto: bool = True,
                       chunk_mb: float = DEFAULT_CHUNK_MB, chunk_zyx: Sequence[int] = (32, 256, 256),
                       zarr_format: int = 3, shards: str = "auto", shard_mb: float | None = None,
                       shard_exclude_axes: Sequence[str] = ("t", "c"), voxel_size=None) -> dict:
    """:func:`level0_layout` with the chunks chosen as the napari widgets do: automatic
    (:func:`auto_chunk_shape` of ``chunk_mb`` and ``voxel_size``) or ``chunk_zyx``."""
    chunks = (auto_chunk_shape(axes, shape, dtype, chunk_mb, voxel_size=voxel_size) if chunk_auto
              else manual_chunks(axes, shape, chunk_zyx))
    return level0_layout(axes, shape, dtype, chunks, zarr_format=zarr_format, shards=shards,
                         shard_mb=shard_mb, shard_exclude_axes=shard_exclude_axes)


def layout_summary(axes: str, dtype, layout: dict) -> str:
    """``chunks 32x256x256 (4.0 MB), shards 160x1280x1280 (500 MB, 24 shards)``, followed by
    ``, 8 fusion workers (auto)`` when ``layout`` has ``workers``."""
    itemsize = np.dtype(dtype).itemsize

    def shape(block):
        spatial = [int(b) for ax, b in zip(axes, block) if ax in "zyx"]
        mb = math.prod(int(b) for b in block) * itemsize / 2**20
        return "x".join(str(b) for b in spatial), f"{mb:.1f} MB" if mb < 10 else f"{mb:.0f} MB"

    text = "chunks {} ({})".format(*shape(layout["chunks"]))
    if layout["shards"] is not None:
        dims, size = shape(layout["shards"])
        n = layout["n_blocks"]
        text += f", shards {dims} ({size}, {n} shard{'s' if n != 1 else ''})"
    else:
        text += ", no shards"
    if "workers" not in layout:
        return text
    workers = layout["workers"]
    return text + f", {workers} fusion worker{'s' if workers != 1 else ''}" + (" (auto)" if layout.get("auto_workers") else "")


def output_summary_line(axes: str, sizes: dict, dtype, voxel_size: Sequence[float], units: Sequence[str] = (),
                        drop_singleton: bool = True) -> str:
    """One line such as ``2 ch x 301 x 1880 x 2566 px (czyx), 2.90 GB, voxel 2 x 0.325 x 0.325
    micrometer`` (MB below 0.1 GB), in the format of the mosaic fusion plan: ``sizes`` maps every
    axis of ``axes`` to its size, ``voxel_size`` and ``units`` are those of the spatial axes, and
    with ``drop_singleton`` the t/c/z axes of size 1 are left out as the writer does."""
    out_axes = "".join(ax for ax in axes if not (drop_singleton and ax in "tcz" and int(sizes[ax]) == 1))
    labels = {"t": "t", "c": "ch"}
    parts = [f"{sizes[ax]} {labels[ax]}" for ax in out_axes if ax in labels]
    spatial = spatial_axes_in_order(out_axes)
    parts.append(" x ".join(str(sizes[ax]) for ax in spatial) + " px")
    nbytes = math.prod(int(sizes[ax]) for ax in out_axes) * np.dtype(dtype).itemsize
    size = f"{nbytes / 1e9:.2f} GB" if nbytes >= 1e8 else f"{nbytes / 1e6:.1f} MB"
    all_spatial = spatial_axes_in_order(axes)
    vox = dict(zip(all_spatial, voxel_size or ()))
    unit_of = dict(zip(all_spatial, units or ()))
    voxel = " x ".join(f"{float(vox[ax]):.3g}" for ax in spatial if ax in vox)
    unit = " ".join(dict.fromkeys(str(unit_of[ax]) for ax in spatial if unit_of.get(ax)))
    return f"{' x '.join(parts)} ({out_axes}), {size}, voxel {voxel} {unit}".rstrip()
