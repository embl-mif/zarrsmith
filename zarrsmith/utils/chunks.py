"""Chunk / shape helpers shared by the pyramid builder, the Zarr writer, the CLI and napari."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import dask.array as da
import numpy as np

from .axes import metadata_axes, spatial_values_for_axes

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
#: the chunks are grouped into shards (about 512 MB, see ``ZarrWriteConfig.shard_target_mb``).
DEFAULT_CHUNK_MB = 4.0
#: Most z planes in a default chunk, so a viewer showing one plane reads at most this many.
DEFAULT_CHUNK_Z = 16


def default_chunk_size(metadata: dict[str, Any], max_size_mb: float | None = None):
    """Default chunk shape of a dataset (``metadata['axes']``, ``['size']`` and ``['dtype']``).

    Chunks are about ``max_size_mb`` (default :data:`DEFAULT_CHUNK_MB`, 4 MB) uncompressed:
    T and C chunks are 1; Z holds up to :data:`DEFAULT_CHUNK_Z` planes; Y and X grow by
    powers of two, about square, until the chunk would exceed the size (e.g. 16 x 256 x 512
    for uint16, 16 x 256 x 256 for uint32 labels, 1024 x 2048 for a 2D uint16 image); Z
    grows further only once Y and X span the image. No axis exceeds the image. Returns
    ``(chunks, chunk size in MB, number of chunks along z, y, x)``.
    """
    axes = metadata_axes(metadata)
    size_map = {ax: int(n) for ax, n in zip(axes, metadata["size"][0])}
    itemsize = np.dtype(metadata.get("dtype", "uint16")).itemsize
    target = float(DEFAULT_CHUNK_MB if max_size_mb is None else max_size_mb) * 1024 * 1024
    spatial = [ax for ax in ("z", "y", "x") if ax in size_map]
    chunk = {ax: 1 for ax in spatial}

    def nbytes(c):
        return itemsize * math.prod(c.values())

    def grow(ax, limit):
        """Double ``ax`` (up to ``limit``) if the chunk still fits; whether it grew."""
        new = min(limit, chunk[ax] * 2)
        if new <= chunk[ax] or nbytes({**chunk, ax: new}) > target:
            return False
        chunk[ax] = new
        return True

    if "z" in chunk:  # up to DEFAULT_CHUNK_Z planes, fewer if they alone exceed the size
        while grow("z", min(size_map["z"], DEFAULT_CHUNK_Z)):
            pass
    plane = [ax for ax in ("y", "x") if ax in chunk]
    while any(grow(ax, size_map[ax]) for ax in sorted(plane, key=lambda a: (chunk[a], a == "y"))):
        pass  # the smaller of y and x first (x on a tie): about square
    if "z" in chunk and all(chunk[ax] == size_map[ax] for ax in plane):
        while grow("z", size_map["z"]):
            pass
    chunks = tuple(chunk.get(ax, 1) for ax in axes)
    n_chunks = {ax: math.ceil(size_map.get(ax, 1) / chunk.get(ax, 1)) for ax in ("z", "y", "x")}
    return chunks, nbytes(chunk) / 1024 / 1024, n_chunks


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
