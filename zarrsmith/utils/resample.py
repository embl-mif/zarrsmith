"""Resample a dataset to a new pixel size.

Every output pixel ``i`` samples the input at ``i * factor``, with
``factor = new pixel size / old pixel size``. The grid is global, so the result
does not depend on the chunking or on the batches, and index 0 stays in place:
this is the convention of the pyramid builder, and the output overlays the input
in a viewer.

:func:`resample_levels` is lazy and suits data that fit in one dask graph.
:func:`write_resampled` scales to large stores: it computes one batch at a time,
so the graph and the memory depend on the batch and not on the dataset.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from math import ceil, floor
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import zarr
from dask.array.overlap import overlap

from .axes import normalize_axes, spatial_axes_in_order, spatial_axis_indices, spatial_values_for_axes
from .dataset_ops import infer_downscale_factors
from .pyramid import build_pyramid

#: Keeps the dask graph of one batch small, whatever the memory available.
MAX_BLOCKS_PER_BATCH = 2048


def _resample_axis(block, coords, width, axis, nearest):
    """Sample ``block`` along ``axis`` at the (fractional) positions ``coords``.

    The triangular kernel spans ``width`` input pixels on each side: with a width
    of 1 this is linear interpolation, and wider it averages all the input pixels
    under a new one (anti-aliasing), computed only at the output positions.
    """
    if nearest:
        return np.take(block, np.floor(coords + 0.5).astype(np.intp), axis=axis, mode="clip")

    float_type = np.result_type(block.dtype, np.float32)  # float64 only if float32 would lose precision
    block = block.astype(float_type, copy=False)  # casting once is far faster than in every product
    positions = np.ceil(coords - width) + np.arange(ceil(2 * width))[:, None]
    weights = np.maximum(1 - np.abs(positions - coords) / width, 0)
    weights = (weights / weights.sum(axis=0)).astype(float_type)

    shape = list(block.shape)
    shape[axis] = len(coords)
    result, term = np.zeros(shape, float_type), np.empty(shape, float_type)
    shape = [-1 if a == axis else 1 for a in range(block.ndim)]
    for position, weight in zip(positions.astype(np.intp), weights):
        if weight.any():  # integer factors land on existing pixels: fewer terms
            # mode="clip" repeats the edge pixels of the dataset
            np.take(block, position, axis=axis, out=term, mode="clip")
            term *= weight.reshape(shape)
            result += term
    return result


def _resample_block(block, plan, nearest, block_id):
    """Resample one chunk (with its overlap) axis by axis."""
    dtype = block.dtype
    for axis, factor, width, depth, in_edges, out_edges in plan:
        i = block_id[axis]
        start = max(in_edges[i] - depth, in_edges[0])  # position of this block in the input
        coords = np.arange(out_edges[i], out_edges[i + 1]) * factor - start
        block = _resample_axis(block, coords, width, axis, nearest)
    if block.dtype == dtype:
        return block
    return (np.rint(block) if dtype.kind in "iub" else block).astype(dtype)


def _effective_factors(factors: Mapping[int, float]) -> dict[int, float]:
    """Drop the axes that do not change, so they are never read with an overlap."""
    return {axis: float(factor) for axis, factor in factors.items() if not np.isclose(factor, 1)}


def _output_shape(shape: Sequence[int], factors: Mapping[int, float]) -> tuple[int, ...]:
    """An axis of length ``n`` becomes ``ceil(n / factor)`` long, as in the pyramid builder."""
    return tuple(
        ceil(round(n / factors[axis], 6)) if axis in factors else int(n)
        for axis, n in enumerate(shape)
    )


def resample_region(
    source: da.Array,
    factors: Mapping[int, float],
    region: Sequence[slice],
    chunks: Sequence[int],
    *,
    nearest: bool = False,
) -> da.Array:
    """Lazily compute ``region`` of the output of resampling ``source``.

    ``factors`` must come from :func:`_effective_factors`, and ``region`` holds one
    slice per axis, in output pixels. Output blocks are multiples of ``chunks``
    from the start of the region. Only the input under the region is read.
    """
    plan, depths, reads, in_chunks, out_chunks = [], {}, [], [], []
    for axis, (piece, chunk) in enumerate(zip(region, chunks)):
        start, stop = piece.start, piece.stop
        if axis not in factors:
            edges = [*range(start, stop, chunk), stop]
            reads.append(slice(start, stop))
            in_chunks.append(tuple(np.diff(edges).tolist()))
            out_chunks.append(in_chunks[-1])
            continue

        factor = factors[axis]
        width = 0 if nearest else max(factor, 1)
        # Striding by a whole number needs no neighbour. Otherwise: the kernel,
        # plus the rounding of the chunk edges.
        depth = 0 if nearest and factor.is_integer() else ceil(width) + 2
        first = max(floor(start * factor) - depth, 0)
        last = min(floor(stop * factor) + depth, source.shape[axis])

        # Every input chunk must hold the overlap, so small chunks are grouped.
        step = chunk
        while floor(step * factor) < max(depth, 1):
            step += chunk
        out_edges = [*range(start, stop, step), stop]
        in_edges = [first, *(floor(edge * factor) for edge in out_edges[1:-1]), last]
        while len(out_edges) > 2 and in_edges[-1] - in_edges[-2] < max(depth, 1):
            del out_edges[-2], in_edges[-2]
        if len(in_edges) == 2:
            depth = 0  # one input chunk: it holds everything it needs

        plan.append((axis, factor, width, depth, in_edges, out_edges))
        depths[axis] = depth
        reads.append(slice(first, last))
        in_chunks.append(tuple(np.diff(in_edges).tolist()))
        out_chunks.append(tuple(np.diff(out_edges).tolist()))

    data = source[tuple(reads)].rechunk(tuple(in_chunks))
    if not plan:
        return data
    if any(depths.values()):
        data = overlap(
            data,
            depth={axis: depths.get(axis, 0) for axis in range(source.ndim)},
            boundary="none",
        )
    # Shrinking axes come first: the following passes then handle less data.
    plan.sort(key=lambda entry: -entry[1])
    return data.map_blocks(
        _resample_block,
        plan=plan,
        nearest=nearest,
        chunks=tuple(out_chunks),
        dtype=source.dtype,
        meta=source._meta,
    )


def resample_array(
    arr: da.Array,
    factors: Mapping[int, float],
    *,
    nearest: bool = False,
    chunks: Sequence[int] | None = None,
) -> da.Array:
    """Resample ``arr``, given ``{axis index: new pixel size / old pixel size}``.

    Linear interpolation, anti-aliased on the axes that shrink, or nearest
    neighbour with ``nearest=True``. Output chunks are ``chunks`` (default: those
    of ``arr``), except when a small chunk cannot hold the overlap on its own.
    """
    factors = _effective_factors(factors)
    if not factors:
        return arr
    region = tuple(slice(0, n) for n in _output_shape(arr.shape, factors))
    return resample_region(arr, factors, region, chunks or arr.chunksize, nearest=nearest)


def _normalize_target_scale(target_scale, axes, base_scale) -> tuple[float, ...]:
    """Return the target pixel size per spatial axis; a dict may omit axes to keep them."""
    if not isinstance(target_scale, Mapping):
        return tuple(spatial_values_for_axes(target_scale, axes, name="target_scale"))
    unknown = set(target_scale) - set("zyx")
    if unknown:
        raise ValueError(f"target_scale keys must be spatial axes 'z', 'y' or 'x', got {sorted(unknown)}.")
    old = dict(zip(spatial_axes_in_order(axes), base_scale))
    target = {**old, **{ax: value for ax, value in target_scale.items() if ax in old}}
    if any(value <= 0 for value in target.values()):
        raise ValueError(f"target_scale values must be > 0, got {target_scale!r}.")
    return tuple(float(target[ax]) for ax in old)


def _is_label(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("is_label")) or metadata.get("data_type") == "label"


def _source_level(scales, target, *, use_pyramid: bool) -> int:
    """The coarsest level at least as fine as ``target`` on every axis, or level 0.

    Pyramid levels are usually built by skipping pixels, so an image read from one
    keeps its aliasing: only labels, or a request for speed, should use them.
    """
    if not use_pyramid:
        return 0
    return max(
        (i for i, scale in enumerate(scales) if all(s <= t * (1 + 1e-6) for s, t in zip(scale, target))),
        default=0,
    )


def resample_levels(
    data: Sequence[da.Array],
    metadata: dict[str, Any],
    target_scale: float | Sequence[float] | Mapping[str, float],
    *,
    fast: bool = False,
) -> tuple[list[da.Array], dict[str, Any]]:
    """Lazily resample a dataset to ``target_scale`` and rebuild its pyramid.

    ``target_scale`` is the new pixel size in the units of the dataset: one value,
    one per spatial axis, or a dict such as ``{"z": 1.0}`` (omitted axes are kept).

    Labels use nearest neighbour and images linear interpolation. Images are read
    from level 0 unless ``fast``, which reads the coarsest level still fine enough:
    much less data, but it keeps the aliasing of that level. Labels always do this.
    The number of levels is preserved.
    """
    if not data:
        raise ValueError("No data loaded.")
    if not metadata.get("scales"):
        raise ValueError("metadata['scales'] is required to resample a dataset.")

    axes = normalize_axes(metadata.get("axes"), ndim=data[0].ndim)
    scales = metadata["scales"]
    target = _normalize_target_scale(target_scale, axes, scales[0])
    nearest = _is_label(metadata)

    level = _source_level(scales, target, use_pyramid=fast or nearest)
    factors = {i: t / s for i, t, s in zip(spatial_axis_indices(axes), target, scales[level])}

    resampled = resample_array(data[level], factors, nearest=nearest)
    return build_pyramid(
        [resampled],
        {**metadata, "scales": [target]},
        num_levels=len(data),
        downscale_factor=infer_downscale_factors(metadata),
    )


def _batches(
    shape: Sequence[int],
    block: Sequence[int],
    max_pixels: float,
) -> Iterator[tuple[slice, ...]]:
    """Split an output array into regions of at most ``max_pixels``, computed in turn.

    Regions stay as large as allowed, because only the blocks of one region run in
    parallel: axes are split in array order (usually ``t`` first) and only as far
    as needed. Regions are whole blocks, so two of them never share a shard.
    """
    sizes = [int(n) for n in shape]
    for axis, chunk in enumerate(block):
        room = max_pixels / (math.prod(sizes) // sizes[axis])
        if room < sizes[axis]:
            sizes[axis] = min(max(int(room // chunk), 1) * int(chunk), sizes[axis])
    for starts in itertools.product(*(range(0, n, size) for n, size in zip(shape, sizes))):
        yield tuple(slice(s, min(s + size, n)) for s, size, n in zip(starts, sizes, shape))


def _write_level(source, dst, factors, *, nearest, cfg, desc) -> None:
    """Resample ``source`` into the array ``dst``, one batch at a time."""
    from .ngff import DEFAULT_MEMORY_FRACTION, _available_memory_mb, _run_stores

    block = dst.shards or dst.chunks
    block_pixels = math.prod(min(b, n) for b, n in zip(block, dst.shape))
    # What a pixel costs while it is being computed: its input, as floats.
    float_size = np.dtype(np.result_type(source.dtype, np.float32)).itemsize
    pixel_bytes = float_size * max(1.0, math.prod(factors.values()))
    # Peak memory measured at about twice the size of a batch.
    max_pixels = _available_memory_mb() * 1024**2 * DEFAULT_MEMORY_FRACTION / (2 * pixel_bytes)
    max_pixels = min(max_pixels, MAX_BLOCKS_PER_BATCH * block_pixels)

    regions = list(_batches(dst.shape, block, max_pixels))
    block_bytes = int(block_pixels * pixel_bytes)
    if cfg.progress:
        from tqdm import tqdm

        regions = tqdm(regions, desc=desc, unit="batch")
    quiet = replace(cfg, progress=False)
    for region in regions:
        data = resample_region(source, factors, region, block, nearest=nearest)
        store = da.store(data, dst, regions=region, lock=False, compute=False)
        _run_stores([store], quiet, block_bytes=block_bytes)


def _fitted_array_kwargs(src: zarr.Array, shape: Sequence[int]) -> dict[str, Any]:
    """``create_array`` keywords of ``src``, with chunks and shards fitted to ``shape``.

    Zarr stores every chunk in full, so a chunk reaching past the end of the array
    stores padding, which takes real space when uncompressed. Chunks therefore divide
    each axis evenly, and neither they nor the shards outgrow the input's or the array.
    """
    from .chromatic_shift import _mirrored_array_kwargs

    kwargs = _mirrored_array_kwargs(src, shape)
    counts = [ceil(n / chunk) for n, chunk in zip(shape, src.chunks)]
    chunks = tuple(ceil(n / count) for n, count in zip(shape, counts))
    kwargs["chunks"] = chunks
    if src.shards:
        per_shard = (shard // chunk for shard, chunk in zip(src.shards, src.chunks))
        kwargs["shards"] = tuple(c * min(m, count) for c, m, count in zip(chunks, per_shard, counts))
    return kwargs


def _set_level_scales(group: zarr.Group, scales, axes) -> None:
    """Write the pixel size of every level in the NGFF metadata copied from the input."""
    attrs = group.attrs.asdict()
    ome = attrs["ome"] if isinstance(attrs.get("ome"), dict) else attrs
    spatial = spatial_axis_indices(axes)
    for dataset, scale in zip(ome["multiscales"][0]["datasets"], scales):
        for transform in dataset.get("coordinateTransformations", []):
            if transform.get("type") == "scale":
                for axis, value in zip(spatial, scale):
                    transform["scale"][axis] = value
    group.attrs.update(attrs)


def write_resampled(
    source_root: zarr.Group,
    datasets: Sequence[tuple[str, Any]],
    path: str | Path,
    target_scale: float | Sequence[float] | Mapping[str, float],
    *,
    fast: bool = False,
    overwrite: bool = False,
    num_workers: int | None = None,
    progress: bool = True,
) -> zarr.Group:
    """Write a copy of an OME-Zarr store with a new pixel size.

    ``datasets`` are ``(name, dataset)`` pairs as yielded by
    ``ZarrManager._iter_datasets``. Every array copies the codecs and attributes of
    its source, with chunks and shards fitted to its new shape (see
    :func:`_fitted_array_kwargs`), and every group its attributes. Level 0 is
    resampled first (see :func:`resample_levels` for ``target_scale`` and ``fast``)
    and every other level is built from the level below it, read back from the
    output, keeping the downscale factors of the input.
    """
    from .ngff import ZarrWriteConfig
    from .output_path import check_output_path

    # Everything that can be refused is checked before the output is touched.
    datasets = [(name, dataset) for name, dataset in datasets if dataset.data]
    if not datasets:
        raise ValueError("No datasets to write.")
    targets = {}
    for name, dataset in datasets:
        if dataset.zarr_data is None or len(dataset.zarr_data) != len(dataset.data):
            raise ValueError(
                f"Dataset {name!r} has in-memory changes that are not on disk, so its "
                "on-disk structure is unknown. Write it with to_zarr() first."
            )
        axes = normalize_axes(dataset.metadata.get("axes"), ndim=dataset.data[0].ndim)
        targets[name] = _normalize_target_scale(target_scale, axes, dataset.metadata["scales"][0])
    check_output_path(path, overwrite=overwrite, sources=[getattr(source_root.store, "root", None)])

    cfg = ZarrWriteConfig(num_workers=num_workers, progress=progress)
    root = zarr.open_group(
        str(Path(path)),
        mode="w" if overwrite else "w-",
        zarr_format=source_root.metadata.zarr_format,
    )
    root.attrs.update(source_root.attrs.asdict())
    if any(name.startswith("labels/") for name, _ in datasets) and "labels" in source_root:
        root.require_group("labels").attrs.update(source_root["labels"].attrs.asdict())

    for name, dataset in datasets:
        axes = normalize_axes(dataset.metadata.get("axes"), ndim=dataset.data[0].ndim)
        spatial = spatial_axis_indices(axes)
        source_scales = dataset.metadata["scales"]
        target = targets[name]
        nearest = _is_label(dataset.metadata)

        group_path = (dataset.path or "/").strip("/")
        group = root if not group_path else root.require_group(group_path)
        if group_path:
            group.attrs.update(source_root[group_path].attrs.asdict())

        level = _source_level(source_scales, target, use_pyramid=fast or nearest)
        source = dataset.data[level]
        factors = _effective_factors(
            {i: t / s for i, t, s in zip(spatial, target, source_scales[level])}
        )
        scales = [target]
        for k, src in enumerate(dataset.zarr_data):
            dst = group.create_array(**_fitted_array_kwargs(src, _output_shape(source.shape, factors)))
            dst.attrs.update(src.attrs.asdict())
            _write_level(
                source, dst, factors,
                nearest=nearest or k > 0,  # as build_pyramid: coarser levels skip pixels
                cfg=cfg,
                desc=f"{name} level {k}",
            )
            if k + 1 < len(dataset.zarr_data):
                steps = [
                    max(1, round(coarse / fine))
                    for coarse, fine in zip(source_scales[k + 1], source_scales[k])
                ]
                source = da.from_zarr(dst)
                factors = _effective_factors(dict(zip(spatial, steps)))
                scales.append(tuple(s * step for s, step in zip(scales[-1], steps)))
        _set_level_scales(group, scales, axes)
    return root
