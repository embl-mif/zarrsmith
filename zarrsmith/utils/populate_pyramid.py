"""Build the pyramid of an OME-Zarr image in place, from level 0 on disk.

Used by :meth:`zarrsmith.microscope_manager.ZarrManager.populate_pyramid`
and ``zarrsmith pyramid``. Level 0 is never rewritten: each coarser level is
read back from the level above it, one output block (a shard on zarr v3, a
chunk otherwise) per task, so no two tasks write the same file and memory
stays bounded by the block size.
"""
from __future__ import annotations

import copy
import itertools
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, Sequence

import numpy as np
import zarr

from .axes import normalize_axes, spatial_axis_indices
from .chunks import default_num_levels
from .downsampling import (
    SpatialFactor,
    axis_names_from_multiscales,
    DOWNSAMPLING_METHODS,
    dataset_transform_vector,
    downsample_numpy,
    downsampling_record,
    normalize_spatial_factor_for_axes,
    resolve_downsampling_method,
)
from .translation import pyramid_translations
from .ngff import (
    ZarrWriteConfig,
    _get_group_multiscales,
    _infer_data_type_from_group,
    _infer_ngff_version,
    _resolve_num_workers,
    _set_multiscales,
)


def _blocks(region: Sequence[slice], block: Sequence[int]) -> Iterator[tuple[slice, ...]]:
    """Every block of the grid ``block`` inside ``region`` (edge blocks clipped).

    The grid starts at 0, so blocks of a region that starts on the grid are
    whole blocks of the array.
    """
    ranges = [range(r.start, r.stop, int(b)) for r, b in zip(region, block)]
    for starts in itertools.product(*ranges):
        yield tuple(slice(s, min(s + int(b), r.stop)) for s, b, r in zip(starts, block, region))


def _downsample_region(src, region, per_axis, spatial_axes, factors, method) -> np.ndarray:
    """``region`` of the next level, computed from the part of ``src`` it covers."""
    source_region = tuple(
        slice(r.start * f, min(r.stop * f, n)) for r, f, n in zip(region, per_axis, src.shape)
    )
    return downsample_numpy(np.asarray(src[source_region]), factors, spatial_axes, method)


def _write_block(src: zarr.Array, dst: zarr.Array, region: tuple[slice, ...], per_axis, spatial_axes, factors, method):
    """Compute ``region`` (one shard, or one chunk) of ``dst`` and write it in one go.

    A shard is assembled one inner chunk at a time, so a task holds the output
    shard plus the source window of a single chunk, not the source of the whole
    shard (``f**3`` times larger in 3D).
    """
    if dst.shards is None:
        dst[region] = _downsample_region(src, region, per_axis, spatial_axes, factors, method)
        return
    out = np.empty(tuple(r.stop - r.start for r in region), dtype=dst.dtype)
    for piece in _blocks(region, dst.chunks):
        local = tuple(slice(p.start - r.start, p.stop - r.start) for p, r in zip(piece, region))
        out[local] = _downsample_region(src, piece, per_axis, spatial_axes, factors, method)
    dst[region] = out


def _task_bytes(dst: zarr.Array, src_dtype, per_axis, method) -> int:
    """Peak memory of one task: its output block, plus one chunk's source window
    and the float copy that ``"mean"`` makes of it."""
    from .downsampling import mean_accumulator_dtype

    block = tuple(dst.shards or dst.chunks)
    out_bytes = math.prod(min(b, n) for b, n in zip(block, dst.shape)) * np.dtype(dst.dtype).itemsize
    window = math.prod(min(c, n) for c, n in zip(dst.chunks, dst.shape)) * math.prod(per_axis)
    window_bytes = window * np.dtype(src_dtype).itemsize
    if method == "mean":
        window_bytes += window * np.dtype(mean_accumulator_dtype(src_dtype)).itemsize
    return out_bytes + window_bytes


def _write_level(src: zarr.Array, dst: zarr.Array, factors, spatial_axes, method, *, cfg: ZarrWriteConfig, desc: str):
    """Fill ``dst`` from ``src``, one task per shard (or chunk) of ``dst``."""
    per_axis = [1] * src.ndim
    for axis, f in zip(spatial_axes, factors):
        per_axis[axis] = int(f)

    block = tuple(dst.shards or dst.chunks)
    regions = list(_blocks(tuple(slice(0, int(n)) for n in dst.shape), block))
    # Limit the threads so the tasks in flight fit in memory, sharded or not.
    workers = _resolve_num_workers(cfg, block_bytes=_task_bytes(dst, src.dtype, per_axis, method))

    bar = None
    if cfg.progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(regions), desc=desc, unit="block")
        except ImportError:  # tqdm is optional at runtime
            bar = None
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_write_block, src, dst, region, per_axis, spatial_axes, factors, method)
                for region in regions
            ]
            for future in futures:
                future.result()
                if bar is not None:
                    bar.update(1)
    finally:
        if bar is not None:
            bar.close()


def _level_array_kwargs(level0: zarr.Array, name: str, shape: Sequence[int]) -> dict[str, Any]:
    """``create_array`` keywords of a new level: level 0's format, codecs and chunk
    key encoding, with chunks (and shards) fitted to the level's shape."""
    from .resample import _fitted_array_kwargs

    kwargs = _fitted_array_kwargs(level0, shape)
    kwargs["name"] = name
    return kwargs


def resolve_group(root: zarr.Group, group_name: str | None) -> zarr.Group:
    """``root`` itself, or a (possibly nested, e.g. ``labels/nuclei``) group of it."""
    if group_name in (None, "", "/"):
        return root
    group = root
    for part in [p for p in str(group_name).split("/") if p]:
        if part not in group or not isinstance(group[part], zarr.Group):
            raise KeyError(f"Group {group_name!r} not found in the store.")
        group = group[part]
    return group


def _resolve_start_level(datasets: Sequence[dict], start_level: int | str) -> int:
    """Index in ``datasets`` of ``start_level``: an index, or a dataset path such as ``"1"``."""
    paths = [str(d["path"]) for d in datasets]
    if isinstance(start_level, str) and not start_level.lstrip("-").isdigit():
        if start_level not in paths:
            raise ValueError(f"start_level {start_level!r} is not a level of this image (levels: {paths}).")
        return paths.index(start_level)
    index = int(start_level)
    if isinstance(start_level, str) and start_level in paths:
        index = paths.index(start_level)  # a numeric path names the level stored there
    if not 0 <= index < len(paths):
        raise ValueError(f"start_level {start_level!r} is out of range: the image has {len(paths)} level(s).")
    return index


def level_factors(shape_a: Sequence[int], shape_b: Sequence[int], spatial_axes: Sequence[int]) -> tuple[int, ...]:
    """Integer factor per spatial axis between a level of shape ``shape_a`` and the
    next, coarser one of shape ``shape_b`` (sizes rounded up or down both work)."""
    return tuple(max(1, round(int(shape_a[i]) / max(1, int(shape_b[i])))) for i in spatial_axes)


def _existing_level_factors(group: zarr.Group, datasets, start: int, spatial_axes) -> tuple[int, ...] | None:
    """Factors between the start level and the existing level after it, if there is one."""
    if len(datasets) <= start + 1:
        return None
    try:
        a = group[str(datasets[start]["path"])].shape
        b = group[str(datasets[start + 1]["path"])].shape
    except KeyError:
        return None
    factors = level_factors(a, b, spatial_axes)
    return factors if any(f > 1 for f in factors) else None


def populate_pyramid(
    root: zarr.Group,
    group_name: str | None = None,
    *,
    num_levels: int | None = None,
    downscale_factor: SpatialFactor | None = None,
    method: str | None = None,
    start_level: int | str = 0,
    num_workers: int | None = None,
    progress: bool = True,
) -> zarr.Group:
    """Create or overwrite the levels after ``start_level`` of an image, from that level.

    Parameters
    ----------
    root
        The store's root group, opened for writing.
    group_name
        Image to process: ``None`` for the root image, else a group path such as
        ``"segmentation"`` or ``"labels/nuclei"``.
    num_levels
        Total number of levels, level 0 included. ``None`` keeps the number of
        levels the image already has (e.g. as laid out by ``allocate`` from a
        template); with no level after the start level, it keeps downscaling until
        no spatial axis is above 2048 pixels (see
        :func:`zarrsmith.utils.chunks.default_num_levels`). ``start_level + 1``
        removes every level after the start level.
    start_level
        Level the new levels are built from: its index in ``multiscales.datasets``
        or its path (e.g. ``"1"``). It and the levels above it are kept as they are
        (data and metadata); every level after it is removed and rebuilt. ``0``
        (default) rebuilds the whole pyramid from full resolution.
    downscale_factor
        Integer factor between consecutive levels: one value, or one per spatial
        axis (e.g. ``(1, 2, 2)`` to keep Z). ``None`` takes it from the shapes of the
        start level and the existing level after it, else 2.
    method
        ``"mean"``, ``"nearest"`` or ``"mode"``. ``None`` picks ``"nearest"`` for
        label images and ``"mean"`` otherwise; ``"mean"`` is refused for labels.
    num_workers, progress
        As in :class:`~zarrsmith.utils.ngff.ZarrWriteConfig`.

    The zarr format, NGFF version, codecs and chunk key encoding come from level 0,
    so a zarr v2 store gets v2 levels (and no shards). Each new level's ``scale`` is
    the start level's times the factor per step; a ``translation`` follows the method (see
    :func:`zarrsmith.utils.translation.pyramid_translations`, the NGFF 0.6 convention). Every other key
    of the ``multiscales`` block, and every other group attribute, is kept.
    """
    group = resolve_group(root, group_name)
    label = group_name or "/"
    multiscales_all = copy.deepcopy(_get_group_multiscales(group) or [])
    if not multiscales_all or not multiscales_all[0].get("datasets"):
        raise ValueError(f"Group {label!r} has no multiscales metadata.")
    multiscales = multiscales_all[0]
    datasets = multiscales["datasets"]

    ngff_version = _infer_ngff_version(group)
    zarr_format = group.metadata.zarr_format
    if (str(ngff_version), zarr_format) not in {("0.4", 2), ("0.5", 3)}:
        raise ValueError(
            f"Group {label!r} is NGFF {ngff_version} on zarr v{zarr_format}. zarrsmith writes "
            "pyramids only into NGFF 0.4 (zarr v2) or 0.5 (zarr v3) images, so the store keeps "
            "one version throughout."
        )

    start = _resolve_start_level(datasets, start_level)
    kept = datasets[: start + 1]
    kept_paths = [str(d["path"]) for d in kept]
    level0 = group[kept_paths[0]]
    source = group[kept_paths[-1]]
    axes = normalize_axes(axis_names_from_multiscales(multiscales) or None, ndim=level0.ndim)
    spatial_axes = spatial_axis_indices(axes)
    method = resolve_downsampling_method(method, _infer_data_type_from_group(group))
    # Levels already laid out (e.g. by allocate from a template) set the defaults.
    existing = _existing_level_factors(group, datasets, start, spatial_axes)
    if downscale_factor is None:
        downscale_factor = existing or 2
    factors = tuple(
        int(f) for f in normalize_spatial_factor_for_axes(downscale_factor, axes, allow_float=False)
    )
    if any(f < 1 for f in factors):
        raise ValueError(f"downscale_factor must be >= 1, got {downscale_factor!r}.")
    if num_levels is None:
        if existing is not None:
            num_levels = len(datasets)
        else:
            num_levels = default_num_levels({"axes": "".join(axes), "size": [level0.shape]}, factors)
        num_levels = max(int(num_levels), start + 1)
    num_levels = int(num_levels)
    if num_levels < start + 1:
        raise ValueError(
            f"num_levels ({num_levels}) counts every level from level 0, so it must be at least "
            f"start_level + 1 ({start + 1})."
        )
    if num_levels > start + 1 and all(f == 1 for f in factors):
        raise ValueError("downscale_factor is 1 on every spatial axis: the levels would all be identical.")

    names = kept_paths + [str(k) for k in range(start + 1, num_levels)]
    clash = set(kept_paths) & set(names[start + 1:])
    if clash:
        raise ValueError(f"Kept level(s) stored at {sorted(clash)}, which a new level would replace.")

    # Remove the old levels from the metadata first, so an interrupted run never
    # leaves the metadata pointing at a half-written level.
    multiscales["datasets"] = list(kept)
    _set_multiscales(group, multiscales_all, ngff_version)
    stale = {str(d["path"]) for d in datasets[start + 1:]} | {
        str(k) for k in group.array_keys() if str(k).isdigit()
    }
    for key in sorted(stale - set(kept_paths)):
        if key in group:
            del group[key]

    scale_s = dataset_transform_vector(kept[-1], "scale") or [1.0] * len(axes)
    translation_s = dataset_transform_vector(kept[-1], "translation")
    # NGFF 0.6 convention (zarrsmith.utils.translation.pyramid_translations): "mean" and
    # "mode" levels move by half their pixel growth; "nearest" keeps the start level's
    # origin. Without a translation, "nearest" levels need none and the others get one.
    spatial_names = [axes[i] for i in spatial_axes]
    n_new = num_levels - start  # the start level plus the levels built from it
    level_scales = [[scale_s[i] * f ** k for i, f in zip(spatial_axes, factors)] for k in range(n_new)]
    spatial_translations = pyramid_translations(
        None if translation_s is None else {ax: translation_s[i] for ax, i in zip(spatial_names, spatial_axes)},
        level_scales[0], level_scales, "".join(axes), method,
    )

    cfg = ZarrWriteConfig(num_workers=num_workers, progress=progress)
    src = source
    new_datasets = list(kept)
    for k in range(start + 1, num_levels):
        j = k - start  # steps below the start level
        shape = list(src.shape)
        for axis, f in zip(spatial_axes, factors):
            shape[axis] = -(-int(shape[axis]) // f)
        dst = group.create_array(**_level_array_kwargs(level0, names[k], shape))
        dst.attrs.update(level0.attrs.asdict())
        _write_level(src, dst, factors, spatial_axes, method, cfg=cfg,
                     desc=f"{label.strip('/') or 'image'} level {k}")

        scale = list(scale_s)
        for i, value in zip(spatial_axes, level_scales[j]):
            scale[i] = value
        transforms: list[dict[str, Any]] = [{"type": "scale", "scale": scale}]
        if spatial_translations is not None:
            translation = list(translation_s or [0.0] * len(axes))
            for ax, i in zip(spatial_names, spatial_axes):
                translation[i] = spatial_translations[j][ax]
            transforms.append({"type": "translation", "translation": translation})
        new_datasets.append({"path": names[k], "coordinateTransformations": transforms})
        src = dst

    multiscales["datasets"] = new_datasets
    if num_levels > start + 1:
        from .axes import DATA_TYPES

        # NGFF 0.4-0.6: "type" SHOULD name the downscaling method, "metadata" SHOULD describe it.
        record = downsampling_record(method, factors)
        if start:  # the levels up to the start level were kept as they were
            record["metadata"]["kwargs"]["start_level"] = kept_paths[-1]
            record["metadata"]["description"] = record["metadata"]["description"].replace(
                "Each level", f"Each level after {kept_paths[-1]!r}", 1)
        # Older stores used "type" to mark labels, and readers still honour it: keep such a marker
        # (the method is then read from "metadata").
        if multiscales.get("type") not in set(DATA_TYPES) | {"image", "labels"}:
            multiscales["type"] = record["type"]
        multiscales["metadata"] = record["metadata"]
    elif start == 0:  # no level built from level 0: no method to record
        if multiscales.get("type") in DOWNSAMPLING_METHODS:
            multiscales.pop("type")
        multiscales.pop("metadata", None)
    _set_multiscales(group, multiscales_all, ngff_version)
    return group
