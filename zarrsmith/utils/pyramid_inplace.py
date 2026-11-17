"""Build the pyramid of an OME-Zarr image in place, from level 0 on disk.

Used by :meth:`zarrsmith.microscope_manager.ZarrManager.build_pyramid_inplace`
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
    downsampled_translation,
    normalize_spatial_factor_for_axes,
    resolve_downsampling_method,
)
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


def build_pyramid_inplace(
    root: zarr.Group,
    group_name: str | None = None,
    *,
    num_levels: int | None = None,
    downscale_factor: SpatialFactor = 2,
    method: str | None = None,
    num_workers: int | None = None,
    progress: bool = True,
) -> zarr.Group:
    """Create or overwrite levels ``1..num_levels-1`` of an image from its level 0.

    Parameters
    ----------
    root
        The store's root group, opened for writing.
    group_name
        Image to process: ``None`` for the root image, else a group path such as
        ``"segmentation"`` or ``"labels/nuclei"``.
    num_levels
        Total number of levels, level 0 included. ``None`` keeps halving until no
        spatial axis is above 2048 pixels (see
        :func:`zarrsmith.utils.chunks.default_num_levels`). ``1`` removes every
        coarser level.
    downscale_factor
        Integer factor between consecutive levels: one value, or one per spatial
        axis (e.g. ``(1, 2, 2)`` to keep Z).
    method
        ``"mean"``, ``"nearest"`` or ``"mode"``. ``None`` picks ``"nearest"`` for
        label images and ``"mean"`` otherwise; ``"mean"`` is refused for labels.
    num_workers, progress
        As in :class:`~zarrsmith.utils.ngff.ZarrWriteConfig`.

    The zarr format, NGFF version, codecs and chunk key encoding come from level 0,
    so a zarr v2 store gets v2 levels (and no shards). Each level's ``scale`` is
    level 0's times the factor; a ``translation`` follows the method (see
    :func:`zarrsmith.utils.downsampling.downsampled_translation`). Every other key
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

    level0_path = str(datasets[0]["path"])
    level0 = group[level0_path]
    axes = normalize_axes(axis_names_from_multiscales(multiscales) or None, ndim=level0.ndim)
    spatial_axes = spatial_axis_indices(axes)
    method = resolve_downsampling_method(method, _infer_data_type_from_group(group))
    factors = tuple(
        int(f) for f in normalize_spatial_factor_for_axes(downscale_factor, axes, allow_float=False)
    )
    if any(f < 1 for f in factors):
        raise ValueError(f"downscale_factor must be >= 1, got {downscale_factor!r}.")
    if num_levels is None:
        num_levels = default_num_levels({"axes": "".join(axes), "size": [level0.shape]}, factors)
    num_levels = int(num_levels)
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1.")
    if num_levels > 1 and all(f == 1 for f in factors):
        raise ValueError("downscale_factor is 1 on every spatial axis: the levels would all be identical.")

    names = [level0_path] + [str(k) for k in range(1, num_levels)]
    if level0_path in names[1:]:
        raise ValueError(f"Level 0 is stored at {level0_path!r}, which a new level would replace.")

    # Remove the old levels from the metadata first, so an interrupted run never
    # leaves the metadata pointing at a half-written level.
    multiscales["datasets"] = [datasets[0]]
    _set_multiscales(group, multiscales_all, ngff_version)
    stale = {str(d["path"]) for d in datasets[1:]} | {
        str(k) for k in group.array_keys() if str(k).isdigit()
    }
    for key in sorted(stale - {level0_path}):
        if key in group:
            del group[key]

    scale0 = dataset_transform_vector(datasets[0], "scale") or [1.0] * len(axes)
    translation0 = dataset_transform_vector(datasets[0], "translation")
    # Without a translation, "nearest" levels need none; "mean"/"mode" levels move
    # by half a block, so they get one (level 0 stays at the origin).
    track_translation = translation0 is not None or method != "nearest"
    translation = translation0 or [0.0] * len(axes)
    scale = list(scale0)

    cfg = ZarrWriteConfig(num_workers=num_workers, progress=progress)
    src = level0
    new_datasets = [datasets[0]]
    for k in range(1, num_levels):
        shape = list(src.shape)
        for axis, f in zip(spatial_axes, factors):
            shape[axis] = -(-int(shape[axis]) // f)
        dst = group.create_array(**_level_array_kwargs(level0, names[k], shape))
        dst.attrs.update(level0.attrs.asdict())
        _write_level(src, dst, factors, spatial_axes, method, cfg=cfg,
                     desc=f"{label.strip('/') or 'image'} level {k}")

        spatial_translation = downsampled_translation(
            [translation[i] for i in spatial_axes], [scale[i] for i in spatial_axes], factors, method
        )
        for i, value in zip(spatial_axes, spatial_translation):
            translation[i] = value
        for i, f in zip(spatial_axes, factors):
            scale[i] *= f
        transforms: list[dict[str, Any]] = [{"type": "scale", "scale": list(scale)}]
        if track_translation:
            transforms.append({"type": "translation", "translation": list(translation)})
        new_datasets.append({"path": names[k], "coordinateTransformations": transforms})
        src = dst

    multiscales["datasets"] = new_datasets
    if num_levels > 1:
        # NGFF 0.4-0.6: "type" SHOULD name the downscaling method, "metadata" SHOULD describe it.
        from .. import __version__

        from .axes import DATA_TYPES

        # Older stores used "type" to mark labels, and readers still honour it: keep such a marker.
        if multiscales.get("type") not in set(DATA_TYPES) | {"image", "labels"}:
            multiscales["type"] = method
        multiscales["metadata"] = {
            "description": f"Each level is the {method} of {'x'.join(map(str, factors))} blocks "
                           "of the level above it.",
            "method": "zarrsmith.utils.downsampling.downsample_numpy",
            "version": __version__,
            "kwargs": {"method": method, "factors": list(factors)},
        }
    else:
        if multiscales.get("type") in DOWNSAMPLING_METHODS:
            multiscales.pop("type")
        multiscales.pop("metadata", None)
    _set_multiscales(group, multiscales_all, ngff_version)
    return group
