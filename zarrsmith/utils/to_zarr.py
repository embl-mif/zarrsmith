from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import dask.array as da
import numpy as np
import zarr

from .axes import normalize_axes, normalize_data_type, spatial_axes_in_order
from .display_window import update_display_windows
from .output_path import check_output_path
from .translation import drop_axes, normalize_translations
from .ngff import (
    ZarrWriteConfig,
    _resolve_format,
    _validate_metadata,
    _write_ngff_metadata,
    _write_pyramid_v2,
    _write_pyramid_v3,
)

logger = logging.getLogger("zarrsmith")

_DROPPABLE_AXES = ("t", "c", "z")


def _drop_axes_from_config_shapes(value, keep: list[int], ndim: int):
    """Drop entries of a chunks/shards spec that refer to removed axes.

    Handles a single shape or one shape per level; anything that does not look
    like a full-``ndim`` shape (``None``, ``"auto"``) is returned unchanged.
    """
    if value is None or isinstance(value, str):
        return value
    seq = list(value)
    if seq and all(isinstance(v, (int, np.integer)) for v in seq):
        return tuple(seq[i] for i in keep) if len(seq) == ndim else value
    return [
        tuple(s[i] for i in keep) if len(s) == ndim else s
        for s in seq
    ]


def _drop_singleton_axes(
    data_levels: Sequence[da.Array],
    metadata: dict,
    config: ZarrWriteConfig,
) -> tuple[list[da.Array], dict, ZarrWriteConfig]:
    """Remove singleton ``t``/``c``/``z`` axes from the data and metadata.

    ``y`` and ``x`` are never dropped. Returns the inputs unchanged when
    ``config.drop_singleton`` is false or no axis qualifies.
    """
    data_levels = list(data_levels)
    if not config.drop_singleton or not data_levels:
        return data_levels, metadata, config

    ndim = data_levels[0].ndim
    axes = normalize_axes(metadata.get("axes"), ndim=ndim)
    drop = [
        i for i, ax in enumerate(axes)
        if ax in _DROPPABLE_AXES and all(arr.shape[i] == 1 for arr in data_levels)
    ]
    if not drop:
        return data_levels, metadata, config

    keep = [i for i in range(ndim) if i not in drop]
    dropped_axes = {axes[i] for i in drop}
    new_axes = "".join(axes[i] for i in keep)

    new_levels = [arr.reshape(tuple(arr.shape[i] for i in keep)) for arr in data_levels]

    out = dict(metadata)
    out["axes"] = new_axes
    for key in ("size", "chunksize"):
        if out.get(key) is not None:
            out[key] = [
                tuple(v[i] for i in keep) if len(v) == ndim else tuple(v)
                for v in out[key]
            ]
    # scales/units only cover spatial axes, so drop the matching positions.
    spatial = spatial_axes_in_order(axes)
    keep_spatial = [i for i, ax in enumerate(spatial) if ax not in dropped_axes]
    if len(keep_spatial) != len(spatial):
        if out.get("scales") is not None:
            out["scales"] = [tuple(s[i] for i in keep_spatial) for s in out["scales"]]
        if out.get("units"):
            out["units"] = tuple(out["units"][i] for i in keep_spatial)
        out["translations"] = drop_axes(normalize_translations(out.get("translations"), axes), dropped_axes)
    if "t" in dropped_axes:
        out.pop("time_increment", None)
        out.pop("time_increment_unit", None)

    new_config = replace(
        config,
        chunks=_drop_axes_from_config_shapes(config.chunks, keep, ndim),
        shards=_drop_axes_from_config_shapes(config.shards, keep, ndim),
        shard_exclude_axes=tuple(
            ax for ax in (config.shard_exclude_axes or ()) if ax not in dropped_axes
        ),
    )
    return new_levels, out, new_config


def _metadata_for_write(
    metadata: dict,
    axes: tuple[str, ...],
    *,
    config: ZarrWriteConfig,
    is_label: bool | None = None,
) -> dict:
    out = dict(metadata)
    data_type = normalize_data_type(config.data_type or metadata.get("data_type"), is_label=is_label)
    out["axes"] = "".join(axes)
    out["data_type"] = data_type
    return out

def _set_display_windows(group: zarr.Group, metadata: dict, axes, cfg: ZarrWriteConfig) -> None:
    """The display window of every channel: ``metadata["channel_windows"]`` where given, else
    estimated from the written data (``cfg.display_window``), so viewers do not open the image
    over the whole range of its data type. Intensity data only; with ``compute=False`` nothing
    is written yet, so only the given windows are stored."""
    if metadata.get("data_type") != "intensity":
        return
    given = metadata.get("channel_windows")
    if not cfg.compute and not given:
        return
    windows = update_display_windows(group, axes, given, from_data=cfg.compute and cfg.display_window)
    if any(w is not None for w in windows):
        logger.info("Display window per channel: %s", windows)


def _prepare_write(data_levels, metadata, cfg, *, is_label=None):
    """The levels, axes, metadata and configuration to write: singleton axes dropped (if asked),
    the metadata completed for the output and checked against the levels."""
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")
    data_levels, metadata, cfg = _drop_singleton_axes(data_levels, metadata, cfg)
    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    effective_metadata = _metadata_for_write(metadata, axes, config=cfg, is_label=is_label)
    _validate_metadata(data_levels, effective_metadata, axes)
    return data_levels, axes, effective_metadata, cfg


def _write_levels(group, data_levels, metadata, axes, cfg, *, zarr_format, ngff_version, desc, name,
                  label_source=None):
    """Write the pyramid levels into ``group`` (as zarr v2 or v3), then its NGFF metadata and the
    channels' display windows. Returns the delayed writes (computed already unless ``cfg.compute``
    is off)."""
    if zarr_format == 3:
        delayed = _write_pyramid_v3(root=group, data_levels=data_levels, cfg=cfg, axes=axes, desc=desc)
    else:
        delayed = _write_pyramid_v2(root=group, data_levels=data_levels, cfg=cfg, desc=desc)
    _write_ngff_metadata(group, metadata, axes, data_levels[0], ngff_version=ngff_version, name=name,
                         label_source=label_source)
    _set_display_windows(group, metadata, axes, cfg)
    return delayed


def to_zarr(
    path: str | Path,
    data_levels: Sequence[da.Array],
    metadata: dict,
    *,
    config: ZarrWriteConfig | None = None,
    sources: Iterable[Any] = (),
):
    """Write a pyramid of dask arrays to an OME-Zarr root group.

    Parameters
    ----------
    path : str | Path
        Destination zarr store.
    data_levels : sequence of dask.array.Array
        Pyramid levels ordered from finest to coarsest resolution.
    metadata : dict
        Normalized ZarrSmith metadata dictionary describing axes, scales, channel
        metadata and units.
    config : ZarrWriteConfig | None
        Output configuration controlling NGFF version, zarr format, overwrite
        behaviour and compression.
    sources : iterable of paths
        Paths ``data_levels`` is read from: the output must not replace them.
    """
    data_levels, axes, effective_metadata, cfg = _prepare_write(data_levels, metadata, config or ZarrWriteConfig())

    ngff_version, zarr_format = _resolve_format(cfg)
    check_output_path(path, overwrite=cfg.overwrite, sources=sources)
    root = zarr.open_group(
        str(Path(path)),
        mode="w" if cfg.overwrite else "w-",
        zarr_format=zarr_format,
    )
    # A standalone label store has no source image, hence no ``label_source``.
    delayed = _write_levels(root, data_levels, effective_metadata, axes, cfg, zarr_format=zarr_format,
                            ngff_version=ngff_version, desc=Path(path).name,
                            name=effective_metadata.get("name") or "dataset")
    return root if cfg.compute else delayed


def write_multiscale_to_group(
    group: zarr.Group,
    data_levels: Sequence[da.Array],
    metadata: dict,
    *,
    config: ZarrWriteConfig | None = None,
    name: str | None = None,
    is_label: bool = False,
):
    """Write a pyramid of Dask arrays into an existing zarr group.

    Used by :class:`zarrsmith.microscope_manager.ZarrManager` for raw data, image
    subgroups and label groups.  The axes may be any subset of ``tczyx``.
    """
    data_levels, axes, effective_metadata, cfg = _prepare_write(
        data_levels, metadata, config or ZarrWriteConfig(), is_label=True if is_label else None
    )

    ngff_version, zarr_format = _resolve_format(cfg)

    if cfg.overwrite:
        for key in list(group.array_keys()):
            if str(key).isdigit():
                del group[key]

    # Only labels nested in ``<image>/labels/<name>`` have a source image.
    delayed = _write_levels(group, data_levels, effective_metadata, axes, cfg, zarr_format=zarr_format,
                            ngff_version=ngff_version, desc=name or group.name.strip('/') or 'raw',
                            name=name or effective_metadata.get("name") or "dataset",
                            label_source={"source": {"image": "../../"}} if is_label else None)
    return group if cfg.compute else delayed
