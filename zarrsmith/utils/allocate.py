"""Create an empty OME-Zarr image ready to be filled region by region.

Backs ``zarrsmith allocate``: the geometry comes either from a template
OME-Zarr image (optionally cropped, retyped or with other channels) or from an
explicit size, axes and pixel size. Only empty arrays and metadata are
written; tiles are then written with ``ZarrManager.write_*_region`` (ideally
with ``aligned=True``) and the pyramid with ``zarrsmith pyramid``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from .axes import normalize_axes, normalize_data_type, spatial_axes_in_order
from .chunks import default_chunk_size
from .colors import default_channel_color
from .downsampling import default_downsampling_method, normalize_spatial_factor_for_axes
from .translation import pyramid_translations
from .ngff import DEFAULT_COMPRESSOR, DEFAULT_COMPRESSOR_LEVEL

Crop = Mapping[str, tuple[int | None, int | None] | slice]


def _template_info(path: str | Path) -> dict[str, Any]:
    """Level-0 geometry and metadata of the root image of an OME-Zarr store."""
    from ..microscope_manager.zarr_manager import ZarrManager

    z = ZarrManager(str(path), mode="r")
    md = z.raw.metadata
    level0 = z.raw.zarr_data[0]
    return {
        "axes": md["axes"],
        "size": tuple(int(s) for s in md["size"][0]),
        "dtype": str(md["dtype"]),
        "scale": tuple(md["scales"][0]),
        "translation": None if not md.get("translations") else dict(md["translations"][0]),
        "units": tuple(md.get("units") or ()),
        "time_increment": md.get("time_increment"),
        "time_increment_unit": md.get("time_increment_unit"),
        "channel_names": list(md.get("channel_names") or []),
        "channel_colors": list(md.get("channel_colors") or []),
        "data_type": md.get("data_type"),
        "chunks": tuple(int(c) for c in level0.chunks),
        "shards": None if level0.shards is None else tuple(int(s) for s in level0.shards),
        "zarr_format": z.root.metadata.zarr_format,
        "name": (z._get_multiscales(z.root) or [{}])[0].get("name"),
    }


def _crop_range(spec: Any, n: int, axis: str) -> tuple[int, int]:
    if isinstance(spec, slice):
        if spec.step not in (None, 1):
            raise ValueError(f"crop along {axis!r} must be contiguous, got step {spec.step}.")
        start, stop = spec.start, spec.stop
    else:
        start, stop = spec
    start = 0 if start is None else int(start)
    stop = n if stop is None else int(stop)
    if start < 0:
        start += n
    if stop < 0:
        stop += n
    if not 0 <= start < stop <= n:
        raise ValueError(f"crop {start}:{stop} along {axis!r} is outside the axis (size {n}).")
    return start, stop


def _per_axis(values: Sequence[Any] | None, axes: str, name: str) -> tuple[Any, ...] | None:
    if values is None:
        return None
    values = tuple(values)
    if len(values) != len(axes):
        raise ValueError(f"{name} needs one value per axis ({axes}), got {len(values)}: {values}.")
    return values


def _spatial(values: Any, axes: str, name: str) -> tuple[float, ...]:
    """One value per spatial axis, from one value or one per spatial axis."""
    spatial = spatial_axes_in_order(axes)
    if isinstance(values, (int, float)):
        return tuple(float(values) for _ in spatial)
    values = tuple(float(v) for v in values)
    if len(values) == 1:
        return values * len(spatial)
    if len(values) != len(spatial):
        raise ValueError(
            f"{name} needs one value per spatial axis ({''.join(spatial)}), got {len(values)}: {values}."
        )
    return values


def build_allocation_metadata(
    *,
    template: str | Path | None = None,
    size: Sequence[int] | None = None,
    axes: str | None = None,
    scale: float | Sequence[float] | None = None,
    units: str | Sequence[str] | None = None,
    time_increment: float | None = None,
    time_increment_unit: str | None = None,
    dtype: str | None = None,
    data_type: str | None = None,
    channel_names: Sequence[str] | None = None,
    channel_colors: Sequence[str] | None = None,
    num_channels: int | None = None,
    crop: Crop | None = None,
    chunk_size: Sequence[int] | None = None,
    shards: Any = None,
    num_levels: int = 1,
    downscale_factor: int | Sequence[int] = 2,
    translation: bool | Mapping[str, float] | Sequence[float] | None = None,
    zarr_format: int | None = None,
    extra_attributes: Mapping[str, Any] | None = None,
    name: str | None = None,
) -> tuple[dict[str, Any], int, Any]:
    """The metadata, zarr format and shards :func:`allocate_zarr` creates.

    See :func:`allocate_zarr` for the parameters. Returns
    ``(metadata, zarr_format, shards)``, where ``shards`` is what to pass to
    :func:`zarrsmith.utils.create_empty_dataset.create_empty_dataset`.
    """
    if template is not None:
        if size is not None or axes is not None:
            raise ValueError("Give either a template or size/axes, not both.")
        info = _template_info(template)
    else:
        if size is None or axes is None:
            raise ValueError("Without a template, size and axes are required.")
        axes_t = normalize_axes(axes, ndim=len(size))
        info = {
            "axes": "".join(axes_t),
            "size": tuple(int(s) for s in size),
            "dtype": "uint16",
            "scale": _spatial(1.0 if scale is None else scale, "".join(axes_t), "scale"),
            "translation": None,
            "units": (),
            "time_increment": 1.0 if "t" in axes_t else None,
            "time_increment_unit": None,
            "channel_names": [],
            "channel_colors": [],
            "data_type": None,
            "chunks": None,
            "shards": None,
            "zarr_format": None,
            "name": None,
        }

    axes_out = info["axes"]
    sizes = dict(zip(axes_out, info["size"]))
    chunks = dict(zip(axes_out, info["chunks"])) if info["chunks"] else None
    template_shards = dict(zip(axes_out, info["shards"])) if info["shards"] else None
    spatial = list(spatial_axes_in_order(axes_out))
    scale_by_axis = dict(zip(spatial, info["scale"]))
    if template is not None and scale is not None:
        scale_by_axis = dict(zip(spatial, _spatial(scale, axes_out, "scale")))
    origin = {ax: float((info["translation"] or {}).get(ax, 0.0)) for ax in spatial}
    unit_values = info["units"]
    if units is not None:
        unit_values = (units,) * len(spatial) if isinstance(units, str) else tuple(units)
    if template is None and not unit_values:
        unit_values = ("micrometer",) * len(spatial)
    units_by_axis = dict(zip(spatial, unit_values)) if unit_values else {}

    out_type = normalize_data_type(data_type or info["data_type"])

    # Crop: the first kept pixel becomes the new origin.
    moved = False
    for ax, spec in (crop or {}).items():
        ax = str(ax).lower()
        if ax not in sizes:
            raise ValueError(f"Cannot crop axis {ax!r}: the image axes are {axes_out!r}.")
        start, stop = _crop_range(spec, sizes[ax], ax)
        sizes[ax] = stop - start
        if ax == "c":
            names, colors = info["channel_names"], info["channel_colors"]
            info["channel_names"] = names[start:stop] if names else []
            info["channel_colors"] = colors[start:stop] if colors else []
        if ax in origin and start:
            origin[ax] += start * scale_by_axis[ax]
            moved = True

    # Channels: a label image drops the template's channel axis unless channels are asked for.
    wants_channels = num_channels is not None or bool(channel_names)
    if wants_channels:
        n_channels = int(num_channels if num_channels is not None else len(channel_names))
        if channel_names and len(channel_names) != n_channels:
            raise ValueError(f"{len(channel_names)} channel names given for {n_channels} channels.")
        if "c" not in sizes:
            insert_at = 1 if axes_out.startswith("t") else 0
            axes_out = axes_out[:insert_at] + "c" + axes_out[insert_at:]
        elif sizes["c"] != n_channels:
            info["channel_names"], info["channel_colors"] = [], []
        sizes["c"] = n_channels
    elif out_type == "label" and "c" in sizes:
        axes_out = axes_out.replace("c", "")
        sizes.pop("c")

    n_c = sizes.get("c", 0)
    names_out = list(channel_names) if channel_names else list(info["channel_names"])
    if len(names_out) != n_c:
        names_out = [f"channel_{i}" for i in range(n_c)]
    colors_out = list(channel_colors) if channel_colors else list(info["channel_colors"])
    if len(colors_out) != n_c:
        colors_out = [default_channel_color(i) for i in range(n_c)]

    size0 = tuple(int(sizes[ax]) for ax in axes_out)
    dtype_out = str(np.dtype(dtype or info["dtype"]))

    if chunk_size is not None:
        chunk0 = tuple(int(c) for c in _per_axis(chunk_size, axes_out, "chunk_size"))
    elif chunks is not None:
        chunk0 = tuple(int(chunks.get(ax, 1)) for ax in axes_out)
    else:
        chunk0 = default_chunk_size({"axes": axes_out, "size": [size0], "dtype": dtype_out})[0]
    chunk0 = tuple(max(1, min(c, s)) for c, s in zip(chunk0, size0))

    out_format = int(zarr_format or info["zarr_format"] or 3)
    if out_format not in (2, 3):
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format!r}.")
    if shards is None and out_format == 3:
        if template_shards:
            # Keep the template's number of chunks per shard along every axis.
            shards = tuple(
                c * max(1, template_shards.get(ax, 1) // chunks.get(ax, 1)) for c, ax in zip(chunk0, axes_out)
            )
        else:
            shards = "auto"
    elif isinstance(shards, (list, tuple)):
        shards = tuple(int(s) for s in _per_axis(shards, axes_out, "shards"))

    # Pyramid geometry: level k is level 0 shrunk k times.
    num_levels = int(num_levels)
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1.")
    factors = dict(zip(spatial, normalize_spatial_factor_for_axes(downscale_factor, axes_out, allow_float=False)))
    method = default_downsampling_method(out_type)
    level_sizes, level_chunks = [size0], [chunk0]
    scale0 = tuple(scale_by_axis[ax] for ax in spatial)
    if translation is None or isinstance(translation, bool):
        translation0 = dict(origin)
    elif isinstance(translation, Mapping):
        # zarrsmith's translation format: keyed by spatial axis; axes left out keep the origin.
        given = {str(ax).lower(): v for ax, v in translation.items()}
        unknown = sorted(set(given) - set(spatial))
        if unknown:
            raise ValueError(
                f"translation has axes {unknown} that are not spatial axes of the output ({''.join(spatial)})."
            )
        translation0 = {ax: float(given.get(ax, origin[ax])) for ax in spatial}
    else:
        translation0 = dict(zip(spatial, _spatial(translation, axes_out, "translation")))
    level_scales = [scale0]
    f = tuple(int(factors[ax]) for ax in spatial)
    for _ in range(1, num_levels):
        prev = level_sizes[-1]
        shape = tuple(math.ceil(n / factors[ax]) if ax in factors else n for n, ax in zip(prev, axes_out))
        level_sizes.append(shape)
        level_chunks.append(tuple(min(c, s) for c, s in zip(chunk0, shape)))
        level_scales.append(tuple(s * k for s, k in zip(level_scales[-1], f)))

    # Coarser levels by the NGFF 0.6 convention, for the method zarrsmith pyramid will use.
    level_translations = pyramid_translations(translation0, scale0, level_scales, axes_out, method)
    if translation is None:
        write_translation = info["translation"] is not None or moved
    else:
        write_translation = translation is not False
    time_inc = time_increment if time_increment is not None else info["time_increment"]
    if "t" in axes_out and time_inc is None:
        time_inc = 1.0

    metadata: dict[str, Any] = {
        "axes": axes_out,
        "size": level_sizes,
        "chunksize": level_chunks,
        "dtype": dtype_out,
        "scales": level_scales,
        "translations": level_translations if write_translation else None,
        "units": tuple(units_by_axis.get(ax) for ax in spatial) if units_by_axis else (),
        "time_increment": time_inc,
        "time_increment_unit": time_increment_unit or info["time_increment_unit"],
        "channel_names": names_out,
        "channel_colors": colors_out,
        "data_type": out_type,
        "name": name or info["name"] or "image",
        "extra_attributes": dict(extra_attributes) if extra_attributes else None,
    }
    return metadata, out_format, shards


def allocate_zarr(
    out: str | Path,
    *,
    template: str | Path | None = None,
    overwrite: bool = False,
    compressor: str | None = DEFAULT_COMPRESSOR,
    compressor_level: int = DEFAULT_COMPRESSOR_LEVEL,
    shard_target_mb: float = 1024.0,
    **kwargs: Any,
) -> zarr.Group:
    """Create an empty OME-Zarr image at ``out``, from a template or explicit geometry.

    Parameters
    ----------
    out
        The new store. An existing non-empty output is refused unless ``overwrite``.
    template
        OME-Zarr image whose root image gives the axes, level-0 size, dtype, pixel
        size, units, origin (translation), channels, chunks, shards and zarr format.
        Without it, ``size`` and ``axes`` (and optionally ``scale``, ``units``,
        ``time_increment``) describe the image.
    size, axes, scale, units, time_increment, time_increment_unit
        Explicit geometry, when there is no template. ``scale`` is one value or one
        per spatial axis; ``units`` defaults to micrometer. With a template,
        ``scale`` and ``units`` override the template's.
    dtype, data_type
        Output dtype and ``"intensity"``/``"label"`` type (default: the template's).
        A label output drops the template's channel axis unless channels are given.
    channel_names, channel_colors, num_channels
        Channels of the output (adds a ``c`` axis if the template has none).
    crop
        ``{axis: (start, stop)}`` (or slices) of the template to keep. Spatial crops
        move the origin, so the output gets a translation that keeps it in place.
    chunk_size, shards, shard_target_mb
        Level-0 chunks (one per output axis) and shards (``"auto"``, a shape or
        ``None``). By default the template's; ``"auto"`` shards on zarr v3 without a
        template. Zarr v2 has no shards: an explicit shape raises, ``"auto"`` warns.
    num_levels, downscale_factor
        Levels created (default 1: fill level 0 then run ``zarrsmith pyramid``).
    translation
        ``None`` writes a translation when the template has one or the crop moves
        the origin; ``True`` always writes one; ``False`` writes none. A dict keyed
        by spatial axis, as zarrsmith stores translations (e.g.
        ``{"z": 0.0, "y": 10.0, "x": 20.0}``), sets the level-0 origin; axes it
        leaves out keep the template's origin (moved by ``crop``). A sequence with
        one value per spatial axis, in axis order, is also accepted.
    zarr_format
        2 (NGFF 0.4) or 3 (NGFF 0.5); defaults to the template's format, else 3.
    compressor, compressor_level
        As in :class:`~zarrsmith.utils.ngff.ZarrWriteConfig`.
    extra_attributes
        Custom group attributes (e.g. ``{"t_range": [0, 10]}``), stored next to the
        NGFF metadata (outside ``ome`` on zarr v3).
    name
        Name of the multiscales image (default: the template's).
    """
    from .create_empty_dataset import create_empty_dataset
    from .ngff import _check_compressor
    from .output_path import check_output_path

    metadata, zarr_format, shards = build_allocation_metadata(template=template, **kwargs)
    ngff_version = "0.4" if zarr_format == 2 else "0.5"
    _check_compressor(compressor, compressor_level, zarr_format)
    check_output_path(out, overwrite=overwrite, sources=[template] if template else [])

    root = zarr.open_group(str(Path(out)), mode="w" if overwrite else "w-", zarr_format=zarr_format)
    create_empty_dataset(
        root,
        metadata,
        ngff_version=ngff_version,
        zarr_format=zarr_format,
        compressor=compressor,
        compressor_level=compressor_level,
        data_type=metadata["data_type"],
        shards=shards,
        shard_target_mb=shard_target_mb,
    )
    return root
