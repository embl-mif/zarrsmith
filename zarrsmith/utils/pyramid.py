from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dask.array as da

from .axes import normalize_axes, spatial_axis_indices, spatial_axes_in_order
from .chunks import rechunk_to_shape, shape_tuple
from .downsampling import downsample, normalize_spatial_factor_for_axes
from .translation import normalize_translations, pyramid_translations

SpatialFactor = Union[int, Sequence[int]]


def _target_chunks(data_level: da.Array, metadata: Dict[str, Any]) -> tuple[int, ...]:
    """Choose the level-independent chunk shape for a pyramid build."""
    ndim = data_level.ndim
    metadata_chunks = metadata.get("chunksize")
    if metadata_chunks:
        target = shape_tuple(metadata_chunks, ndim)
        if target is not None:
            return target
        try:
            target = shape_tuple(metadata_chunks[0], ndim)
        except (TypeError, IndexError):
            target = None
        if target is not None:
            return target
    return tuple(int(c) for c in data_level.chunksize)


def factor_power(factors: Sequence[int], exponent: int) -> tuple[int, ...]:
    return tuple(int(f) ** exponent for f in factors)


def multiply_scales(scales: Sequence[float], factors: Sequence[int]) -> tuple[float, ...]:
    if len(scales) != len(factors):
        raise ValueError("scales and factors must have the same length.")
    return tuple(float(s) * f for s, f in zip(scales, factors))


def build_pyramid(
    data_levels: List[da.Array],
    metadata: Dict[str, Any],
    num_levels: int = 3,
    downscale_factor: SpatialFactor = 2,
    start_level: int = 0,
    method: Optional[str] = None,
) -> Tuple[List[da.Array], Dict[str, Any]]:
    """
    Generate a multiscale pyramid and updated metadata for NGFF-compatible
    OME-Zarr writing.

    This function is axis-aware and uses metadata["axes"] to find Z, Y, X.

    downscale_factor can be either:

    - 2, meaning downsample Z, Y and X by 2
    - (1, 2, 2), meaning keep Z unchanged and downsample only YX

    ``method`` is ``"mean"``, ``"nearest"`` or ``"mode"`` (see
    :func:`zarrsmith.utils.downsampling.downsample_numpy`); ``None`` (default)
    uses ``"mean"`` for intensity data and ``"nearest"`` for labels, as
    :func:`zarrsmith.utils.populate_pyramid.populate_pyramid` does. The
    origin of every new level follows the method, by the NGFF 0.6 convention (see
    :func:`zarrsmith.utils.translation.pyramid_translations`), and the method is
    kept in ``metadata['downsampling_method']``, which the writer records in the
    ``multiscales`` ``type`` so that region writes refresh the levels the same way.
    """
    from .downsampling import resolve_downsampling_method

    method = resolve_downsampling_method(method, metadata.get("data_type"))
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")
    if num_levels is None:
        num_levels = len(data_levels)
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1.")
    if start_level < 0:
        raise ValueError("start_level must be >= 0.")

    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    metadata = dict(metadata)
    metadata["axes"] = "".join(axes)

    for level in data_levels:
        if level.ndim != len(axes):
            raise ValueError(
                f"Array ndim ({level.ndim}) does not match metadata['axes'] ({metadata['axes']!r})."
            )

    spatial_axes = spatial_axis_indices(axes)
    spatial_labels = spatial_axes_in_order(axes)
    if "scales" not in metadata or not metadata["scales"]:
        metadata["scales"] = [tuple(1.0 for _ in spatial_labels) for _ in data_levels]

    if len(metadata["scales"][0]) != len(spatial_axes):
        raise ValueError(
            "metadata['scales'] entries must match the number of spatial axes "
            f"({len(spatial_axes)})."
        )

    factors = normalize_spatial_factor_for_axes(downscale_factor, axes)
    target_chunks = _target_chunks(data_levels[0], metadata)

    # The level the new pyramid starts from, whose position is known.
    source = min(start_level, len(data_levels) - 1) if start_level < len(data_levels) else 0
    source_scale = tuple(metadata["scales"][min(source, len(metadata["scales"]) - 1)])
    translations = normalize_translations(metadata.get("translations"), axes)
    source_translation = None if translations is None else translations[min(source, len(translations) - 1)]

    if start_level < len(data_levels):
        pyramid = [rechunk_to_shape(data_levels[start_level], target_chunks)]
        new_scales = [source_scale]
    else:
        base_downscale = factor_power(factors, start_level)
        down = downsample(data_levels[0], base_downscale, spatial_axes, method)
        pyramid = [rechunk_to_shape(down, target_chunks)]
        new_scales = [multiply_scales(metadata["scales"][0], base_downscale)]

    for _ in range(1, num_levels):
        if spatial_axes:
            down = downsample(pyramid[-1], factors, spatial_axes, method)
        else:
            down = pyramid[-1]
        pyramid.append(rechunk_to_shape(down, target_chunks))

    for level in range(1, num_levels):
        scale_factor = factor_power(factors, level)
        new_scales.append(multiply_scales(new_scales[0], scale_factor))

    metadata["scales"] = new_scales
    # NGFF 0.6 convention: see zarrsmith.utils.translation.pyramid_translations.
    metadata["translations"] = pyramid_translations(
        source_translation, source_scale, new_scales, axes, method
    )
    metadata["downsampling_method"] = method
    metadata["size"] = [tuple(level.shape) for level in pyramid]
    metadata["chunksize"] = [tuple(level.chunksize) for level in pyramid]
    return pyramid, metadata
