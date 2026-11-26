from __future__ import annotations

from collections.abc import Iterator, Sequence
from numbers import Real
from typing import Any, TypeAlias

import dask.array as da
import numpy as np

from .axes import SPATIAL_AXIS_SET, normalize_axes, spatial_values_for_axes

SpatialFactor: TypeAlias = int | float | Sequence[int | float]
SpatialScale: TypeAlias = tuple[float, ...]
SpatialIntFactor: TypeAlias = tuple[int, ...]
SPATIAL_AXES = SPATIAL_AXIS_SET


def normalize_spatial_factor(
    value: SpatialFactor,
    *,
    name: str = "downscale_factor",
    allow_float: bool = True,
) -> SpatialScale | SpatialIntFactor:
    """Normalize scalar or ZYX factor to a 3-tuple.

    This helper keeps the old public behavior.  Axis-aware code should use
    :func:`normalize_spatial_factor_for_axes`.

    Scalar 2      -> (2, 2, 2)
    Tuple (1,2,2) -> (1, 2, 2)
    """
    if isinstance(value, Real) and not isinstance(value, bool):
        values = (float(value),) * 3
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(
                f"{name} must be a scalar or a ZYX sequence of length 3, got {value!r}."
            )
        values = tuple(float(v) for v in value)
    else:
        raise TypeError(
            f"{name} must be a scalar or a ZYX sequence of length 3, "
            f"got {type(value).__name__}."
        )

    if any(v <= 0 for v in values):
        raise ValueError(f"{name} values must be > 0, got {values!r}.")

    if not allow_float:
        ints = tuple(int(v) for v in values)
        if any(float(i) != v for i, v in zip(ints, values)):
            raise ValueError(f"{name} values must be integers, got {values!r}.")
        return ints
    return values



def normalize_spatial_factor_for_axes(
    value: SpatialFactor,
    axes: str | Sequence[str],
    *,
    name: str = "downscale_factor",
    allow_float: bool = True,
) -> SpatialScale | SpatialIntFactor:
    """Normalize a factor to the spatial axes present in ``axes``."""
    return spatial_values_for_axes(value, axes, name=name, allow_float=allow_float)


def spatial_factor_power_for_axes(
    factor: SpatialFactor,
    axes: str | Sequence[str],
    exponent: int,
) -> SpatialScale:
    values = normalize_spatial_factor_for_axes(factor, axes, allow_float=True)
    if exponent >= 0:
        return tuple(v**exponent for v in values)
    return tuple(1.0 / (v ** (-exponent)) for v in values)


def relative_level_factors_for_axes(
    total_levels: int,
    ref_level: int,
    axes: str | Sequence[str],
    downscale_factor: SpatialFactor = 2,
) -> list[SpatialScale]:
    """Return axis-aware level scale ratios relative to ``ref_level``."""
    if not 0 <= ref_level < total_levels:
        raise ValueError(f"ref_level must be in [0, {total_levels}), got {ref_level}.")
    return [
        spatial_factor_power_for_axes(downscale_factor, axes, level - ref_level)
        for level in range(total_levels)
    ]

def iter_dataset_transforms(dataset: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The transformations of one multiscales dataset, in order.

    NGFF 0.4/0.5 list ``[scale, translation]`` directly; NGFF 0.6 may wrap them
    in one ``{"type": "sequence", "transformations": [...]}``, which is unpacked.
    """
    for transform in dataset.get("coordinateTransformations", []) or []:
        if not isinstance(transform, dict):
            continue
        if transform.get("type") == "sequence":
            yield from iter_dataset_transforms(
                {"coordinateTransformations": transform.get("transformations", [])}
            )
        else:
            yield transform


def dataset_transform_vector(dataset: dict[str, Any], kind: str) -> list[float] | None:
    """The ``scale`` or ``translation`` vector of a dataset, or ``None`` if it has none."""
    for transform in iter_dataset_transforms(dataset):
        # An untyped entry with a "scale" key is read as a scale, as before.
        if transform.get("type", kind) == kind and transform.get(kind) is not None:
            return [float(v) for v in transform[kind]]
    return None


def multiscales_axes(multiscales: dict[str, Any]) -> list[Any]:
    """The axes entries of a multiscales block, whatever the NGFF version.

    NGFF 0.4/0.5 store them in ``axes``. NGFF 0.6 moves them into
    ``coordinateSystems``: the one the level transforms output to (the
    image's intrinsic system) is used, else the first one.
    """
    if multiscales.get("axes"):
        return list(multiscales["axes"])
    systems = multiscales.get("coordinateSystems") or []
    if not systems:
        return []
    target = None
    datasets = multiscales.get("datasets") or []
    if datasets:
        for transform in datasets[0].get("coordinateTransformations", []) or []:
            output = transform.get("output") if isinstance(transform, dict) else None
            if isinstance(output, dict):
                target = output.get("name")
            elif isinstance(output, str):
                target = output
            if target:
                break
    for system in systems:
        if target is not None and system.get("name") == target:
            return list(system.get("axes") or [])
    return list(systems[0].get("axes") or [])


def axis_names_from_multiscales(multiscales: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for axis in multiscales_axes(multiscales):
        if isinstance(axis, dict):
            name = axis.get("name")
        else:
            name = axis
        if name is not None:
            names.append(str(name).lower())
    return names


def _dataset_scale_vector(dataset: dict[str, Any]) -> Sequence[int | float] | None:
    return dataset_transform_vector(dataset, "scale")


def level_scale_ratios_from_multiscales(
    multiscales: dict[str, Any],
    datasets: Sequence[dict[str, Any]],
    ref_level: int,
) -> list[SpatialScale] | None:
    """Infer relative spatial level factors from NGFF multiscales metadata.

    Returned tuples are ordered like the spatial axes in the metadata, not
    necessarily legacy ZYX.
    """
    if not datasets or not 0 <= ref_level < len(datasets):
        return None

    axis_names = axis_names_from_multiscales(multiscales)
    if not axis_names:
        return None
    axes = normalize_axes(axis_names)
    spatial_positions = [i for i, name in enumerate(axes) if name in SPATIAL_AXIS_SET]

    level_scales: list[SpatialScale] = []
    for dataset in datasets:
        scale_vector = _dataset_scale_vector(dataset)
        if scale_vector is None:
            return None
        if len(scale_vector) != len(axes):
            return None
        spatial = tuple(float(scale_vector[i]) for i in spatial_positions)
        level_scales.append(spatial)

    ref_scale = level_scales[ref_level]
    return [
        tuple(
            level_scale[i] / ref_scale[i]
            for i in range(len(ref_scale))
        )
        for level_scale in level_scales
    ]


def downsample_nearest(arr, factors: Sequence[int], spatial_axes: Sequence[int]):
    """Nearest-neighbour downsampling by striding the spatial axes.

    Works on NumPy and Dask arrays. Axes whose length is not divisible by the
    factor are edge-padded first, so the output length is ``ceil(n / factor)``.
    A factor of 1 leaves an axis untouched.
    """
    if len(factors) != len(spatial_axes):
        raise ValueError("factors must match spatial_axes length.")
    xp = da if isinstance(arr, da.Array) else np
    pad_width = [(0, 0)] * arr.ndim
    slicing = [slice(None)] * arr.ndim
    for axis, factor in zip(spatial_axes, factors):
        factor = int(factor)
        if factor <= 1:
            continue
        remainder = int(arr.shape[axis]) % factor
        if remainder:
            pad_width[axis] = (0, factor - remainder)
        slicing[axis] = slice(0, None, factor)
    if any(width != (0, 0) for width in pad_width):
        arr = xp.pad(arr, pad_width, mode="edge")
    return arr[tuple(slicing)]


#: Downsampling methods of :func:`downsample` (and the pyramid builders).
DOWNSAMPLING_METHODS: tuple[str, ...] = ("mean", "nearest", "mode")


def downsampling_record(method: str, factors: Sequence[int | float] | None = None) -> dict[str, Any]:
    """The ``multiscales`` ``type`` and ``metadata`` entries naming how a pyramid was built.

    NGFF 0.4-0.6: ``type`` SHOULD name the downscaling method and ``metadata`` SHOULD
    describe it. ``factors`` (spatial, between consecutive levels) is left out when
    unknown. :func:`recorded_downsampling_method` reads it back.
    """
    from .. import __version__

    kwargs: dict[str, Any] = {"method": method}
    blocks = "blocks"
    if factors:
        kwargs["factors"] = [int(f) if float(f).is_integer() else float(f) for f in factors]
        blocks = f"{'x'.join(map(str, kwargs['factors']))} blocks"
    return {
        "type": method,
        "metadata": {
            "description": f"Each level is the {method} of {blocks} of the level above it.",
            "method": "zarrsmith.utils.downsampling.downsample_numpy",
            "version": __version__,
            "kwargs": kwargs,
        },
    }


def recorded_downsampling_method(multiscales: dict[str, Any] | None) -> str | None:
    """The downsampling method a ``multiscales`` block records, or ``None``.

    Read from ``type`` (see :func:`downsampling_record`), else from the
    ``metadata`` ``kwargs`` that zarrsmith writes next to it.
    """
    if not multiscales:
        return None
    ms_type = multiscales.get("type")
    if isinstance(ms_type, str) and ms_type.lower() in DOWNSAMPLING_METHODS:
        return ms_type.lower()
    meta = multiscales.get("metadata")
    kwargs = meta.get("kwargs") if isinstance(meta, dict) else None
    method = kwargs.get("method") if isinstance(kwargs, dict) else None
    if isinstance(method, str) and method.lower() in DOWNSAMPLING_METHODS:
        return method.lower()
    return None


def default_downsampling_method(data_type: str | None) -> str:
    """``"nearest"`` for label data, ``"mean"`` for intensity (and probability) data."""
    from .axes import normalize_data_type

    return "nearest" if normalize_data_type(data_type) == "label" else "mean"


def resolve_downsampling_method(method: str | None, data_type: str | None) -> str:
    """Validate ``method`` for data of ``data_type``; ``None`` picks the default.

    ``"mean"`` invents label values that are in neither neighbour, so it is
    refused for label data.
    """
    from .axes import normalize_data_type

    if method is None:
        return default_downsampling_method(data_type)
    method = str(method).strip().lower()
    if method not in DOWNSAMPLING_METHODS:
        raise ValueError(f"Unknown downsampling method {method!r}; use one of {', '.join(DOWNSAMPLING_METHODS)}.")
    if method == "mean" and normalize_data_type(data_type) == "label":
        raise ValueError("method='mean' would mix label values: use 'nearest' or 'mode' for labels.")
    return method




def mean_accumulator_dtype(dtype) -> np.dtype:
    """Float type ``"mean"`` averages in: float32 for 8/16-bit integers, bool and
    float16/32 (exact enough, half the memory of float64), float64 otherwise."""
    dtype = np.dtype(dtype)
    if dtype.itemsize <= 2 or dtype == np.float32:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _block_mode(windows: np.ndarray) -> np.ndarray:
    """Most frequent value along the last axis (ties: the smallest value)."""
    ordered = np.sort(windows, axis=-1)
    counts = np.zeros(ordered.shape, dtype=np.int32)
    for i in range(ordered.shape[-1]):
        counts[..., i] = (ordered == ordered[..., i : i + 1]).sum(axis=-1)
    best = np.argmax(counts, axis=-1)  # first maximum: the smallest of tied values
    return np.take_along_axis(ordered, best[..., None], axis=-1)[..., 0]


def downsample_numpy(
    arr: np.ndarray,
    factors: Sequence[int],
    spatial_axes: Sequence[int],
    method: str = "nearest",
) -> np.ndarray:
    """Downsample a NumPy array along ``spatial_axes`` by integer ``factors``.

    ``"nearest"`` keeps the first pixel of every block, ``"mean"`` averages the
    block (rounded back to the input dtype for integers) and ``"mode"`` keeps its
    most frequent value. Axes that do not divide by their factor are edge-padded
    first, so the output length is always ``ceil(n / factor)``.
    """
    if len(factors) != len(spatial_axes):
        raise ValueError("factors must match spatial_axes length.")
    if method == "nearest":
        return np.asarray(downsample_nearest(arr, factors, spatial_axes))
    if method not in DOWNSAMPLING_METHODS:
        raise ValueError(f"Unknown downsampling method {method!r}; use one of {', '.join(DOWNSAMPLING_METHODS)}.")

    per_axis = [1] * arr.ndim
    for axis, factor in zip(spatial_axes, factors):
        per_axis[axis] = int(factor)
    if all(f == 1 for f in per_axis):
        return arr
    pad = [(0, -n % f) for n, f in zip(arr.shape, per_axis)]
    if any(p != (0, 0) for p in pad):
        arr = np.pad(arr, pad, mode="edge")

    # (n0/f0, f0, n1/f1, f1, ...) then move every factor axis to the end.
    split = []
    for n, f in zip(arr.shape, per_axis):
        split.extend((n // f, f))
    blocks = arr.reshape(split)
    ndim = arr.ndim
    blocks = blocks.transpose([2 * i for i in range(ndim)] + [2 * i + 1 for i in range(ndim)])
    windows = blocks.reshape(blocks.shape[:ndim] + (-1,))

    if method == "mode":
        return _block_mode(windows)
    out = windows.mean(axis=-1, dtype=mean_accumulator_dtype(arr.dtype))
    dtype = arr.dtype
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(dtype)
    if np.issubdtype(dtype, np.bool_):
        return out >= 0.5
    return out.astype(dtype)


def downsample(arr, factors: Sequence[int], spatial_axes: Sequence[int], method: str = "nearest"):
    """Downsample a NumPy or Dask array (see :func:`downsample_numpy` for ``method``)."""
    if method == "nearest" or not isinstance(arr, da.Array):
        if method == "nearest":
            return downsample_nearest(arr, factors, spatial_axes)
        return downsample_numpy(arr, factors, spatial_axes, method)

    per_axis = [1] * arr.ndim
    for axis, factor in zip(spatial_axes, factors):
        per_axis[axis] = int(factor)
    pad = [(0, -n % f) for n, f in zip(arr.shape, per_axis)]
    if any(p != (0, 0) for p in pad):
        arr = da.pad(arr, pad, mode="edge")
    # Every block must hold whole windows: round the chunks up to multiples of the factors.
    chunks = tuple(
        max(f, -(-int(c) // f) * f) for c, f in zip(arr.chunksize, per_axis)
    )
    arr = arr.rechunk(chunks)
    out_chunks = tuple(tuple(c // f for c in axis_chunks) for axis_chunks, f in zip(arr.chunks, per_axis))
    return arr.map_blocks(
        downsample_numpy,
        factors,
        spatial_axes,
        method,
        chunks=out_chunks,
        dtype=arr.dtype,
    )
