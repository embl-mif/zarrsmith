"""Regions of an OME-Zarr image: physical <-> pixel bounds, and reads with a halo.

Positions follow the NGFF 0.6 convention: the ``translation`` of a level is the
centre of its first pixel, so pixel ``i`` of an axis with pixel size ``s`` and
translation ``t`` is centred on ``t + i * s`` and covers
``[t + (i - 1/2) * s, t + (i + 1/2) * s)``.

Bounds are half-open ``(start, stop)`` per axis name. Pixel bounds are integers
at one pyramid level; physical bounds are in the axes' units (e.g. micrometer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .axes import spatial_axes_in_order

_EPS = 1e-9

PixelBounds = Mapping[str, tuple[int, int]]
PhysicalBounds = Mapping[str, tuple[float | None, float | None]]


def _value_for_every_axis(value: float | Mapping[str, float] | None, axes: Sequence[str], default: float) -> dict[str, float]:
    if value is None:
        return {ax: default for ax in axes}
    if isinstance(value, Mapping):
        return {ax: float(value.get(ax, default)) for ax in axes}
    return {ax: float(value) for ax in axes}


def physical_to_pixel_bounds(
    bounds: PhysicalBounds,
    scale: Mapping[str, float],
    translation: Mapping[str, float] | None = None,
    shape: Mapping[str, int] | None = None,
    *,
    clip: bool = False,
) -> dict[str, tuple[int, int]]:
    """Pixels that overlap physical ``bounds``, per axis.

    Parameters
    ----------
    bounds
        ``{axis: (start, stop)}`` in physical units, half-open. ``None`` for a
        side means the edge of the image (needs ``shape``).
    scale, translation
        Pixel size and translation (centre of the first pixel) of the level, by
        axis. A missing translation is 0.
    shape
        Size of the level, by axis: used for ``None`` sides and to check the
        result lies inside the image.
    clip
        Clip bounds that reach outside the image instead of raising.

    A pixel is included when any part of it lies inside ``[start, stop)``; a
    bound exactly on the edge between two pixels does not include the pixel that
    only touches it.
    """
    out: dict[str, tuple[int, int]] = {}
    for ax, (lo, hi) in bounds.items():
        s = float(scale[ax])
        if s <= 0:
            raise ValueError(f"Pixel size of {ax!r} must be > 0, got {s}.")
        t = float((translation or {}).get(ax, 0.0))
        n = None if shape is None else int(shape[ax])
        if (lo is None or hi is None) and n is None:
            raise ValueError(f"An open bound on {ax!r} needs the image shape.")
        if lo is not None and hi is not None and float(hi) <= float(lo):
            raise ValueError(f"{ax}-bounds [{lo}, {hi}) cover no pixel.")
        start = 0 if lo is None else math.floor((float(lo) - t) / s - 0.5 + _EPS) + 1
        stop = n if hi is None else math.ceil((float(hi) - t) / s + 0.5 - _EPS)
        if n is not None and (start < 0 or stop > n):
            if not clip:
                extent = pixel_to_physical_bounds({ax: (0, n)}, scale, translation)[ax]
                raise ValueError(
                    f"{ax}-bounds [{lo}, {hi}) reach outside the image, which covers "
                    f"[{extent[0]:g}, {extent[1]:g})."
                )
            start, stop = max(0, start), min(n, stop)
        if stop <= start:
            raise ValueError(f"{ax}-bounds [{lo}, {hi}) cover no pixel.")
        out[ax] = (int(start), int(stop))
    return out


def pixel_to_physical_bounds(
    bounds: PixelBounds,
    scale: Mapping[str, float],
    translation: Mapping[str, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Physical extent ``[first pixel's lower edge, last pixel's upper edge)`` of pixel ``bounds``."""
    out = {}
    for ax, (start, stop) in bounds.items():
        s = float(scale[ax])
        t = float((translation or {}).get(ax, 0.0))
        out[ax] = (t + (int(start) - 0.5) * s, t + (int(stop) - 0.5) * s)
    return out


def physical_to_pixels(length: float, scale: float) -> int:
    """Smallest number of pixels of size ``scale`` covering ``length`` (rounded up)."""
    if length < 0:
        raise ValueError(f"A length must be >= 0, got {length}.")
    return max(0, math.ceil(float(length) / float(scale) - _EPS))


def halo_window(
    bounds: PixelBounds,
    halo: int | Mapping[str, int],
    shape: Mapping[str, int],
) -> tuple[dict[str, tuple[int, int]], dict[str, slice]]:
    """The pixel ``bounds`` grown by ``halo`` on each side, clipped to the image.

    Returns ``(window, crop)``: the bounds to read, and per axis the slice of the
    read block that gives back exactly ``bounds``.
    """
    window, crop = {}, {}
    for ax, (start, stop) in bounds.items():
        h = int(halo.get(ax, 0) if isinstance(halo, Mapping) else halo)
        if h < 0:
            raise ValueError(f"The halo must be >= 0, got {h} for {ax!r}.")
        lo, hi = max(0, int(start) - h), min(int(shape[ax]), int(stop) + h)
        window[ax] = (lo, hi)
        crop[ax] = slice(int(start) - lo, int(stop) - lo)
    return window, crop


@dataclass
class Region:
    """A block read from one level of an image, with its halo.

    ``data`` is the block as read (core plus halo); ``core`` is the requested
    region alone. ``bounds`` and ``window`` are the pixel bounds of the core and
    of the block, for every axis of the image. ``source`` is the metadata of the
    image it was read from (set by ``ZarrManager.read_region``), which
    :meth:`to_label_image` follows.
    """

    data: np.ndarray
    axes: str
    level: int
    bounds: dict[str, tuple[int, int]]
    window: dict[str, tuple[int, int]]
    scale: dict[str, float] = field(default_factory=dict)
    translation: dict[str, float] = field(default_factory=dict)
    source: dict | None = field(default=None, repr=False, compare=False)

    @property
    def crop(self) -> tuple[slice, ...]:
        """Index of the core inside ``data``."""
        return tuple(
            slice(self.bounds[ax][0] - self.window[ax][0], self.bounds[ax][1] - self.window[ax][0])
            for ax in self.axes
        )

    @property
    def core(self) -> np.ndarray:
        return self.data[self.crop]

    @property
    def halo(self) -> dict[str, tuple[int, int]]:
        """Pixels of halo actually read before and after the core, per axis."""
        return {ax: (self.bounds[ax][0] - self.window[ax][0], self.window[ax][1] - self.bounds[ax][1])
                for ax in self.axes}

    def physical_bounds(self, which: str = "core") -> dict[str, tuple[float, float]]:
        """Physical extent of the core (or of the read ``"window"``), spatial axes only."""
        src = self.bounds if which == "core" else self.window
        return pixel_to_physical_bounds({ax: src[ax] for ax in self.scale}, self.scale, self.translation)

    def core_translation(self) -> dict[str, float]:
        """Translation of an image holding only the core: the centre of its first pixel."""
        return {ax: self.translation.get(ax, 0.0) + self.bounds[ax][0] * s for ax, s in self.scale.items()}

    def to_label_image(
        self,
        labels,
        path,
        *,
        num_levels: int | None = None,
        downscale_factor=None,
        method: str = "nearest",
        chunks: Sequence[int] | None = None,
        ngff_version: str | None = None,
        zarr_format: int | None = None,
        dtype="uint32",
        overwrite: bool = False,
        **write_options,
    ):
        """Write ``labels`` segmented from this region as a standalone OME-Zarr label image.

        ``labels`` covers the core (``.core``) without its channel axis, or the whole block
        read (``.data``, halo included): it is then cropped to the core. With a ``t`` axis in
        the region, ``labels`` may keep it or not (one timepoint). The image is placed at the
        core (:meth:`core_translation`), with the voxel size of the region's level, is marked
        as label data and has ``dtype`` (an id that does not fit raises ``ValueError``).

        The rest follows the source image (``source``, set by ``ZarrManager.read_region``):
        units and time step, the format (NGFF 0.4 / zarr v2 for a 0.4 source, else 0.5 / zarr v3;
        ``ngff_version``/``zarr_format`` override it), the chunks (clipped to the labels) and the
        pyramid from the region's level on (:func:`zarrsmith.utils.dataset_ops.infer_downscale_factors`),
        sampled with ``method`` ("nearest" or "mode") so ids are never mixed. ``num_levels=1``
        writes level 0 only. Without a source: one level, zarr v3, micrometer.
        ``write_options`` go to the writer (e.g. ``shards``, ``compressor``). Returns the written
        image as a ``ZarrManager``.
        """
        import dask.array as da

        from ..microscope_manager import ArrayManager, ZarrManager
        from .dataset_ops import infer_downscale_factors

        labels = np.asarray(labels)
        no_c = [ax for ax in self.axes if ax != "c"]
        crop = dict(zip(self.axes, self.crop))
        if "t" in no_c and labels.ndim == len(no_c) - 1:
            no_c.remove("t")  # one timepoint
        if labels.ndim != len(no_c):
            raise ValueError(f"labels have {labels.ndim} axes; expected {len(no_c)} ({''.join(no_c)}).")
        core_shape = tuple(self.bounds[ax][1] - self.bounds[ax][0] for ax in no_c)
        window_shape = tuple(self.window[ax][1] - self.window[ax][0] for ax in no_c)
        if labels.shape == window_shape and labels.shape != core_shape:
            labels = labels[tuple(crop[ax] for ax in no_c)]
        elif labels.shape != core_shape:
            raise ValueError(f"labels have shape {labels.shape}; expected the core {core_shape} "
                             f"or the block read {window_shape} ({''.join(no_c)}).")
        dtype = np.dtype(dtype)
        if dtype.kind not in "iu":
            raise ValueError(f"Labels need an integer dtype, got {dtype}.")
        if labels.size and (labels.max() > np.iinfo(dtype).max or labels.min() < np.iinfo(dtype).min):
            raise ValueError(f"Label ids up to {int(labels.max())} do not fit in {dtype}.")
        labels = labels.astype(dtype)

        source = self.source or {}
        axes = "".join(no_c)
        spatial = spatial_axes_in_order(axes)
        src_spatial = spatial_axes_in_order(source.get("axes") or self.axes)
        src_units = list(source.get("units") or ())
        units = dict(zip(src_spatial, src_units)) if len(src_units) == len(src_spatial) else {}
        translation = self.core_translation()
        metadata = {
            "axes": axes,
            "size": [labels.shape],
            "dtype": str(dtype),
            "data_type": "label",
            "scales": [tuple(float(self.scale.get(ax, 1.0)) for ax in spatial)],
            "units": tuple(units.get(ax, "micrometer") for ax in spatial),
            "translations": [{ax: float(translation.get(ax, 0.0)) for ax in spatial}],
        }
        if "t" in axes:
            for key in ("time_increment", "time_increment_unit"):
                if source.get(key) is not None:
                    metadata[key] = source[key]

        if chunks is None:
            src_axes = "".join(source.get("axes") or self.axes)
            src_chunks = (source.get("chunksize") or [None] * (self.level + 1))[self.level] if source else None
            by_axis = dict(zip(src_axes, src_chunks)) if src_chunks else {}
            chunks = tuple(min(int(by_axis.get(ax, n)), int(n)) for ax, n in zip(axes, labels.shape))
        metadata["chunksize"] = [tuple(chunks)]

        if num_levels is None or downscale_factor is None:
            levels, factor = infer_downscale_factors(source, self.level) if source.get("scales") else (1, (2, 2, 2))
            num_levels = levels if num_levels is None else num_levels
            downscale_factor = factor if downscale_factor is None else downscale_factor
        if zarr_format is None:
            zarr_format = 2 if ngff_version == "0.4" else (int(source.get("zarr_format") or 3) if ngff_version is None else 3)
        ngff_version = ngff_version or ("0.4" if int(zarr_format) == 2 else "0.5")

        manager = ArrayManager([da.from_array(labels, chunks=tuple(chunks))], metadata)
        if int(num_levels) > 1:
            manager.build_pyramid(num_levels=int(num_levels), downscale_factor=downscale_factor, method=method)
        write_options.setdefault("drop_singleton", False)  # keep the axes as segmented
        write_options.setdefault("progress", False)
        manager.to_zarr(str(path), ngff_version=ngff_version, zarr_format=int(zarr_format), overwrite=overwrite,
                        **write_options)
        return ZarrManager(str(path), mode="r")


def read_region(
    array,
    axes: str,
    *,
    level: int = 0,
    scale: Mapping[str, float] | None = None,
    translation: Mapping[str, float] | None = None,
    selection: Mapping[str, int | slice | tuple] | None = None,
    halo: int | float | Mapping[str, float] = 0,
    physical: bool = False,
    clip: bool = False,
) -> Region:
    """Read ``selection`` of ``array`` (zarr, dask or numpy) plus a halo.

    ``selection`` maps axis names to an index, a slice or a ``(start, stop)``
    pair; axes left out are read whole. ``halo`` (one value, or one per axis)
    is added on both sides of the spatial axes only and clipped at the image
    edge. With ``physical=True``, spatial bounds and the halo are in physical
    units and converted with ``scale`` and ``translation`` (see
    :func:`physical_to_pixel_bounds`); ``t`` and ``c`` are always indices.
    """
    axes = "".join(axes)
    shape = dict(zip(axes, (int(n) for n in array.shape)))
    spatial = list(spatial_axes_in_order(axes))
    scale_d = _value_for_every_axis(scale, spatial, 1.0) if scale is not None else {ax: 1.0 for ax in spatial}
    trans_d = _value_for_every_axis(translation, spatial, 0.0)

    selection = {str(k).lower(): v for k, v in (selection or {}).items()}
    unknown = sorted(set(selection) - set(axes))
    if unknown:
        raise ValueError(f"Selection on {unknown}, which the image (axes {axes!r}) does not have.")

    def as_range(ax, sel):
        if sel is None:
            return (0, shape[ax])
        if isinstance(sel, slice):
            if sel.step not in (None, 1):
                raise ValueError(f"A region must be contiguous; got step {sel.step} on {ax!r}.")
            return (sel.start, sel.stop)
        if isinstance(sel, (int, np.integer)):
            i = int(sel) + (shape[ax] if sel < 0 else 0)
            return (i, i + 1)
        lo, hi = sel
        return (lo, hi)

    ranges = {ax: as_range(ax, selection.get(ax)) for ax in axes}
    if physical:
        phys = {ax: ranges[ax] for ax in spatial if ax in selection}
        ranges.update(physical_to_pixel_bounds(phys, scale_d, trans_d, shape, clip=clip))
        halo_px = {ax: physical_to_pixels(h, scale_d[ax]) for ax, h in _value_for_every_axis(halo, spatial, 0.0).items()}
    else:
        halo_px = {ax: int(h) for ax, h in _value_for_every_axis(halo, spatial, 0.0).items()}
        if any(float(h) != int(h) for h in _value_for_every_axis(halo, spatial, 0.0).values()):
            raise ValueError("A halo in pixels must be whole; use physical=True for physical units.")

    bounds = {}
    for ax in axes:
        lo, hi = ranges[ax]
        n = shape[ax]
        lo = 0 if lo is None else int(lo)
        hi = n if hi is None else int(hi)
        lo, hi = (lo + n if lo < 0 else lo), (hi + n if hi < 0 else hi)
        if clip:
            lo, hi = max(0, lo), min(n, hi)
        if not 0 <= lo < hi <= n:
            raise ValueError(f"{ax}-range [{lo}, {hi}) is empty or outside the image (size {n}).")
        bounds[ax] = (lo, hi)

    window, _ = halo_window({ax: bounds[ax] for ax in spatial}, halo_px, shape)
    window = {ax: window.get(ax, bounds[ax]) for ax in axes}
    data = np.asarray(array[tuple(slice(*window[ax]) for ax in axes)])
    return Region(data=data, axes=axes, level=level, bounds=bounds, window=window,
                  scale=scale_d, translation=trans_d)
