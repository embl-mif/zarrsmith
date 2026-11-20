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

_EPS = 1e-9

PixelBounds = Mapping[str, tuple[int, int]]
PhysicalBounds = Mapping[str, tuple[float | None, float | None]]


def _per_axis(value: float | Mapping[str, float] | None, axes: Sequence[str], default: float) -> dict[str, float]:
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
    of the block, for every axis of the image.
    """

    data: np.ndarray
    axes: str
    level: int
    bounds: dict[str, tuple[int, int]]
    window: dict[str, tuple[int, int]]
    scale: dict[str, float] = field(default_factory=dict)
    translation: dict[str, float] = field(default_factory=dict)

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
    from .axes import spatial_axes_in_order

    axes = "".join(axes)
    shape = dict(zip(axes, (int(n) for n in array.shape)))
    spatial = list(spatial_axes_in_order(axes))
    scale_d = _per_axis(scale, spatial, 1.0) if scale is not None else {ax: 1.0 for ax in spatial}
    trans_d = _per_axis(translation, spatial, 0.0)

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
        halo_px = {ax: physical_to_pixels(h, scale_d[ax]) for ax, h in _per_axis(halo, spatial, 0.0).items()}
    else:
        halo_px = {ax: int(h) for ax, h in _per_axis(halo, spatial, 0.0).items()}
        if any(float(h) != int(h) for h in _per_axis(halo, spatial, 0.0).values()):
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
