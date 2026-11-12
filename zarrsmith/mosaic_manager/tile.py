from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(eq=False)
class Tile:
    """One acquired tile of a mosaic: a lazy ``(z, y, x)`` array and where it sits.

    Placement is a translation, after optional axis flips: the data is first
    mirrored along the axes where ``flip`` is true (e.g. the mirror image seen by an
    opposing camera), then ``origin`` is the position (in ``units``) of voxel
    ``[0, 0, 0]`` of the flipped data in the common sample frame and ``voxel_size``
    the voxel spacing. All tiles of a mosaic must use the same units and voxel size.

    Parameters
    ----------
    data
        Anything with ``shape``, ``dtype`` and numpy-style slicing that returns
        array data: an ``h5py.Dataset``, a dask or a numpy array. It is only
        read when a fusion needs it.
    origin, voxel_size
        ``(z, y, x)`` position of the first voxel and voxel spacing.
    channel
        Channel (dye) name. Tiles with the same name are fused into one output
        channel.
    group
        Optional label of tiles that should be merged among themselves first,
        e.g. ``"L"``/``"R"`` illumination side. Used by ``blend="sigmoid"`` and ``"midline"``.
    timepoint
        Index of the timepoint the tile belongs to (``0`` for a single timepoint).
        Tiles of one position at different timepoints are matched by
        :func:`series_keys`.
    attrs
        Free-form, reader specific metadata (stack, grid position, ...). ``stack``
        names the position of the tile, the same at every timepoint.
    units
        Units of ``origin`` and ``voxel_size``.
    field
        Name of the field (connected group of touching tiles) the tile belongs to,
        e.g. ``"field0"``. Set by :class:`~zarrsmith.mosaic_manager.MosaicManager` when
        the mosaic is loaded (see :func:`~zarrsmith.mosaic_manager.fields.label_fields`),
        unless the reader gives one.
    camera
        Optional name of the camera (detection view) the tile was taken with, e.g.
        ``"left"``/``"right"`` for the opposing cameras of a MuVi SPIM. Tiles of
        several cameras are fused as separate views, blended along their detection
        axis (see :func:`~zarrsmith.mosaic_manager.fuse_tiles`); ``attrs["detection_direction"]``
        (``(z, y, x)``, in the frame of ``origin``) says which side each camera images best.
    flip
        ``(z, y, x)`` axes along which the data is mirrored before it is placed.
    """

    data: Any
    origin: Tuple[float, float, float]
    voxel_size: Tuple[float, float, float]
    channel: str
    group: Optional[str] = None
    timepoint: int = 0
    attrs: Dict[str, Any] = _dc_field(default_factory=dict)
    units: str = "micrometer"
    field: Optional[str] = None
    camera: Optional[str] = None
    flip: Tuple[bool, bool, bool] = (False, False, False)

    def __post_init__(self) -> None:
        self.flip = tuple(bool(f) for f in self.flip)
        if len(self.flip) != 3:
            raise ValueError("flip must have three (z, y, x) values.")
        if len(self.data.shape) != 3:
            raise ValueError(f"Tile data must be 3D (z, y, x), got shape {tuple(self.data.shape)}.")
        self.origin = tuple(float(v) + 0.0 for v in self.origin)  # + 0.0 turns -0.0 into 0.0
        self.timepoint = int(self.timepoint)
        self.voxel_size = tuple(float(v) for v in self.voxel_size)
        if len(self.origin) != 3 or len(self.voxel_size) != 3:
            raise ValueError("origin and voxel_size must have three (z, y, x) values.")
        if any(v <= 0 for v in self.voxel_size):
            raise ValueError("voxel_size must be positive.")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(s) for s in self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data.dtype)

    @property
    def extent(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """``(min, max)`` corners of the tile in sample coordinates."""
        end = tuple(o + n * v for o, n, v in zip(self.origin, self.shape, self.voxel_size))
        return self.origin, end


def series_keys(tiles: Sequence[Tile]) -> List[tuple]:
    """Key of every tile, shared by the tiles of one position at different timepoints.

    The key is ``(channel, group, camera, stack, rank)``, with ``stack`` from ``attrs``
    and ``rank`` the place of the tile among those that share the rest within its
    timepoint. Tiles without a ``stack`` are therefore matched by their order.
    """
    seen: Dict[tuple, int] = {}
    keys = []
    for t in tiles:
        base = (t.channel, t.group, t.camera, t.attrs.get("stack"))
        rank = seen[base + (t.timepoint,)] = seen.get(base + (t.timepoint,), -1) + 1
        keys.append(base + (rank,))
    return keys


def first_of_series(tiles: Sequence[Tile]) -> List[int]:
    """For every tile, the index of the earliest tile at the same position (see :func:`series_keys`)."""
    keys = series_keys(tiles)
    first: Dict[tuple, int] = {}
    for i in sorted(range(len(tiles)), key=lambda i: tiles[i].timepoint):
        first.setdefault(keys[i], i)
    return [first[k] for k in keys]
