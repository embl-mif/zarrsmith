from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .tiff_array import PlaneStackArray


class CziTileArray(PlaneStackArray):
    """Lazy, picklable ``(z, y, x)`` stack of one tile of a CZI file.

    Like :class:`~zarrsmith.mosaic_manager.H5Array`, it holds no open file, so it can be
    sent to worker processes and needs no ``close()``. CZI planes are stored whole, so
    slicing reads only the planes it needs, each one in full.

    Parameters
    ----------
    path
        The ``.czi`` file.
    coords
        Coordinates of the tile in the file, without ``Z``, e.g. ``{"S": 0, "T": 1, "C": 2, "M": 1}``.
    planes
        ``Z`` index in the file of every plane, in order. ``None`` marks a missing plane, read as zeros.
    shape, dtype
        ``(z, y, x)`` shape and data type, from the metadata (no plane is read).
    """

    def __init__(self, path: str | Path, coords: Dict[str, int], planes: Sequence[Optional[int]], shape, dtype):
        super().__init__(shape, dtype)
        self.path = str(path)
        self.coords = dict(coords)
        self.planes = list(planes)
        if self.shape[0] != len(self.planes):
            raise ValueError(f"shape {self.shape} does not match {len(self.planes)} plane(s).")

    def _read(self, z: int) -> np.ndarray:
        if self.planes[z] is None:
            return np.zeros(self.shape[1:], self.dtype)
        from aicspylibczi import CziFile

        with open(self.path, "rb") as f:
            plane, _ = CziFile(f).read_image(Z=self.planes[z], **self.coords)
        return plane.reshape(self.shape[1:]).astype(self.dtype, copy=False)

    def __repr__(self) -> str:
        return f"CziTileArray({self.path!r}, {self.coords}, shape={self.shape}, dtype={self.dtype})"
