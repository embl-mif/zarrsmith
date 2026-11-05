from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from ..utils.czi import CziBlock, read_block
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
    blocks
        Where every plane is stored (:func:`~zarrsmith.utils.czi.read_block_index`), to read
        it directly. A plane without one is read with aicspylibczi, which goes through the
        whole directory of the file: seconds per plane if it holds a million subblocks.
    """

    def __init__(self, path: str | Path, coords: Dict[str, int], planes: Sequence[Optional[int]], shape, dtype,
                 blocks: Optional[Sequence[Optional[CziBlock]]] = None):
        super().__init__(shape, dtype)
        self.path = str(path)
        self.coords = dict(coords)
        self.planes = list(planes)
        self.blocks = [None] * len(self.planes) if blocks is None else list(blocks)
        if not self.shape[0] == len(self.planes) == len(self.blocks):
            raise ValueError(f"shape {self.shape} does not match {len(self.planes)} plane(s) and "
                             f"{len(self.blocks)} block(s).")

    def _read(self, z: int) -> np.ndarray:
        if self.planes[z] is None:
            return np.zeros(self.shape[1:], self.dtype)
        if self.blocks[z] is not None:
            return read_block(self.path, self.blocks[z], self.shape[1:], self.dtype)
        from aicspylibczi import CziFile

        with open(self.path, "rb") as f:
            plane, _ = CziFile(f).read_image(Z=self.planes[z], **self.coords)
        return plane.reshape(self.shape[1:]).astype(self.dtype, copy=False)

    def __repr__(self) -> str:
        return f"CziTileArray({self.path!r}, {self.coords}, shape={self.shape}, dtype={self.dtype})"
