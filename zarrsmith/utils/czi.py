"""Helpers for the metadata of Zeiss CZI files, shared by the Zeiss readers."""
from __future__ import annotations

import warnings
from typing import Optional

from .units import to_micrometers


def czi_distance(czi_metadata, axis: str, default: Optional[float] = None) -> float:
    """Voxel size along ``axis`` in micrometers (CZI stores meters)."""
    node = czi_metadata.find(f"./Metadata/Scaling/Items/Distance[@Id='{axis}']/Value")
    if node is None or not node.text:
        if default is None:
            raise ValueError(f"CZI metadata has no voxel size for axis {axis}.")
        warnings.warn(f"CZI metadata has no voxel size for axis {axis}; assuming {default}.", stacklevel=3)
        return default
    (um,), _ = to_micrometers((float(node.text),), ("meter",))
    return um
