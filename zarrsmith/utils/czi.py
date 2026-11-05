"""Helpers for Zeiss CZI files, shared by the Zeiss readers: metadata, and direct access to planes."""
from __future__ import annotations

import struct
import warnings
from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from numcodecs import Zstd

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


# Direct access to planes. aicspylibczi goes through the whole subblock directory of the
# file twice for every plane (to open it, then to find the plane), which takes seconds
# when a file holds a million subblocks, while the plane itself takes about a millisecond.
# So the directory is read once, with the position of every plane, and planes are then
# read from their position.

#: Size of the header of every segment of the file (id, allocated size, used size).
_SEGMENT_HEADER = 32
#: Part of the file header before ``SubBlockDirectoryPosition``, after the segment header.
_DIRECTORY_POSITION = 52
#: Fields of the directory segment before its entries.
_DIRECTORY_HEADER = 128
#: Smallest size of the fixed part of a subblock segment (sizes, directory entry, padding).
_SUBBLOCK_HEADER = 256
#: Sizes at the start of a subblock segment: metadata, attachment, data.
_SUBBLOCK_SIZES = struct.Struct("<iiq")
#: Directory entry: schema, pixel type, file position, file part, compression, pyramid
#: type, spare bytes, number of dimensions.
_ENTRY = struct.Struct("<2siqiiB5xi")
#: Dimension of a directory entry: name, start, size, start coordinate, stored size.
_DIMENSION = struct.Struct("<4siifi")

_UNCOMPRESSED, _ZSTD0, _ZSTD1 = 0, 5, 6
#: Pixel types (as in the directory) that hold one value per pixel.
_PIXEL_DTYPES = {0: np.dtype(np.uint8), 1: np.dtype(np.uint16), 2: np.dtype(np.float32), 12: np.dtype(np.uint32)}

_zstd = Zstd()


class CziBlock(NamedTuple):
    """Where one plane is stored in a CZI file (see :func:`read_block`)."""

    position: int  # of its subblock segment
    data_offset: int  # of its metadata, which comes right before its pixels
    compression: int


def block_key(coords: Mapping[str, int]) -> Tuple[Tuple[str, int], ...]:
    """Key of a plane in :func:`read_block_index`, from its coordinates such as
    ``{"S": 0, "M": 3, "T": 1, "C": 2, "Z": 0}`` (a file without ``Z`` has one plane)."""
    return tuple(sorted({"Z": 0, **coords}.items()))


def read_block_index(path, dtype) -> Dict[Tuple[Tuple[str, int], ...], CziBlock]:
    """``{block_key(coordinates): CziBlock}`` of the planes :func:`read_block` can read.

    Only the directory of the file is read. Planes that have to go through aicspylibczi
    are left out: other compressions (e.g. JPEG XR), pixel types other than ``dtype``,
    planes stored in another part of a file split in several, and planes that share their
    coordinates. Pyramid levels stored in the file are left out too.
    """
    dtype = np.dtype(dtype)
    with open(path, "rb") as f:
        header = f.read(_SEGMENT_HEADER + _DIRECTORY_POSITION + 8)
        directory = struct.unpack_from("<q", header, _SEGMENT_HEADER + _DIRECTORY_POSITION)[0]
        f.seek(directory)
        segment = f.read(_SEGMENT_HEADER + _DIRECTORY_HEADER)
        if not segment.startswith(b"ZISRAWDIRECTORY"):
            return {}
        allocated = struct.unpack_from("<q", segment, 16)[0]
        count = struct.unpack_from("<i", segment, _SEGMENT_HEADER)[0]
        entries = f.read(allocated - _DIRECTORY_HEADER)

    index: Dict[Tuple[Tuple[str, int], ...], CziBlock] = {}
    shared = set()
    start = 0
    for _ in range(count):
        schema, pixel, position, part, compression, pyramid, n_dims = _ENTRY.unpack_from(entries, start)
        if schema != b"DV":
            return {}
        stop = start + _ENTRY.size + n_dims * _DIMENSION.size
        coords = {}
        whole = True  # stored at full size: not a pyramid level
        for name, first, size, _, stored in _DIMENSION.iter_unpack(entries[start + _ENTRY.size : stop]):
            if name in (b"X\0\0\0", b"Y\0\0\0"):
                whole = whole and size == stored
            else:
                coords[name.rstrip(b"\0").decode()] = first
        entry_size = stop - start
        start = stop
        if (not whole or pyramid != 0 or part != 0 or compression not in (_UNCOMPRESSED, _ZSTD0, _ZSTD1)
                or _PIXEL_DTYPES.get(pixel) != dtype):
            continue
        key = block_key(coords)
        if key in index:
            shared.add(key)
        data_offset = _SEGMENT_HEADER + max(_SUBBLOCK_HEADER, 16 + entry_size)
        index[key] = CziBlock(position, position + data_offset, compression)
    for key in shared:
        del index[key]
    return index


def read_block(path, block: CziBlock, shape: Sequence[int], dtype) -> np.ndarray:
    """Read one plane of shape ``(y, x)`` from ``block`` of :func:`read_block_index`."""
    with open(path, "rb") as f:
        f.seek(block.position + _SEGMENT_HEADER)
        metadata, _, size = _SUBBLOCK_SIZES.unpack(f.read(_SUBBLOCK_SIZES.size))
        f.seek(block.data_offset + metadata)
        raw = f.read(size)
    if len(raw) != size:
        raise ValueError(f"{path}: the plane at byte {block.position} is cut short.")

    hi_lo = False
    if block.compression == _ZSTD1:  # a small header: its size, then (type 1, flags) if there is more
        skip = raw[0]
        if skip not in (1, 3):
            raise ValueError(f"{path}: the plane at byte {block.position} has an unknown zstd header.")
        hi_lo = skip == 3 and raw[1] == 1 and bool(raw[2] & 1)
        raw = _zstd.decode(raw[skip:])
    elif block.compression == _ZSTD0:
        raw = _zstd.decode(raw)

    dtype = np.dtype(dtype).newbyteorder("<")
    pixels = int(np.prod(shape))
    if len(raw) != pixels * dtype.itemsize or (hi_lo and dtype.itemsize != 2):
        raise ValueError(f"{path}: the plane at byte {block.position} does not hold {tuple(shape)} pixels of "
                         f"{dtype.name}.")
    if hi_lo:  # all the low bytes, then all the high bytes
        values = np.frombuffer(raw, np.uint8)
        return (values[:pixels] | (values[pixels:].astype(np.uint16) << 8)).reshape(shape)
    return np.frombuffer(raw, dtype).reshape(shape).copy()  # the buffer is read-only
