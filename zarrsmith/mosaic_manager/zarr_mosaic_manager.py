"""OME-Zarr stores as the tiles of a mosaic, placed at their NGFF ``translation``."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import zarr

from ..utils.axes import UNRECORDED_SCALE_KEY, normalize_axes, spatial_axes_in_order
from ..utils.units import normalize_unit
from .mosaic import MosaicManager
from .tile import Tile

logger = logging.getLogger("zarrsmith")

#: Files that make a folder a Zarr group (v3, then v2).
_GROUP_FILES = ("zarr.json", ".zgroup", ".zattrs")


def is_zarr_store(path: str | Path) -> bool:
    """Whether ``path`` is a Zarr group (an OME-Zarr store), from its metadata files."""
    path = Path(path)
    return path.is_dir() and any((path / name).is_file() for name in _GROUP_FILES)


def zarr_stores(folder: str | Path) -> List[Path]:
    """The Zarr stores directly inside ``folder``, by name."""
    folder = Path(folder)
    return sorted((p for p in folder.iterdir() if is_zarr_store(p)), key=lambda p: p.name) if folder.is_dir() else []


def _same_folder(a: str | Path, b: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class ZarrTileArray:
    """Lazy, picklable ``(z, y, x)`` view of one timepoint and channel of an OME-Zarr level.

    The store is opened on first read in each process (the handle is not pickled), so the
    view can be sent to worker processes. A store without ``z`` gets a ``z`` axis of size 1.
    It behaves like a read-only array: ``shape``, ``dtype``, ``ndim`` and numpy-style slicing.
    """

    #: Tells ``fuse_tiles(executor="auto")`` that this data can be used in worker processes.
    picklable = True

    def __init__(self, store: str | Path, array_path: str, axes: str, t: Optional[int], c: Optional[int],
                 shape: Sequence[int], dtype):
        """``axes`` are the axes of the stored array; ``t`` and ``c`` the fixed indices of its
        ``t`` and ``c`` axes (``None`` without them); ``shape`` the ``(z, y, x)`` shape of the view."""
        self.store = str(store)
        self.array_path = str(array_path)
        self.axes = "".join(normalize_axes(axes))
        self.t, self.c = t, c
        self.shape: Tuple[int, int, int] = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = 3
        self._array = None

    def _open(self):
        if self._array is None:
            root = zarr.open_group(zarr.storage.LocalStore(self.store, read_only=True), mode="r")
            self._array = root[self.array_path]
        return self._array

    def __getitem__(self, key) -> np.ndarray:
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            i = next(i for i, k in enumerate(key) if k is Ellipsis)
            key = key[:i] + (slice(None),) * (3 - len(key) + 1) + key[i + 1:]
        kz, ky, kx = key + (slice(None),) * (3 - len(key))
        fixed = {"t": self.t, "c": self.c, "z": kz, "y": ky, "x": kx}
        out = np.asarray(self._open()[tuple(fixed[ax] for ax in self.axes)])
        if "z" not in self.axes:  # the z axis of size 1 added to a 2D store
            out = out[np.newaxis][kz]
        return out

    def __len__(self) -> int:
        return self.shape[0]

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_array"] = None  # reopened where it is read
        return state

    def __repr__(self) -> str:
        return (f"ZarrTileArray({self.store!r}, {self.array_path!r}, t={self.t}, c={self.c}, "
                f"shape={self.shape}, dtype={self.dtype})")


class RelabeledArray:
    """Read-only view of a label array whose ids are changed as they are read.

    Every voxel other than ``background`` first gets ``offset`` added, then the ids found
    in ``mapping`` are replaced by their value; other ids are kept. This is how the tiles
    of a segmentation run tile by tile are given one numbering before they are fused:
    ``offset`` makes the ids of each tile unique (e.g. ``tile_index * step``) and
    ``mapping`` merges the objects split between tiles. The result has ``dtype``
    (default ``uint32``); an id that does not fit raises ``ValueError``. It can be sent to
    worker processes when the wrapped array can (``picklable``).
    """

    def __init__(self, data, *, offset: int = 0, mapping: Optional[Dict[int, int]] = None,
                 background: int = 0, dtype="uint32"):
        self.data = data
        self.offset = int(offset)
        self.background = int(background)
        self.dtype = np.dtype(dtype)
        if self.dtype.kind not in "iu":
            raise ValueError(f"RelabeledArray needs an integer dtype, got {self.dtype}.")
        items = sorted((int(k), int(v)) for k, v in (mapping or {}).items())
        self._keys = np.array([k for k, _ in items], dtype=np.int64)
        self._values = np.array([v for _, v in items], dtype=np.int64)
        self.shape = tuple(int(s) for s in data.shape)
        self.ndim = len(self.shape)
        self.picklable = bool(getattr(data, "picklable", False))

    def __getitem__(self, key) -> np.ndarray:
        block = np.asarray(self.data[key]).astype(np.int64)
        fg = block != self.background
        block[fg] += self.offset
        if self._keys.size and block.size:
            ids, inverse = np.unique(block, return_inverse=True)
            pos = np.clip(np.searchsorted(self._keys, ids), 0, self._keys.size - 1)
            hit = (self._keys[pos] == ids) & (ids != self.background)
            block = np.where(hit, self._values[pos], ids)[inverse.reshape(block.shape)]
        info = np.iinfo(self.dtype)
        if block.size and (block.max() > info.max or block.min() < info.min):
            raise ValueError(f"Relabelled ids up to {int(block.max())} do not fit in {self.dtype}.")
        return block.astype(self.dtype)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"RelabeledArray({self.data!r}, offset={self.offset}, {self._keys.size} mapped ids)"


class ZarrMosaicManager(MosaicManager):
    """OME-Zarr stores (any microscope, converted with ZarrSmith or not) as one mosaic.

    Every OME-Zarr store directly inside the folder is a tile per timepoint and channel,
    placed at its NGFF ``translation`` (``metadata["translations"]``, the stage position
    of its voxel 0) with its level-0 voxel size. A folder that is itself a store gives the
    tiles of that store. :meth:`visualize` shows the stores side by side at their
    positions; stores that touch form one field and can be fused.

    * ``channel``: the channel name of the store (``ch<i>`` without names): stores with the
      same channel names are fused channel by channel;
    * ``timepoint``: the ``t`` index of the store;
    * ``attrs``: ``stack`` (the store name without ``.ome.zarr``), ``path`` and ``store``.

    Label stores (a segmentation, tile by tile) are read when the folder holds nothing
    else, or with ``data_type="label"``: their tiles are marked as label data
    (``attrs["data_type"]``), so a fusion writes a label image without blending them (see
    :func:`~zarrsmith.mosaic_manager.fusion.tile_data_type`). In a folder that mixes
    images and labels, the images are read and the labels left out, unless
    ``data_type="label"`` (``"intensity"`` always reads the images).

    Also left out, with a message: stores fused from the stores of the same folder (a
    previous fusion of this mosaic, found by its ``zarrsmith_sources`` attribute) and stores
    that cannot be read. A store without a translation is placed at the origin, with a
    warning. All stores must share the voxel size and units to be fused together.
    """

    def __init__(self, path: str | Path, data_type: Optional[str] = None):
        if data_type not in (None, "intensity", "label"):
            raise ValueError(f"data_type must be None, 'intensity' or 'label', got {data_type!r}.")
        self.data_type = data_type
        super().__init__(path)

    def _scan(self) -> List[Tile]:
        if not self.path.is_dir():
            raise FileNotFoundError(f"{self.path} is not a folder of OME-Zarr stores.")
        stores = [self.path] if is_zarr_store(self.path) else zarr_stores(self.path)
        found = {}
        for store in stores:
            result = self._store_tiles(store)
            if result is not None:
                found[store] = result
        kinds = {store: "label" if any(t.attrs.get("data_type") == "label" for t in ts) else "intensity"
                 for store, (ts, _) in found.items()}
        wanted = self.data_type or ("intensity" if "intensity" in kinds.values() else "label")
        if self.data_type is None and wanted == "label" and found:
            logger.info("Only label stores in %s: reading them as label tiles.", self.path.name)
        tiles: List[Tile] = []
        unplaced = []
        for store, (store_tiles, placed) in found.items():
            if kinds[store] != wanted:
                logger.info("Skipping %s: %s data (read with data_type=%r).", store.name, kinds[store], kinds[store])
                continue
            tiles.extend(store_tiles)
            if not placed:
                unplaced.append(store.name)
        if unplaced:
            logger.warning("%d store(s) have no position (NGFF translation) and are placed at the origin: %s",
                           len(unplaced), ", ".join(unplaced))
        voxels = {t.voxel_size for t in tiles}
        if len(voxels) > 1:
            logger.warning("The stores have different voxel sizes %s: they can be shown, not fused together.",
                           sorted(voxels))
        return tiles

    def _store_tiles(self, store: Path) -> Optional[Tuple[List[Tile], bool]]:
        """``(tiles, has a position)`` of one store, or ``None`` when it is left out."""
        from ..microscope_manager.zarr_manager import ZarrManager
        from .fusion import SOURCES_KEY

        try:
            zm = ZarrManager(str(store), mode="r")
            datasets = zm._get_multiscales(zm.root)[0]["datasets"]
        except Exception as e:  # noqa: BLE001 - one unreadable store must not stop the mosaic
            logger.warning("Skipping %s: %s", store.name, e)
            return None
        sources = zm.root.attrs.asdict().get(SOURCES_KEY) or []
        if store != self.path and sources and all(_same_folder(Path(s).parent, self.path) for s in sources):
            logger.info("Skipping %s: fused from the stores of this folder.", store.name)
            return None

        return store_tiles(store, zm=zm)


def store_tiles(store: str | Path, *, zm=None) -> Tuple[List[Tile], bool]:
    """``(tiles, has a position)`` of one OME-Zarr store: one tile per timepoint and channel.

    Each tile reads level 0 lazily (:class:`ZarrTileArray`) and sits at the store's NGFF
    ``translation`` with its level-0 voxel size; the second value is ``False`` when the
    store has no translation (the tiles are then placed at the origin). The tiles of a
    label store (e.g. a segmentation run tile by tile) are marked as label data
    (``attrs["data_type"] = "label"``), so :func:`~zarrsmith.mosaic_manager.fuse_tiles`
    writes a label image without blending them, if need be through
    :class:`RelabeledArray`. ``zm`` is the store, already opened as a ``ZarrManager``.
    """
    from ..microscope_manager.zarr_manager import ZarrManager

    store = Path(store)
    if zm is None:
        zm = ZarrManager(str(store), mode="r")
    datasets = zm._get_multiscales(zm.root)[0]["datasets"]
    md = zm.metadata
    axes = md["axes"]
    shape = dict(zip(axes, zm.data[0].shape))
    spatial = spatial_axes_in_order(axes)
    scale = dict(zip(spatial, md["scales"][0]))
    # axes without a recorded voxel size (a 1.0 placeholder), and z for 2D stores
    unrecorded = tuple(ax for ax in "zyx" if ax not in scale or ax in (md.get(UNRECORDED_SCALE_KEY) or ()))
    translations = md.get("translations")
    position = translations[0] if translations else {}
    units = [normalize_unit(u) for u in (md.get("units") or ()) if u]
    unit = units[-1] if units else "micrometer"
    names = list(md.get("channel_names") or [])
    n_c = shape.get("c", 1)
    names = names if len(names) == n_c else [f"ch{i}" for i in range(n_c)]
    stack = store.name[:-len(".ome.zarr")] if store.name.endswith(".ome.zarr") else store.stem

    tiles = []
    for t in range(shape.get("t", 1)):
        for c in range(n_c):
            data = ZarrTileArray(store, datasets[0]["path"], axes,
                                 t if "t" in axes else None, c if "c" in axes else None,
                                 (shape.get("z", 1), shape["y"], shape["x"]), zm.data[0].dtype)
            tiles.append(Tile(
                data=data,
                origin=tuple(float(position.get(ax, 0.0)) for ax in "zyx"),
                voxel_size=tuple(float(scale.get(ax, 1.0)) for ax in "zyx"),
                channel=names[c],
                timepoint=t,
                attrs={"stack": stack, "path": os.path.abspath(store), "store": store.name,
                       **({UNRECORDED_SCALE_KEY: unrecorded} if unrecorded else {}),
                       **({"data_type": "label"} if md.get("data_type") == "label" else {})},
                units=unit,
            ))
    return tiles, bool(translations)
