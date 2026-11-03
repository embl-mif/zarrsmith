from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.czi import czi_distance
from .czi_array import CziTileArray
from .mosaic import MosaicManager
from .tile import Tile

logger = logging.getLogger("zarrsmith")

_PIXEL_TYPES = {"gray8": np.uint8, "gray16": np.uint16, "gray32": np.uint32, "gray32float": np.float32}

_FOCUS = re.compile(r"<FocusPosition>([^<]*)<")


def scene_names(czi_metadata) -> Dict[int, str]:
    """``{scene index: name}`` of a CZI file (names may be missing)."""
    return {int(s.get("Index")): s.get("Name")
            for s in czi_metadata.findall(".//Dimensions/S/Scenes/Scene") if s.get("Index") is not None}


def channel_names(czi_metadata, indices) -> Dict[int, str]:
    """``{channel index: name}`` for ``indices``, unique so that channels are not fused together."""
    found = czi_metadata.findall(".//Information/Image/Dimensions/Channels/Channel")
    names = {c: (found[c].get("Name") or found[c].get("Id")) if c < len(found) else None for c in indices}
    names = {c: n or f"C{c}" for c, n in names.items()}
    counts: Dict[str, int] = {}
    for n in names.values():
        counts[n] = counts.get(n, 0) + 1
    return {c: n if counts[n] == 1 else f"{n} (C{c})" for c, n in names.items()}


def field_name(scene: int, name: Optional[str]) -> str:
    """``scene0_TR1``: the index keeps the acquisition order and spaces are not allowed
    in field names, which the command line and batch CSV separate with spaces."""
    return f"scene{scene}_{re.sub(r'[^A-Za-z0-9._-]+', '_', name)}" if name else f"scene{scene}"


class ZeissMosaicManager(MosaicManager):
    """Raw Zeiss tiles of one ``.czi`` file (a ZEN tile scan), before any stitching.

    Not to be confused with :class:`~zarrsmith.microscope_manager.ZeissManager`, which
    reads one scene as a single image; a file without tiles, such as the output of
    ZEN's *Stitching*, is read with it. Every tile (``M`` index) of every channel and
    timepoint is one tile, and its planes stay on disk until read
    (:class:`~zarrsmith.mosaic_manager.CziTileArray`).

    * ``channel``: the channel name in the metadata (``AF488-T2``);
    * ``field``: the scene: ``scene0_TR1``, ``scene1_P1``, ... with the tile regions and
      positions of the ZEN experiment as ``TR1`` and ``P1``;
    * ``timepoint``: the ``T`` index;
    * ``attrs``: ``stack`` (``scene0_TR1-m00``, the tile id without the channel),
      ``scene``, ``scene_name``, ``mosaic_index`` and ``path``.

    Tiles are placed where the file says: ``y`` and ``x`` from the pixel coordinates
    of the tile in the file (the stage position, rounded to a pixel), ``z`` from the
    ``FocusPosition`` of its first plane. They are not registered, unlike the images
    stitched by ZEN. Positions in the file change a little between timepoints:
    a fusion uses those of one reference timepoint, as ZEN uses one offset for all.
    """

    def __init__(self, path, scene_index: Optional[int] = None):
        """
        Parameters
        ----------
        path
            The ``.czi`` file.
        scene_index
            Load only this scene (default: all of them, one field per scene).
        """
        self.scene_index = None if scene_index is None else int(scene_index)
        super().__init__(path)

    def _scan(self) -> List[Tile]:
        from aicspylibczi import CziFile

        if not self.path.is_file():
            raise FileNotFoundError(f"{self.path} is not a file: {type(self).__name__} reads one .czi file.")
        with open(self.path, "rb") as f:
            czi = CziFile(f)
            if not czi.is_mosaic():
                raise ValueError(f"{self.path.name} has no tiles (no M dimension): "
                                 "read it with ZeissManager (microscope manager).")
            if czi.pixel_type not in _PIXEL_TYPES:
                raise ValueError(f"{self.path.name}: pixel type {czi.pixel_type} is not supported for mosaics.")
            dtype = _PIXEL_TYPES[czi.pixel_type]
            md = czi.meta
            stacks = self._stacks(czi)
            if self.scene_index is not None:
                scenes = sorted({key[0] for key in stacks})
                stacks = {key: s for key, s in stacks.items() if key[0] == self.scene_index}
                if not stacks:
                    raise ValueError(f"Invalid scene index {self.scene_index}: {self.path.name} has scenes {scenes}.")
            zs = [z for s in stacks.values() for z in s["boxes"]]
            first_z, last_z = min(zs), max(zs)
            focus = self._focus_positions(czi, first_z, min(key[3] for key in stacks))
        default_focus = next(iter(focus.values()), 0.0)
        names = channel_names(md, sorted({key[3] for key in stacks}))
        scenes = scene_names(md)
        voxel = (czi_distance(md, "Z", default=1.0), czi_distance(md, "Y"), czi_distance(md, "X"))

        tiles = []
        missing = 0
        for (scene, m, t, c), s in sorted(stacks.items(), key=lambda kv: tuple(-1 if v is None else v for v in kv[0])):
            planes = [z if z in s["boxes"] else None for z in range(first_z, last_z + 1)]
            missing += planes.count(None)
            box = next(iter(s["boxes"].values()))
            field = None if scene is None else field_name(scene, scenes.get(scene))
            tiles.append(Tile(
                data=CziTileArray(self.path, s["coords"], planes, (len(planes), box.h, box.w), dtype),
                origin=(focus.get((scene, m, t), default_focus), box.y * voxel[1], box.x * voxel[2]),
                voxel_size=voxel,
                channel=names[c],
                timepoint=t,
                field=field,
                attrs={"stack": f"{field}-m{m:02d}" if field else f"m{m:02d}", "scene": scene,
                       "scene_name": scenes.get(scene), "mosaic_index": m, "path": str(self.path)},
            ))
        if missing:
            logger.warning("%d plane(s) are missing in %s: they are read as zeros.", missing, self.path.name)
        return tiles

    @staticmethod
    def _stacks(czi) -> Dict[Tuple, dict]:
        """``{(scene, m, t, c): {"coords": ..., "boxes": {z: box}}}``, from the directory of the file."""
        stacks: Dict[Tuple, dict] = {}
        for info, box in czi.get_all_mosaic_tile_bounding_boxes().items():
            d = info.dimension_coordinates
            if d.get("H", 0) != 0:  # phases are not supported: the first one is used
                continue
            key = (d.get("S"), info.m_index, d.get("T", 0), d.get("C", 0))
            stack = stacks.setdefault(key, {"coords": {k: v for k, v in d.items() if k != "Z"}, "boxes": {}})
            stack["boxes"][d.get("Z", 0)] = box
        return stacks

    @staticmethod
    def _focus_positions(czi, z: int, c: int) -> Dict[Tuple, float]:
        """``{(scene, m, t): focus position (um)}`` of plane ``z`` of channel ``c`` (one read per tile)."""
        out = {}
        for d, xml in czi.read_subblock_metadata(Z=z, C=c):
            d = dict(d)
            found = _FOCUS.search(xml)
            if found and d.get("H", 0) == 0:
                out[(d.get("S"), d.get("M"), d.get("T", 0))] = float(found.group(1))
        return out
