from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm.auto import tqdm

from .h5_array import H5Array
from .mosaic import MosaicManager
from .tile import Tile

logger = logging.getLogger("zarrsmith")


class LuxendoMosaicManager(MosaicManager):
    """Raw Luxendo tiles (``*.lux.h5`` + ``*.json``) of one acquisition, before any fusion.

    Not to be confused with :class:`~zarrsmith.microscope_manager.LuxendoManager`, which
    reads images already fused by the Luxendo software. This manager scans a raw
    acquisition folder such as ``2026-10-14_103957`` (tiles live in
    ``raw/<stack folder>/Cam_*.lux.h5``) and exposes every tile lazily:

    * ``channel``: the dye, with the illumination side removed
      (``autofluo-488-L`` and ``autofluo-488-R`` both give ``autofluo-488``), so
      both halves of a sample end up in the same fused channel;
    * ``group``: the illumination side, ``"L"`` or ``"R"`` (used by ``blend="sigmoid"`` and ``"midline"``);
    * ``timepoint``: the JSON ``time_point``, or the order of the acquisitions of a
      stack when the JSON has none;
    * ``attrs``: ``stack``, ``stack_description``, ``grid`` (``(ix, iy)``),
      ``channel_description`` and ``path``.

    Tile positions come from ``affine_to_sample`` in the JSON (the tile is centred
    on the stage translation in x/y; z is the plane position along the stacking
    direction). A negative x or y scale in that affine (the mirror image of an
    opposing camera) sets ``Tile.flip`` on that axis, so the tile is mirrored back
    when it is fused. A last, rotation entry of the affine (the stage rotation) is
    not applied: tiles are placed in the stage frame, which is the same for all the
    tiles of one stack. If a stack was re-acquired (``*_R_<n>.lux.h5``) only the latest
    retake is used. Tiles are :class:`~zarrsmith.mosaic_manager.H5Array` objects, so no
    file stays open. Scanning reads only the JSON files (in parallel) and opens a
    single ``.h5`` file, which keeps loading fast on network drives.

    **MuVi SPIM** acquisitions (two opposing cameras, ``Cam_left``/``Cam_right``) also give:

    * ``camera``: the JSON ``camera``. The cameras are fused as two views blended
      along z, each one on the side it images sharply (``attrs["detection_direction"]``);
    * ``field``: one field per stack, from the stack name (``0-x00-y00`` and
      ``0-x01-y00`` are tiles of stack ``0``, i.e. ``field0``): MuVi experiments often
      hold several samples, each imaged as its own stack, possibly at another stage
      rotation, so fields are not guessed from which tiles touch.
    """

    #: Stack name -> stack of a MuVi tile: ``0-x01-y00`` -> ``0``.
    _GRID_SUFFIX = re.compile(r"-x\d+-y\d+$")

    @staticmethod
    def _retake(h5: Path) -> int:
        """``Cam_bottom_00000.lux.h5`` -> 0, ``Cam_bottom_00000_R_1.lux.h5`` -> 1."""
        m = re.search(r"_R_(\d+)\.lux\.h5$", h5.name)
        return int(m.group(1)) if m else 0

    def _scan(self) -> List[Tile]:
        # Over a network share every listing, stat and open is a slow round trip:
        # list only raw/<stack>/ (one listing per folder, in parallel) instead of
        # walking the whole acquisition, and open a single .h5 file (for the dtype).
        root = self.path / "raw" if (self.path / "raw").is_dir() else self.path
        folders = [root] + sorted(e.path for e in os.scandir(root) if e.is_dir())
        with ThreadPoolExecutor(max_workers=self.scan_workers) as pool:
            found = list(tqdm(pool.map(self._acquisitions, folders), total=len(folders),
                              desc="scanning tiles", unit="folder", disable=not self.progress))
        if not any(found):  # other layouts: search the whole tree
            found = [self._acquisitions(f) for f in sorted({h.parent for h in self.path.rglob("*.lux.h5")})]
        # Several acquisitions of one stack (one per timepoint): their rank is the
        # fallback timepoint when the JSON has no time_point.
        files = [h5 for acquisitions in found for h5 in acquisitions]
        rank = {h5: i for acquisitions in found for i, h5 in enumerate(acquisitions)}
        if not files:
            return []

        with ThreadPoolExecutor(max_workers=self.scan_workers) as pool:
            infos = list(tqdm(pool.map(self._read_info, files), total=len(files),
                              desc="reading tile metadata", unit="tile", disable=not self.progress))

        # All tiles of an acquisition share the dtype: read it (and check the shape) once.
        first = H5Array(files[0])
        expected = self._shape(infos[0])
        if first.shape != expected:
            raise ValueError(f"{files[0]}: data shape {first.shape} does not match the JSON image size {expected}.")
        tiles = [self._tile_from(h5, info, first.dtype, default_timepoint=rank[h5]) for h5, info in zip(files, infos)]
        if any(t.attrs.get("microscope") == "MuVi-SPIM" for t in tiles):
            for t in tiles:  # one field per stack (sample), see the class docstring
                stack = str(t.attrs.get("stack") or "")
                t.field = f"field{self._GRID_SUFFIX.sub('', stack) or stack}"
        return tiles

    def __init__(self, path, scan_workers: int = 16, progress: bool = True):
        """
        Parameters
        ----------
        path
            Acquisition folder (containing ``raw/``), or the ``raw`` folder itself.
        scan_workers
            Folders listed / JSON files read in parallel while scanning (network latency).
        progress
            Show progress bars while scanning.
        """
        self.scan_workers = max(1, int(scan_workers))
        self.progress = progress
        super().__init__(path)

    @classmethod
    def _acquisitions(cls, folder) -> List[Path]:
        """The ``*.lux.h5`` acquisitions of a stack folder (those with a ``.json`` sibling).

        Files sharing a base name are retakes (``*_R_<n>``) of one acquisition and
        only the latest is kept. Different base names are separate acquisitions of
        the same stack (e.g. timepoints); they are returned in name order.
        """
        names = set(os.listdir(folder))
        h5s = [n for n in names if n.endswith(".lux.h5") and n[: -len(".lux.h5")] + ".json" in names]
        by_base = {}  # e.g. main_raw.lux.h5 (no .json, no image data) never gets here
        for n in h5s:
            by_base.setdefault(re.sub(r"(_R_\d+)?\.lux\.h5$", "", n), []).append(n)
        out = []
        for base in sorted(by_base):
            files = sorted(by_base[base], key=lambda n: cls._retake(Path(n)))
            for skipped in files[:-1]:
                logger.info("Skipping %s (superseded by retake %s)", skipped, files[-1])
            out.append(Path(folder) / files[-1])
        return out

    @staticmethod
    def _read_info(h5: Path) -> dict:
        return json.loads(h5.with_name(h5.name.replace(".lux.h5", ".json")).read_text())["processingInformation"]

    @staticmethod
    def _shape(info: dict) -> tuple:
        size = info["image_size_vx"]
        return (size["depth"], size["height"], size["width"])

    @classmethod
    def _tile_from(cls, h5: Path, info: Optional[dict] = None, dtype=None, default_timepoint: int = 0) -> Tile:
        """Tile of one ``.lux.h5`` file; the file is only opened when ``dtype`` is not given.

        The timepoint is the JSON ``time_point``, or ``default_timepoint`` without one.
        """
        info = cls._read_info(h5) if info is None else info
        size, vox = info["image_size_vx"], info["voxel_size_um"]
        affines = info["affine_to_sample"]
        aff = affines[1]
        matrix = np.array(aff["matrix"], float)
        if not np.allclose(matrix, np.diag(np.diag(matrix))):
            raise ValueError(f"{h5}: the voxel-to-stage affine {aff['matrix']} is not diagonal (sheared or rotated "
                             "tiles are not supported).")
        tx, ty, tz = aff["translation"]
        sz = matrix[2][2]  # um per plane, signed (stage z direction)
        desc = info.get("channel_description") or info["channel"]
        side = re.search(r"[-_]([LR])$", desc)

        data = H5Array(h5, shape=cls._shape(info), dtype=dtype)

        attrs = {}
        sources = " ".join(str(s) for s in info.get("sources") or ())
        acquisition = (info.get("acquisition") or [{}])[0]
        if "MuVi" in sources or acquisition.get("microscope_type") == "MuVi-SPIM":
            attrs["microscope"] = "MuVi-SPIM"
        directions = info.get("detection_directions")
        if directions:
            d = np.array(directions[0], float)  # (x, y, z) in the sample frame
            if len(affines) > 2:  # back to the stage frame (before the stage rotation)
                d = np.array(affines[2]["matrix"], float).T @ d
            attrs["detection_direction"] = (float(np.sign(sz) * d[2]), float(d[1]), float(d[0]))
        rotation = [s for s in acquisition.get("stage_positions") or () if s.get("type") == "rotation"]
        if rotation:
            attrs["rotation_deg"] = float(rotation[0].get("start_deg", 0.0))

        grid = re.search(r"x(\d+)-y(\d+)", info["stack"])
        return Tile(
            data=data,
            origin=(
                tz / sz * vox["depth"],  # position of plane 0 along the stacking direction
                ty - (size["height"] - 1) / 2 * vox["height"],
                tx - (size["width"] - 1) / 2 * vox["width"],
            ),
            voxel_size=(vox["depth"], vox["height"], vox["width"]),
            channel=re.sub(r"[-_][LR]$", "", desc),
            group=side.group(1) if side else None,
            timepoint=int(info["time_point"]) if info.get("time_point") not in (None, "") else default_timepoint,
            attrs={
                "stack": info["stack"],
                "stack_description": info.get("stack_description"),
                "grid": (int(grid.group(1)), int(grid.group(2))) if grid else None,
                "channel_description": desc,
                "path": str(h5),
                **attrs,
            },
            camera=info.get("camera"),
            # mirrored along x/y (negative scale): flip the data back (the origin formula holds
            # either way, the tile being centred on the translation); z keeps the stacking direction
            flip=(False, bool(matrix[1][1] < 0), bool(matrix[0][0] < 0)),
        )
