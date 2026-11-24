import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
from dask import delayed
from tifffile import imread

from .microscope_manager import MicroscopeManager
from ..utils.axes import UNRECORDED_SCALE_KEY
from ..utils.metadata import build_metadata
from ..utils.ome import parse_ome_xml, xml_attr
from ..utils.units import normalize_unit

logger = logging.getLogger("zarrsmith")

#: Stage positions of a Viventis acquisition, in its ``Settings`` folder.
STAGE_POSITIONS_FILE = "STAGE_PositionSettings.json"


class ViventisManager(MicroscopeManager):
    """
    Reader for Viventis microscope datasets with OME-TIFF and companion .ome XML files.

    This class lazily loads data into a dask array and parses associated OME-XML metadata.
    """

    def __init__(self,
                 path: str,
                 chunks: Optional[Tuple[int, ...]] = None):
        """
        Initialize the ViventisManager.

        Parameters
        ----------
        path : str
            Path to the folder containing the Viventis dataset (including `.ome` and `.tif` files).
        chunks : Tuple[int, ...], optional
            Desired chunk shape for the output Dask array. Defaults to
            :attr:`DEFAULT_CHUNKS` ``(1, 1, 8, 4096, 4096)``.
        """
        super().__init__(path, chunks)
        self.read()

    @staticmethod
    def _plane_files(tiffdata: List[ET.Element]) -> Dict[Tuple[int, int], str]:
        """Map ``(t, c)`` to the TIFF file holding that plane stack."""
        plane_files = {}
        for entry in tiffdata:
            t = xml_attr(entry, "FirstT", int, what="<TiffData>")
            c = xml_attr(entry, "FirstC", int, what="<TiffData>")
            uuid = entry.find(".//{*}UUID")
            if uuid is None:
                raise ValueError(f"<TiffData> for t={t}, c={c} has no <UUID FileName=...>.")
            plane_files[(t, c)] = xml_attr(uuid, "FileName", what="<UUID>")
        return plane_files

    def _parse_metadata(self) -> Dict[str, Any]:
        """
        Parse the companion `.ome` XML metadata file.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing extracted metadata such as size, scales, units, channel info,
            time increment, and TIFF file mapping (``plane_files``).
        """
        companion = next(self.path.glob("*.ome"), None)
        if companion is None:
            raise FileNotFoundError(f"No companion '*.ome' file found in {self.path}.")

        px = parse_ome_xml(ET.parse(companion).getroot())
        return build_metadata(
            size=[(px.size_t, px.size_c, px.size_z, px.size_y, px.size_x)],
            scales=[px.scale_zyx],
            **{UNRECORDED_SCALE_KEY: px.unrecorded_axes},
            units=px.units_zyx,
            channels=px.channels,
            dtype=px.dtype,
            time_increment=px.time_increment,
            time_increment_unit=px.time_increment_unit,
            translations=self._stage_position(px.units_zyx),
            plane_files=self._plane_files(px.tiffdata),
        )

    def _settings_file(self) -> Optional[Path]:
        """``STAGE_PositionSettings.json`` of the acquisition: in ``<acquisition>/Settings``,
        next to the ``Position N_Settings M`` folders (or inside the dataset folder)."""
        for folder in (self.path.parent, self.path):
            candidate = folder / "Settings" / STAGE_POSITIONS_FILE
            if candidate.is_file():
                return candidate
        return None

    def _stage_position(self, units: Tuple[str, ...]) -> Optional[Dict[str, float]]:
        """Stage position of this position folder, ``{"z", "y", "x"}`` in µm, or ``None``.

        Viventis records the stage position of every position only in the acquisition's
        ``Settings/STAGE_PositionSettings.json`` (the OME-XML has none), as entries with
        ``Name`` (``"Position 3"``), ``PositionX``, ``PositionY`` and ``PositionZ`` in
        micrometres. The entry is matched to the folder name before ``_Settings``
        (``"Position 3_Settings 1"``). The values are used as given: whether they are the
        centre or the corner of the field, and the stage axis directions, are not stated
        by the file. Without the file or a matching entry the position is unknown.
        """
        settings = self._settings_file()
        if settings is None:
            return None
        if any(normalize_unit(u) != "micrometer" for u in units):
            logger.warning("Viventis voxel size is not in micrometers (%s): stage position not used.", units)
            return None
        name = re.split(r"_Settings", self.path.name, maxsplit=1)[0].strip()
        try:
            entries = json.loads(settings.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("Cannot read %s (%s): stage position not used.", settings, e)
            return None
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and str(entry.get("Name", "")).strip() == name:
                try:
                    position = {ax: float(entry[f"Position{ax.upper()}"]) for ax in "zyx"}
                except (KeyError, TypeError, ValueError):
                    logger.warning("Entry %r of %s has no PositionX/Y/Z: stage position not used.", name, settings)
                    return None
                logger.info("Stage position of %r from %s: %s", name, settings.name, position)
                return position
        logger.info("No entry %r in %s: stage position not used.", name, settings)
        return None

    def _build_dask_array(self) -> List[da.Array]:
        """
        Lazily construct a dask array for the image data using tifffile and delayed loading.

        Returns
        -------
        List[da.Array]
            A list containing a single Dask array representing the full dataset (level 0).
        """
        t, c, z, y, x = self.metadata["size"][0]
        filenames = self.metadata["plane_files"]
        dtype = self.metadata["dtype"]

        missing = [(ti, ci) for ti in range(t) for ci in range(c) if (ti, ci) not in filenames]
        if missing:
            raise ValueError(f"No TIFF file listed for (t, c) = {missing[:5]}{'...' if len(missing) > 5 else ''}.")

        lazy_imread = delayed(imread)
        stack = da.stack(
            [
                [
                    da.from_delayed(lazy_imread(str(self.path / filenames[(ti, ci)])), shape=(z, y, x), dtype=dtype)
                    for ci in range(c)
                ]
                for ti in range(t)
            ],
            axis=0,
        ).rechunk(self.chunks)
        return [stack]  # level 0 only
