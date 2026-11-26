# tests/test_zarr_read_write.py
from __future__ import annotations

import pytest
import zarr

import zarrsmith.microscope_manager as mm


def _assert_metadata_equal_basic(m1, m2):
    assert m2["axes"] == m1["axes"]
    assert m2["dtype"] == m1["dtype"]
    assert m2["channel_names"] == m1["channel_names"]
    assert m2["channel_colors"] == m1["channel_colors"]
    assert tuple(m2["units"]) == tuple(m1["units"])
    assert pytest.approx(m2["time_increment"]) == m1["time_increment"]
    assert m2["time_increment_unit"] in ("s", "second")
    assert len(m2["size"]) == len(m1["size"])
    assert len(m2["scales"]) == len(m1["scales"])


def _assert_ngff_attrs(attrs, ngff_version):
    """The NGFF blocks sit at the root of the attributes in 0.4, under ``ome`` in 0.5."""
    if ngff_version == "0.4":
        assert "multiscales" in attrs and "omero" in attrs and "ome" not in attrs
    else:
        assert attrs["ome"]["version"] == ngff_version
        assert "multiscales" in attrs["ome"] and "omero" in attrs["ome"]


@pytest.mark.parametrize("ngff_version, zarr_format", [("0.4", 2), ("0.5", 3)])
def test_write_and_reopen(tmp_path, image_pyramid, metadata, ngff_version, zarr_format):
    out = tmp_path / f"test_v{ngff_version}.zarr"
    mm.ArrayManager(image_pyramid, metadata).to_zarr(
        str(out),
        ngff_version=ngff_version,
        zarr_format=zarr_format,
        compressor="blosc",
        compressor_level=1,
        overwrite=True,
    )
    # low-level structural check
    _assert_ngff_attrs(zarr.open_group(str(out), mode="r").attrs.asdict(), ngff_version)
    # high-level reopen
    d = mm.ZarrManager(str(out), mode="r")
    assert len(d.data) == 3
    _assert_metadata_equal_basic(metadata, d.metadata)
