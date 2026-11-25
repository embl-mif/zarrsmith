"""ZarrManager.update_metadata(in_place=True) writes the updates into the store."""
from __future__ import annotations

import json
import warnings

import pytest
import zarr

import zarrsmith.microscope_manager as mm
from tests.test_ngff_validation import FORMATS, assert_valid_ome_zarr


def _ngff(group: zarr.Group) -> dict:
    attrs = group.attrs.asdict()
    return attrs["ome"] if isinstance(attrs.get("ome"), dict) else attrs


def _store_json(path) -> dict:
    """Every metadata file of the store, to check that nothing was written."""
    return {
        str(p.relative_to(path)): json.loads(p.read_text())
        for p in sorted(path.rglob("*"))
        if p.name in ("zarr.json", ".zattrs", ".zgroup", ".zarray")
    }


def _make_store(tmp_path, image_pyramid, metadata, ngff_version, zarr_format):
    path = tmp_path / "image.zarr"
    mm.ArrayManager(image_pyramid, metadata).to_zarr(
        path, ngff_version=ngff_version, zarr_format=zarr_format, progress=False)
    z = mm.ZarrManager(path, mode="a")
    z.create_empty_group("nuclei", metadata, is_label=True)
    z.create_empty_group("deconvolved", metadata)
    # A display window set by a viewer, which an update must not reset.
    root = zarr.open_group(str(path), mode="r+")
    if zarr_format == 3:
        ome = root.attrs["ome"]
        ome["omero"]["channels"][0]["window"] = {"start": 100, "end": 2000, "min": 0, "max": 65535}
        root.attrs["ome"] = ome
    else:
        omero = root.attrs["omero"]
        omero["channels"][0]["window"] = {"start": 100, "end": 2000, "min": 0, "max": 65535}
        root.attrs["omero"] = omero
    return path


@pytest.mark.parametrize("ngff_version,zarr_format", FORMATS)
def test_updates_reach_every_dataset_on_disk(tmp_path, image_pyramid, metadata,
                                             ngff_version, zarr_format):
    path = _make_store(tmp_path, image_pyramid, metadata, ngff_version, zarr_format)
    z = mm.ZarrManager(path, mode="r+")
    z.update_metadata({
        "channel_names": ["GFP", "RFP"],
        "channel_colors": ["cyan", "magenta"],
        "scales": [(1.0, 0.25, 0.25), (2.0, 0.5, 0.5), (4.0, 1.0, 1.0)],
        "translations": {"z": 5.0, "y": -3.0, "x": 7.0},
        "units": ("nanometer",) * 3,
        "time_increment": 30.0,
        "time_increment_unit": "minute",
        "extra_attributes": {"pipeline": "v2"},
    }, in_place=True)
    in_memory = dict(z.metadata)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mm.ZarrManager(path, mode="r")
    for dataset in (r.raw, r.groups["deconvolved"], r.labels["nuclei"]):
        md = dataset.metadata
        assert md["scales"] == [(1.0, 0.25, 0.25), (2.0, 0.5, 0.5), (4.0, 1.0, 1.0)]
        assert md["translations"][0] == {"z": 5.0, "y": -3.0, "x": 7.0}
        assert md["units"] == ("nanometer",) * 3
        assert md["time_increment"] == 30.0
        assert md["time_increment_unit"] == "minute"
        assert md["extra_attributes"] == {"pipeline": "v2"}
    for dataset in (r.raw, r.groups["deconvolved"]):
        assert dataset.metadata["channel_names"] == ["GFP", "RFP"]
        assert dataset.metadata["channel_colors"] == ["00FFFF", "FF00FF"]
    assert r.labels["nuclei"].metadata["data_type"] == "label"
    for key in ("channel_names", "channel_colors", "scales", "translations", "units"):
        assert r.metadata[key] == in_memory[key]

    # Only the updated entries were rewritten.
    root = _ngff(r.root)
    assert root["omero"]["channels"][0]["window"]["start"] == 100
    assert [d["path"] for d in root["multiscales"][0]["datasets"]] == ["0", "1", "2"]
    assert "image-label" in _ngff(r.root["labels/nuclei"])
    assert assert_valid_ome_zarr(path, ngff_version) == ["/", "/deconvolved", "/labels", "/labels/nuclei"]


def test_without_in_place_the_store_is_untouched(tmp_path, image_pyramid, metadata):
    path = _make_store(tmp_path, image_pyramid, metadata, "0.5", 3)
    before = _store_json(path)
    z = mm.ZarrManager(path, mode="r+")
    z.update_metadata({"channel_names": ["GFP", "RFP"]})
    assert z.metadata["channel_names"] == ["GFP", "RFP"]
    assert _store_json(path) == before


def test_extra_attributes_are_replaced(tmp_path, image_pyramid, metadata):
    path = _make_store(tmp_path, image_pyramid, metadata, "0.5", 3)
    root = zarr.open_group(str(path), mode="r+")
    root.attrs["t_range"] = [0, 5]
    z = mm.ZarrManager(path, mode="r+")
    z.update_metadata({"extra_attributes": {"pipeline": "v2"}}, include_groups=False,
                      include_labels=False, in_place=True)
    attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    assert attrs["pipeline"] == "v2" and "t_range" not in attrs
    z.update_metadata({"extra_attributes": None}, include_groups=False,
                      include_labels=False, in_place=True)
    assert set(zarr.open_group(str(path), mode="r").attrs.asdict()) == {"ome"}


@pytest.mark.parametrize("ngff_version,zarr_format", FORMATS)
def test_data_type_is_switched_on_disk(tmp_path, image_pyramid, metadata, ngff_version, zarr_format):
    path = _make_store(tmp_path, image_pyramid, metadata, ngff_version, zarr_format)
    z = mm.ZarrManager(path, mode="r+")
    z.update_metadata({"data_type": "label"}, include_labels=False, in_place=True)
    group = _ngff(zarr.open_group(str(path), mode="r")["deconvolved"])
    assert "image-label" in group and "omero" not in group
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert mm.ZarrManager(path).groups["deconvolved"].metadata["data_type"] == "label"

    z.update_metadata({"data_type": "intensity"}, include_labels=False, in_place=True)
    group = _ngff(zarr.open_group(str(path), mode="r")["deconvolved"])
    assert "image-label" not in group and len(group["omero"]["channels"]) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert mm.ZarrManager(path).groups["deconvolved"].metadata["data_type"] == "intensity"


def test_read_only_store_is_refused(tmp_path, image_pyramid, metadata):
    path = _make_store(tmp_path, image_pyramid, metadata, "0.5", 3)
    z = mm.ZarrManager(path, mode="r")
    with pytest.raises(PermissionError, match="r\\+"):
        z.update_metadata({"channel_names": ["GFP", "RFP"]}, in_place=True)
    assert z.metadata["channel_names"] != ["GFP", "RFP"]


def test_a_refused_dataset_leaves_store_and_manager_unchanged(tmp_path, image_pyramid, metadata):
    path = _make_store(tmp_path, image_pyramid, metadata, "0.5", 3)
    before = _store_json(path)
    z = mm.ZarrManager(path, mode="r+")
    scales = list(z.metadata["scales"])
    z._invalidate_zarr_data(z.labels["nuclei"])  # e.g. subset in memory
    with pytest.raises(ValueError, match="to_zarr"):
        z.update_metadata({"units": ("nanometer",) * 3, "scales": [(1.0, 1.0, 1.0)] * 3}, in_place=True)
    assert _store_json(path) == before
    assert z.metadata["scales"] == scales
    assert z.metadata["units"] == ("micrometer",) * 3


def test_ngff_06_store_is_refused(tmp_path):
    from tests.test_allocate_pyramid import _make_v06_store

    src = tmp_path / "v06.ome.zarr"
    _make_v06_store(src)
    z = mm.ZarrManager(str(src), mode="r+")
    with pytest.raises(ValueError, match="NGFF 0.6"):
        z.update_metadata({"units": ("nanometer",) * 3}, in_place=True)
