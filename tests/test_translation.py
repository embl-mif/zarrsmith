# tests/test_translation.py
"""Physical position (NGFF ``translation``) of datasets: written, read back, kept by
the dataset operations, read from the microscope files and set by mosaic fusion."""
from __future__ import annotations

import json
import logging
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import zarr

import zarrsmith.microscope_manager as mm
from zarrsmith.mosaic_manager import MosaicManager, OperaMosaicManager, Tile, fuse_tiles, fusion
from zarrsmith.mosaic_manager.opera_mosaic_manager import stitched_position
from zarrsmith.utils.translation import normalize_translations

from tests.test_evident_manager import FakeScene, FakeSlide, _raw_metadata
from tests.test_opera_mosaic import write_export

POSITION = {"z": 10.0, "y": 1200.5, "x": -340.0}
FORMATS = [("0.4", 2), ("0.5", 3)]


def _transforms(path):
    attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    multiscales = (attrs.get("ome") or attrs)["multiscales"][0]
    return [d["coordinateTransformations"] for d in multiscales["datasets"]]


def _image(translations=POSITION, shape=(2, 1, 8, 32, 32)):
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    metadata = {"axes": "tczyx", "scales": [(2.0, 0.5, 0.5)]}
    if translations is not None:
        metadata["translations"] = translations
    return mm.ArrayManager(data, metadata)


# ---------------------------------------------------------------------------
# metadata model
# ---------------------------------------------------------------------------

def test_normalize_translations_forms():
    assert normalize_translations(None, "tczyx") is None
    assert normalize_translations({"y": 1, "x": 2}, "czyx", 2) == [{"z": 0.0, "y": 1.0, "x": 2.0}] * 2
    assert normalize_translations((3, 1, 2), "zyx") == [{"z": 3.0, "y": 1.0, "x": 2.0}]
    per_level = [{"y": 1.0, "x": 2.0}, {"y": 1.0, "x": 2.0}]
    assert normalize_translations(per_level, "yx", 2) == per_level
    assert normalize_translations([{"y": 1.0, "x": 2.0}], "yx", 3) == [{"y": 1.0, "x": 2.0}] * 3
    with pytest.raises(ValueError, match="not spatial axes"):
        normalize_translations({"c": 1.0}, "czyx")
    with pytest.raises(ValueError, match="does not match"):
        normalize_translations((1.0, 2.0), "zyx")
    with pytest.raises(ValueError, match="one per pyramid level"):
        normalize_translations([{"x": 1.0}, {"x": 2.0}], "yx", 3)
    with pytest.raises(TypeError):
        normalize_translations({"x": "far"}, "yx")


def test_array_manager_normalizes_and_update_metadata():
    image = _image(translations=(10, 1200.5, -340))
    assert image.metadata["translations"] == [POSITION]
    image.update_metadata({"translations": {"x": 5.0}})
    assert image.metadata["translations"] == [{"z": 0.0, "y": 0.0, "x": 5.0}]
    image.update_metadata({"translations": None})
    assert image.metadata["translations"] is None
    with pytest.raises(ValueError, match="Invalid 'translations'"):
        image.update_metadata({"translations": {"t": 1.0}})


# ---------------------------------------------------------------------------
# NGFF write / read
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ngff_version,zarr_format", FORMATS)
def test_translation_round_trip(tmp_path, ngff_version, zarr_format):
    image = _image()
    image.build_pyramid(num_levels=3)
    assert image.metadata["translations"] == [POSITION] * 3
    path = tmp_path / "image.ome.zarr"
    image.to_zarr(str(path), ngff_version=ngff_version, zarr_format=zarr_format)

    for transforms in _transforms(path):  # the singleton c is dropped: t, z, y, x
        assert transforms[0]["type"] == "scale"
        assert transforms[1] == {"type": "translation", "translation": [0.0, 10.0, 1200.5, -340.0]}
    assert mm.ZarrManager(str(path)).metadata["translations"] == [POSITION] * 3

    from tests.test_ngff_validation import assert_valid_ome_zarr
    assert_valid_ome_zarr(path, ngff_version)


def test_no_translation_keeps_scale_only_metadata(tmp_path):
    path = tmp_path / "plain.ome.zarr"
    _image(translations=None).to_zarr(str(path))
    assert all(len(t) == 1 and t[0]["type"] == "scale" for t in _transforms(path))
    assert mm.ZarrManager(str(path)).metadata["translations"] is None

    zero = tmp_path / "zero.ome.zarr"
    _image(translations={"z": 0, "y": 0, "x": 0}).to_zarr(str(zero))
    assert all(len(t) == 1 for t in _transforms(zero))


def test_reader_applies_the_multiscale_transform(tmp_path):
    """A coordinateTransformations on the multiscale applies after the dataset's own."""
    path = tmp_path / "image.ome.zarr"
    _image(translations=None).to_zarr(str(path), drop_singleton=False)
    group = zarr.open_group(str(path), mode="r+")
    attrs = group.attrs.asdict()
    ome = attrs.get("ome") or attrs
    ome["multiscales"][0]["datasets"][0]["coordinateTransformations"].append(
        {"type": "translation", "translation": [0.0, 0.0, 1.0, 2.0, 3.0]})
    ome["multiscales"][0]["coordinateTransformations"] = [
        {"type": "scale", "scale": [5.0, 1.0, 1.0, 2.0, 2.0]},
        {"type": "translation", "translation": [0.0, 0.0, 100.0, 0.0, -1.0]},
    ]
    group.attrs.update(attrs)
    md = mm.ZarrManager(str(path)).metadata
    assert md["scales"][0] == (2.0, 1.0, 1.0)
    assert md["time_increment"] == 5.0
    assert md["translations"][0] == {"z": 101.0, "y": 4.0, "x": 5.0}


# ---------------------------------------------------------------------------
# dataset operations
# ---------------------------------------------------------------------------

def test_subset_moves_the_origin(tmp_path):
    image = _image()
    image.subset_dataset(Z=slice(2, 6), Y=slice(10, 40), X=[4, 5, 6], rebuild_pyramid=False)
    assert image.metadata["translations"] == [{"z": 14.0, "y": 1205.5, "x": -338.0}]
    path = tmp_path / "subset.ome.zarr"
    image.to_zarr(str(path))
    assert mm.ZarrManager(str(path)).metadata["translations"][0] == {"z": 14.0, "y": 1205.5, "x": -338.0}


def test_subset_selecting_nothing_is_refused():
    image = _image()  # 2 timepoints
    with pytest.raises(ValueError, match=r"selects nothing on axis 't'.*2 element"):
        image.subset_dataset(T=slice(30, 100, 3))
    assert image.metadata["size"][0][0] == 2  # unchanged


def test_dropped_singleton_axis_drops_its_position(tmp_path):
    path = tmp_path / "flat.ome.zarr"
    _image(shape=(1, 1, 1, 16, 16)).to_zarr(str(path))
    md = mm.ZarrManager(str(path)).metadata
    assert md["axes"] == "yx"
    assert md["translations"] == [{"y": 1200.5, "x": -340.0}]


def test_resample_keeps_the_position(tmp_path):
    image = _image()
    image.resample_dataset({"x": 1.0})
    assert image.metadata["translations"][0] == POSITION


def test_layer_geometry_translate_is_opt_in():
    from zarrsmith.utils.visualize import layer_geometry

    md = _image().metadata
    assert "translate" not in layer_geometry(md, "tczyx", drop_channel=True)
    geometry = layer_geometry(md, "tczyx", drop_channel=True, translate=True)
    assert geometry["translate"] == (0.0, 10.0, 1200.5, -340.0)
    assert "translate" not in layer_geometry(_image(translations=None).metadata, "tczyx",
                                             drop_channel=True, translate=True)


# ---------------------------------------------------------------------------
# microscope readers
# ---------------------------------------------------------------------------

def _viventis(tmp_path, folder="Position 3_Settings 1"):
    manager = object.__new__(mm.ViventisManager)  # only the position lookup is tested
    manager.path = tmp_path / folder
    manager.path.mkdir()
    return manager


def test_viventis_stage_position(tmp_path):
    manager = _viventis(tmp_path)
    um = ("micrometer",) * 3
    assert manager._stage_position(um) is None  # no Settings folder: no position
    settings = tmp_path / "Settings"
    settings.mkdir()
    (settings / "STAGE_PositionSettings.json").write_text(json.dumps([
        {"PositionX": 1.0, "PositionY": 2.0, "PositionZ": 3.0, "Name": "Position 1"},
        {"PositionX": 8988.465, "PositionY": 198.508, "PositionZ": -1049.039, "Name": "Position 3"},
    ]))
    assert manager._stage_position(um) == {"z": -1049.039, "y": 198.508, "x": 8988.465}
    assert _viventis(tmp_path, "Position 7_Settings 1")._stage_position(um) is None
    assert manager._stage_position(("nanometer",) * 3) is None


def test_luxendo_view_translation():
    xml = """<SpimData><ViewRegistrations>
      <ViewRegistration timepoint="0" setup="1"><ViewTransform type="affine">
        <affine>1 0 0 99 0 1 0 99 0 0 1 99</affine></ViewTransform></ViewRegistration>
      <ViewRegistration timepoint="0" setup="0">
        <ViewTransform type="affine"><affine>1 0 0 10 0 1 0 20 0 0 1 30</affine></ViewTransform>
        <ViewTransform type="affine"><affine>0.65 0 0 -2117.18 0 0.65 0 -62197.27 0 0 5 8226.77</affine></ViewTransform>
      </ViewRegistration></ViewRegistrations></SpimData>"""
    position = mm.LuxendoManager._view_translation(ET.fromstring(xml))
    assert position == pytest.approx({"z": 8256.77, "y": -62177.27, "x": -2107.18})
    assert mm.LuxendoManager._view_translation(ET.fromstring("<SpimData/>")) is None


def test_luxendo_rotation_keeps_only_the_translation(caplog):
    xml = ("<SpimData><ViewRegistrations><ViewRegistration timepoint='0' setup='0'><ViewTransform>"
           "<affine>0 -1 0 5 1 0 0 6 0 0 1 7</affine></ViewTransform></ViewRegistration>"
           "</ViewRegistrations></SpimData>")
    with caplog.at_level(logging.WARNING, logger="zarrsmith"):
        assert mm.LuxendoManager._view_translation(ET.fromstring(xml)) == {"z": 7.0, "y": 6.0, "x": 5.0}
    assert "rotates" in caplog.text


def _vsi(monkeypatch, tmp_path, rwc_origin, z_positions):
    data = np.zeros((1, 1, 3, 16, 24), np.uint16)
    raw = json.loads(_raw_metadata("C405", ["C405"], [(0, 0, 255)], z_positions=z_positions))
    stack = raw["value"][0]["value"][0]["value"][0]["value"]
    if rwc_origin is not None:
        stack += [{"name": "RWC frame origin", "tag": 2018, "value": rwc_origin},
                  {"name": "RWC frame unit", "tag": 2020, "value": "10^-6m^1"}]
    slide = FakeSlide([FakeScene("C405", data, ["C405"])], json.dumps(raw))
    module = types.ModuleType("slideio")
    module.open_slide = lambda path, driver: slide
    monkeypatch.setitem(sys.modules, "slideio", module)
    path = tmp_path / "sample.vsi"
    path.write_bytes(b"")
    return mm.EvidentManager(path)


def test_evident_stage_position(monkeypatch, tmp_path):
    vsi = _vsi(monkeypatch, tmp_path, "(-100962.159175, -65812.964763)", [9061.88, 9062.84, 9063.80])
    assert vsi.metadata["translations"][0] == {"z": 9061.88, "y": -65812.964763, "x": -100962.159175}
    # single-tile acquisitions record no stage z: x/y only
    flat = _vsi(monkeypatch, tmp_path, "(-101598.502203, -66401.71825)", None)
    assert flat.metadata["translations"][0] == {"z": 0.0, "y": -66401.71825, "x": -101598.502203}
    assert _vsi(monkeypatch, tmp_path, None, None).metadata["translations"] is None


def test_zeiss_scene_position(monkeypatch, tmp_path):
    from zarrsmith.utils.czi import scene_position

    box = types.SimpleNamespace
    xml = "<METADATA><Tags><FocusPosition>{}</FocusPosition></Tags></METADATA>"

    class FakeCzi:
        def __init__(self, f):
            pass

        def get_all_scene_bounding_boxes(self):
            return {0: box(x=-44089, y=2869, w=1943, h=1024), 1: box(x=-40772, y=5793, w=1024, h=1024)}

        def get_dims_shape(self):
            return [{"Z": (0, 2), "C": (0, 4), "T": (0, 1), "S": (0, 1)},
                    {"Z": (1, 2), "C": (0, 1), "T": (0, 1), "S": (1, 2)}]

        def read_subblock_metadata(self, **selection):
            focus = {0: [7840.0, 7835.711], 1: [7845.711]}[selection["S"]]
            assert selection["Z"] == selection["S"]  # the first plane of that scene
            return [({"S": selection["S"], "M": m}, xml.format(f)) for m, f in enumerate(focus)]

    monkeypatch.setitem(sys.modules, "aicspylibczi", types.SimpleNamespace(CziFile=FakeCzi))
    path = tmp_path / "f.czi"
    path.write_bytes(b"")
    voxel = (10.0, 0.5, 0.25)
    assert scene_position(path, 0, voxel) == {"y": 2869 * 0.5, "x": -44089 * 0.25, "z": 7835.711}
    assert scene_position(path, 1, voxel) == {"y": 5793 * 0.5, "x": -40772 * 0.25, "z": 7845.711}
    assert scene_position(path, 2, voxel) is None


def test_opera_stitched_position_matches_the_fused_raw_fields(tmp_path):
    export = write_export(tmp_path / "plate-Measurement 1")
    stitched = export / "stitchresult" / "A1_F1.ome.tiff"
    stitched.parent.mkdir()
    # the stitched folder has Harmony's own index.xml, which must not be taken for the raw one
    (stitched.parent / "index.xml").write_text("<EvaluationInputData/>")

    mosaic = OperaMosaicManager(export, flatfield=False, progress=False)
    fused = fusion.fused_position(fusion.plan_fusion(mosaic.fusion_tiles(field="A1_field0")))
    # fields 1, 2, 4 touch (cluster F1) and field 3 is far: with all fields the corner is the same
    assert stitched_position(stitched) == pytest.approx(fused)

    (stitched.parent / "FieldTables.json").write_text(json.dumps([{"Well": "A1", "FieldTables": [
        {"ClusterID": 0, "FieldTable": {"Field": [1, 2, 4]}},
        {"ClusterID": 1, "FieldTable": {"Field": [3]}}]}]))
    assert stitched_position(stitched) == pytest.approx(fused)
    far = stitched_position(stitched.with_name("A1_F2.ome.tiff"))
    lone = fusion.fused_position(fusion.plan_fusion(mosaic.fusion_tiles(field="A1_field1")))
    assert far == pytest.approx(lone) and far["x"] > fused["x"]

    assert stitched_position(stitched.with_name("H12_F1.ome.tiff")) is None  # well not in the index
    assert stitched_position(tmp_path / "elsewhere" / "stitchresult" / "A1_F1.ome.tiff") is None


def test_opera_stitched_position_from_a_bare_file_name(tmp_path, monkeypatch):
    """Opened as ``OperaManager("C9_F1.ome.tiff")`` from inside ``stitchresult``: the raw index
    is still found, and Harmony's own ``stitchresult/index.xml`` is not taken for it."""
    export = write_export(tmp_path / "plate-Measurement 1")
    stitched = export / "stitchresult" / "A1_F1.ome.tiff"
    stitched.parent.mkdir()
    (stitched.parent / "Index.xml").write_text("<EvaluationInputData/>")
    expected = stitched_position(stitched)
    monkeypatch.chdir(stitched.parent)
    assert expected is not None and stitched_position("A1_F1.ome.tiff") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# mosaic fusion
# ---------------------------------------------------------------------------

def _tile(origin, value=1, shape=(4, 8, 8), vox=(2.0, 1.0, 1.0)):
    return Tile(np.full(shape, value, np.uint16), origin, vox, "A")


def test_fused_image_records_its_position(tmp_path):
    tiles = [_tile((5.0, 10.0, 20.0)), _tile((5.0, 10.0, 26.0), value=2)]
    zm = fuse_tiles(tiles, tmp_path / "f.zarr", progress=False)
    assert zm.metadata["translations"][0] == {"z": 5.0, "y": 10.0, "x": 20.0}

    # an output voxel averages ds input voxels: its centre is half a block further
    zm = fuse_tiles(tiles, tmp_path / "d.zarr", ds=(1, 2, 2), progress=False)
    assert zm.metadata["translations"][0] == {"z": 5.0, "y": 10.5, "x": 20.5}

    zm = fuse_tiles(tiles, tmp_path / "c.zarr", crop={"z": slice(1, 3), "x": slice(3, 9)}, progress=False)
    assert zm.metadata["translations"][0] == {"z": 7.0, "y": 10.0, "x": 23.0}

    zm = fuse_tiles(tiles, tmp_path / "p.zarr", projection="max", progress=False)
    assert zm.metadata["translations"][0] == {"y": 10.0, "x": 20.0}


def test_mosaic_tile_carries_its_origin():
    class M(MosaicManager):
        def _scan(self):
            return [_tile((1.0, 2.0, 3.0))]

    assert M(".").tile(0).metadata["translations"] == [{"z": 1.0, "y": 2.0, "x": 3.0}]
