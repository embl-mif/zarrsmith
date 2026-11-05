"""Zeiss readers on real (small) CZI files, written by pylibCZIrw when the tests run."""
from __future__ import annotations

import pickle
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.cli import zarrsmith as cli
from zarrsmith.mosaic_manager import (CziTileArray, LuxendoMosaicManager, TiffStackArray, WrongReaderError,
                                      ZeissMosaicManager, detect_microscope)
from zarrsmith.mosaic_manager import zeiss_mosaic_manager as zeiss
from zarrsmith.utils.czi import CziBlock, block_key, read_block, read_block_index

pyczi = pytest.importorskip("pylibCZIrw.czi")

SIZE = 16  # pixels of a tile
OVERLAP_X = 12  # x of the second tile of scene 0
LAYOUT = {0: [(0, 0), (OVERLAP_X, 0)], 1: [(100, 50)]}  # scene -> pixel position (x, y) of its tiles


def plane_value(scene, m, t, c, z):
    """Every pixel of a plane has this value, so a test can tell which plane it read."""
    return 1000 * (scene + 1) + 100 * m + 10 * t + 2 * c + z + 1


def _write_metadata(f):
    f.write_metadata("test", channel_names={0: "A", 1: "B"}, scale_x=2e-7, scale_y=2e-7, scale_z=1e-6)


def _add_scene_names(path, names):
    """Name the scenes, as ZEN does: bioio-czi needs it and pylibCZIrw does not write it.

    The metadata is rewritten as a new segment at the end of the file, and the file
    header (``MetadataPosition``, at byte 92) points to it.
    """
    with open(path, "r+b") as f:
        f.seek(92)
        position = struct.unpack("<q", f.read(8))[0]
        f.seek(position + 32)
        size = struct.unpack("<i", f.read(4))[0]
        f.seek(position + 32 + 256)
        xml = f.read(size).decode("utf-8")
        scenes = "".join(f'<Scene Index="{i}" Name="{n}"/>' for i, n in enumerate(names))
        xml = xml.replace("<Dimensions>", f"<Dimensions><S><Scenes>{scenes}</Scenes></S>", 1).encode("utf-8")
        data = struct.pack("<ii", len(xml), 0) + bytes(248) + xml
        allocated = len(data) + (-len(data) % 32)
        end = f.seek(0, 2)
        new = end + (-end % 32)  # segments start on 32 bytes
        f.seek(new)
        f.write(b"ZISRAWMETADATA\0\0" + struct.pack("<qq", allocated, len(data)) + data.ljust(allocated, b"\0"))
        f.seek(92)
        f.write(struct.pack("<q", new))


@pytest.fixture(scope="module")
def czi_path(tmp_path_factory):
    """2 timepoints, 2 channels, 2 planes; scene 0 has 2 overlapping tiles, scene 1 has one."""
    path = tmp_path_factory.mktemp("zeiss") / "tile_scan.czi"
    with pyczi.create_czi(str(path), exist_ok=True) as f:
        for t in range(2):
            for scene, positions in LAYOUT.items():
                for m, location in enumerate(positions):
                    for c in range(2):
                        for z in range(2):
                            plane = np.full((SIZE, SIZE, 1), plane_value(scene, m, t, c, z), np.uint16)
                            f.write(plane, plane={"T": t, "C": c, "Z": z}, scene=scene, location=location)
        _write_metadata(f)
    _add_scene_names(path, ["TR1", "P1"])
    return path


@pytest.fixture(scope="module")
def single_path(tmp_path_factory):
    """One image without tiles: 2 channels, 2 planes."""
    path = tmp_path_factory.mktemp("zeiss") / "single.czi"
    with pyczi.create_czi(str(path), exist_ok=True) as f:
        for c in range(2):
            for z in range(2):
                plane = np.full((SIZE, SIZE, 1), plane_value(0, 0, 0, c, z), np.uint16)
                f.write(plane, plane={"T": 0, "C": c, "Z": z}, scene=0)
        _write_metadata(f)
    _add_scene_names(path, ["P1"])
    return path


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

def test_czi_is_detected_and_refused_by_other_readers(czi_path):
    assert detect_microscope(czi_path) == "zeiss"
    with pytest.raises(WrongReaderError) as e:
        LuxendoMosaicManager(czi_path)
    assert e.value.detected == "zeiss"


def test_tiles_of_every_scene_timepoint_and_channel(czi_path):
    mosaic = ZeissMosaicManager(czi_path)
    assert len(mosaic.tiles) == 3 * 2 * 2
    assert mosaic.channels == ["A", "B"]
    assert mosaic.timepoints == [0, 1]
    assert mosaic.fields == ["scene0_TR1", "scene1_P1"]  # one field per scene, in acquisition order
    assert {t.attrs["stack"] for t in mosaic.field_tiles("scene0_TR1")} == {"scene0_TR1-m00", "scene0_TR1-m01"}
    assert len(mosaic.positions) == 3 * 2
    t = mosaic.select(channel="B", timepoint=1, stack="scene0_TR1-m01")[0]
    assert mosaic.tile_id(t) == "B/scene0_TR1-m01"
    assert t.shape == (2, SIZE, SIZE)
    assert t.dtype == np.uint16
    assert t.voxel_size == pytest.approx((1.0, 0.2, 0.2))
    assert t.origin[1:] == pytest.approx((0.0, OVERLAP_X * 0.2))


def test_tiles_read_their_own_planes(czi_path):
    mosaic = ZeissMosaicManager(czi_path)
    for tile in mosaic.tiles:
        a = tile.attrs
        c = mosaic.channels.index(tile.channel)
        expected = [plane_value(a["scene"], a["mosaic_index"], tile.timepoint, c, z) for z in range(2)]
        assert np.asarray(tile.data[:]).max(axis=(1, 2)).tolist() == expected
        assert np.asarray(tile.data[1, 2:4, 3:6]).shape == (2, 3)


def test_tile_data_is_picklable_and_displayed_per_plane(czi_path):
    mosaic = ZeissMosaicManager(czi_path)
    t = mosaic.tiles[0]
    assert isinstance(t.data, CziTileArray) and t.data.picklable
    copy = pickle.loads(pickle.dumps(t.data))
    np.testing.assert_array_equal(copy[:], t.data[:])
    assert mosaic._lazy(t).chunksize == (1, SIZE, SIZE)


def test_missing_plane_is_read_as_zeros(czi_path):
    data = CziTileArray(czi_path, {"S": 0, "M": 0, "T": 0, "C": 0}, [0, None], (2, SIZE, SIZE), np.uint16)
    assert data[0].max() == plane_value(0, 0, 0, 0, 0)
    assert data[1].max() == 0


def _write_scan(path, dtype=np.uint16, compression=None):
    """Two tiles of one scene, 2 channels, 2 planes, in which both bytes of a 16-bit pixel
    vary (so that hi-lo packing matters); returns ``{(m, c, z): plane}``."""
    limit = 256 if dtype == np.uint8 else 2**16
    expected = {}
    with pyczi.create_czi(str(path), exist_ok=True, compression_options=compression) as f:
        for m, location in enumerate(LAYOUT[0]):
            for c in range(2):
                for z in range(2):
                    values = (np.arange(SIZE * SIZE) * 251 + 7 * m + 11 * c + 13 * z) % limit
                    plane = values.astype(dtype).reshape(SIZE, SIZE, 1)
                    f.write(plane, plane={"T": 0, "C": c, "Z": z}, scene=0, location=location)
                    expected[m, c, z] = plane[..., 0]
        _write_metadata(f)
    return expected


@pytest.mark.parametrize("dtype, compression, mode", [
    (np.uint16, None, 0),
    (np.uint8, None, 0),
    (np.float32, None, 0),
    (np.uint16, "zstd0:ExplicitLevel=1", 5),
    (np.uint16, "zstd1:ExplicitLevel=1", 6),
    (np.uint8, "zstd1:ExplicitLevel=1", 6),
    (np.uint16, "zstd1:ExplicitLevel=1;PreProcess=HiLoByteUnpack", 6),
])
def test_planes_are_read_from_their_position_in_the_file(tmp_path, monkeypatch, dtype, compression, mode):
    """The pixels of aicspylibczi, without going through it: it scans the whole directory for every plane."""
    from aicspylibczi import CziFile

    path = tmp_path / "scan.czi"
    expected = _write_scan(path, dtype, compression)
    mosaic = ZeissMosaicManager(path)
    assert {b.compression for t in mosaic.tiles for b in t.data.blocks} == {mode}
    with open(path, "rb") as f:
        library = {k: CziFile(f).read_image(S=0, M=k[0], T=0, C=k[1], Z=k[2])[0].reshape(SIZE, SIZE) for k in expected}

    def refuse(*args, **kwargs):
        raise AssertionError("read through aicspylibczi")

    monkeypatch.setattr(CziFile, "read_image", refuse)
    for tile in mosaic.tiles:
        m, c = tile.attrs["mosaic_index"], mosaic.channels.index(tile.channel)
        planes = np.asarray(pickle.loads(pickle.dumps(tile.data))[:])
        assert planes.dtype == dtype
        for z in range(2):
            np.testing.assert_array_equal(planes[z], expected[m, c, z])
            np.testing.assert_array_equal(planes[z], library[m, c, z])
        assert tile.data[0].flags.writeable  # like the planes of aicspylibczi


def test_hi_lo_packing_is_exercised(tmp_path):
    """The file of the hi-lo case above does pack the bytes: its zstd header says so."""
    path = tmp_path / "scan.czi"
    _write_scan(path, compression="zstd1:ExplicitLevel=1;PreProcess=HiLoByteUnpack")
    block = next(iter(read_block_index(path, np.uint16).values()))
    with open(path, "rb") as f:
        f.seek(block.position + 32)
        metadata = struct.unpack("<i", f.read(4))[0]
        f.seek(block.data_offset + metadata)
        assert f.read(3) == b"\x03\x01\x01"


def _patch_entries(path, offset, value):
    """Write the 32-bit ``value`` at ``offset`` in every directory entry of the file (7 dimensions each)."""
    with open(path, "r+b") as f:
        f.seek(84)
        directory = struct.unpack("<q", f.read(8))[0]
        f.seek(directory + 32)
        count = struct.unpack("<i", f.read(4))[0]
        for i in range(count):
            f.seek(directory + 160 + i * (32 + 7 * 20) + offset)
            f.write(struct.pack("<i", value))


@pytest.mark.parametrize("offset, value", [
    (18, 4),  # compressed with JPEG XR
    (14, 1),  # in another part of a split file
    (22, 1),  # a pyramid level
    (2, 0),  # another pixel type (gray8)
    (32 + 16, SIZE // 2),  # X (the first dimension) stored smaller than it covers: a pyramid level
])
def test_planes_that_need_the_library_are_not_indexed(tmp_path, offset, value):
    path = tmp_path / "scan.czi"
    _write_scan(path)
    assert len(read_block_index(path, np.uint16)) == 2 * 2 * 2
    _patch_entries(path, offset, value)
    assert read_block_index(path, np.uint16) == {}


def test_planes_with_the_same_coordinates_are_not_indexed(tmp_path):
    path = tmp_path / "scan.czi"
    _write_scan(path)
    _patch_entries(path, 32 + 2 * 20 + 4, 0)  # M (the third dimension): every tile becomes the first one
    assert read_block_index(path, np.uint16) == {}


def test_plane_without_a_block_is_read_with_the_library(tmp_path):
    path = tmp_path / "scan.czi"
    expected = _write_scan(path)
    block = read_block_index(path, np.uint16)[block_key({"S": 0, "M": 1, "T": 0, "C": 1, "Z": 1})]
    data = CziTileArray(path, {"S": 0, "M": 1, "T": 0, "C": 1}, [0, 1], (2, SIZE, SIZE), np.uint16, [None, block])
    np.testing.assert_array_equal(data[0], expected[1, 1, 0])
    np.testing.assert_array_equal(data[1], expected[1, 1, 1])


def test_plane_that_does_not_fit_is_refused(tmp_path):
    path = tmp_path / "scan.czi"
    _write_scan(path)
    block = next(iter(read_block_index(path, np.uint16).values()))
    with pytest.raises(ValueError, match="does not hold"):
        read_block(path, block, (SIZE, SIZE + 1), np.uint16)
    with pytest.raises(ValueError, match="cut short"):
        read_block(path, CziBlock(block.position, path.stat().st_size - 8, block.compression), (SIZE, SIZE), np.uint16)
    with pytest.raises(ValueError, match="does not match"):
        CziTileArray(path, {}, [0, 1], (2, SIZE, SIZE), np.uint16, [None])


def test_scene_index_loads_one_scene(czi_path):
    assert ZeissMosaicManager(czi_path, scene_index=1).fields == ["scene1_P1"]
    with pytest.raises(ValueError, match=r"Invalid scene index 7.*\[0, 1\]"):
        ZeissMosaicManager(czi_path, scene_index=7)


def test_file_without_tiles_points_to_the_zeiss_manager(single_path):
    with pytest.raises(ValueError, match="ZeissManager"):
        ZeissMosaicManager(single_path)


def test_folder_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"one \.czi file"):
        ZeissMosaicManager(tmp_path)


def test_names_from_the_metadata():
    md = ET.fromstring(
        "<ImageDocument><Metadata><Information><Image><Dimensions>"
        "<Channels><Channel Id='Channel:0' Name='AF488'/><Channel Id='Channel:1' Name='AF488'/><Channel Id='Channel:2'/>"
        "</Channels>"
        "<S><Scenes><Scene Index='0' Name='Tile Region 1'/><Scene Index='1' Name='P1'/><Scene Index='2'/></Scenes></S>"
        "</Dimensions></Image></Information></Metadata></ImageDocument>")
    assert zeiss.channel_names(md, [0, 1, 2, 3]) == {0: "AF488 (C0)", 1: "AF488 (C1)", 2: "Channel:2", 3: "C3"}
    names = zeiss.scene_names(md)
    assert names == {0: "Tile Region 1", 1: "P1", 2: None}
    assert [zeiss.field_name(s, n) for s, n in names.items()] == ["scene0_Tile_Region_1", "scene1_P1", "scene2"]


def test_z_origin_is_the_focus_position_of_the_tile():
    xml = "<METADATA><Tags><StageXPosition>1</StageXPosition><FocusPosition>+0000012.5000</FocusPosition></Tags></METADATA>"
    czi = SimpleNamespace(read_subblock_metadata=lambda **k: [({"S": 0, "M": 1, "T": 2, "Z": k["Z"], "C": k["C"]}, xml),
                                                              ({"S": 0, "M": 2, "T": 2}, "<METADATA />")])
    assert ZeissMosaicManager._focus_positions(czi, 0, 0) == {(0, 1, 2): 12.5}


def test_plane_stacks_share_their_reading_code(tmp_path):
    """The base class moved out of TiffStackArray: a missing plane still reads as zeros."""
    data = TiffStackArray([None, None], (2, 3, 4), np.uint16)
    assert data[:].shape == (2, 3, 4) and data[1].max() == 0
    assert pickle.loads(pickle.dumps(data)).shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workers", [1, 2])
def test_fuse_timelapse_of_one_scene(czi_path, tmp_path, workers):
    mosaic = ZeissMosaicManager(czi_path)
    out = tmp_path / "fused.ome.zarr"
    mosaic.fuse(out, field="scene0_TR1", blend="max", levels=1, workers=workers)

    z = mm.ZarrManager(out)
    assert z.metadata["axes"] == "tczyx"
    data = np.asarray(z.data[0])
    assert data.shape == (2, 2, 2, SIZE, OVERLAP_X + SIZE)
    for t in range(2):
        for c in range(2):
            for k in range(2):
                assert data[t, c, k, 5, 0] == plane_value(0, 0, t, c, k)
                assert data[t, c, k, 5, -1] == plane_value(0, 1, t, c, k)


def test_command_line_scene_index_and_auto_detection(czi_path, tmp_path):
    out = tmp_path / "scene1.ome.zarr"
    assert cli.mosaic_convert(czi_path, out, scene_index=1, timepoints=[1]) is True
    data = np.asarray(mm.ZarrManager(out).data[0])
    assert data.squeeze()[0, 0, 0, 0] == plane_value(1, 0, 1, 0, 0)


# ---------------------------------------------------------------------------
# ZeissManager: opens a real file, unlike a mocked reader
# ---------------------------------------------------------------------------

def test_zeiss_manager_opens_a_czi_file(single_path):
    m = mm.ZeissManager(single_path)
    assert m.data[0].shape == (1, 2, 2, SIZE, SIZE)
    assert m.chunks == (1, 1, 1, SIZE, SIZE)  # one chunk per plane
    assert list(m.scenes) == ["P1"]
    assert np.asarray(m.data[0][0, 1, 1]).max() == plane_value(0, 0, 0, 1, 1)
    assert m.metadata["scales"][0] == pytest.approx((1.0, 0.2, 0.2))


def test_zeiss_manager_stitches_every_scene_of_a_tile_scan(czi_path):
    m = mm.ZeissManager(czi_path)
    assert list(m.scenes) == ["TR1", "P1"]
    assert m.data[0].shape == (2, 2, 2, SIZE, OVERLAP_X + SIZE)
    assert np.asarray(m.data[0][1, 1, 0, 5, 0]) == plane_value(0, 0, 1, 1, 0)
    assert np.asarray(m.data[0][1, 1, 0, 5, -1]) == plane_value(0, 1, 1, 1, 0)
    m.chunks = None
    m.read(scene_index=1)
    assert m.data[0].shape == (2, 2, 2, SIZE, SIZE)
    assert np.asarray(m.data[0][0, 0, 1]).max() == plane_value(1, 0, 0, 0, 1)


@pytest.mark.parametrize("scene_index, width", [(None, OVERLAP_X + SIZE), (0, OVERLAP_X + SIZE), (1, SIZE)])
def test_convert_defaults_to_the_first_scene(czi_path, tmp_path, scene_index, width):
    out = tmp_path / "converted.ome.zarr"
    cli.zarr_convert(czi_path, out, microscope="zeiss", scene_index=scene_index)
    assert mm.ZarrManager(out).data[0].shape[-1] == width


def test_convert_a_file_with_one_scene(single_path, tmp_path):
    out = tmp_path / "converted.ome.zarr"
    cli.zarr_convert(single_path, out, microscope="zeiss")
    assert mm.ZarrManager(out).data[0].shape[-2:] == (SIZE, SIZE)


@pytest.fixture
def converter(monkeypatch):
    pytest.importorskip("napari")
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QApplication

    from zarrsmith.napari import _convert_widget as convert_module

    QApplication.instance() or QApplication([])
    # The plugin redirects stdout/stderr into its log box; undo that after the test.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    viewer = ViewerModel()
    monkeypatch.setattr(convert_module, "current_viewer", lambda: viewer)
    widget = convert_module.convert_widget()
    widget.viewer = viewer
    yield widget
    widget.close()


def test_converter_follows_the_scene(converter, czi_path, tmp_path):
    load, convert = converter.visualize_widget, converter.convert_widget
    load.file_format.value = "zeiss"
    load.input_path.value = czi_path
    assert load.scene_index.max == 1  # this file has 2 scenes
    assert convert.x_range.max == OVERLAP_X + SIZE - 1

    load()
    assert converter.viewer.layers[0].data.shape[-1] == OVERLAP_X + SIZE

    typed = tmp_path / "mine.zarr"
    convert.output_path.value = typed
    load.scene_index.value = 1
    assert convert.x_range.max == SIZE - 1
    assert convert.x_range.value == (0, SIZE - 1)  # not the crop of the other scene
    assert f"x {SIZE} x {SIZE} px" in converter.output_label.text()
    assert Path(convert.output_path.value) == typed

    load()
    layer = next(l for l in converter.viewer.layers if l.name not in ("ROI", "Zrange"))
    assert layer.data.shape[-1] == SIZE
    assert np.asarray(layer.data[0][0, 0]).max() == plane_value(1, 0, 0, 0, 0)

    load.scene_index.value = 0
    assert convert.x_range.max == OVERLAP_X + SIZE - 1


def test_converter_output_follows_the_scene_until_typed(converter, czi_path, single_path, tmp_path):
    load, convert = converter.visualize_widget, converter.convert_widget
    load.file_format.value = "zeiss"
    load.input_path.value = czi_path
    assert Path(convert.output_path.value) == czi_path.parent / "tile_scan_TR1.zarr"
    load.scene_index.value = 1
    assert Path(convert.output_path.value) == czi_path.parent / "tile_scan_P1.zarr"

    typed = tmp_path / "mine.zarr"
    convert.output_path.value = typed
    load.scene_index.value = 0
    assert Path(convert.output_path.value) == typed

    load.input_path.value = single_path  # one scene: no tag; a new file names its output again
    assert Path(convert.output_path.value) == single_path.parent / "single.zarr"


def test_converter_scene_is_limited_to_the_new_file(converter, czi_path, single_path):
    load, convert = converter.visualize_widget, converter.convert_widget
    load.file_format.value = "zeiss"
    load.input_path.value = czi_path
    load.scene_index.value = 1
    load.input_path.value = single_path  # has no scene 1
    assert load.scene_index.value == 0 and load.scene_index.max == 0
    assert convert.x_range.max == SIZE - 1


def test_zeiss_interval_is_in_seconds_whatever_unit_zen_displays(single_path):
    m = mm.ZeissManager(single_path)
    md = ET.fromstring(
        "<ImageDocument><Metadata>"
        "<Scaling><Items><Distance Id='X'><Value>2e-7</Value></Distance><Distance Id='Y'><Value>2e-7</Value></Distance>"
        "<Distance Id='Z'><Value>1e-6</Value></Distance></Items></Scaling>"
        "<TimeSeriesSetup><Interval><TimeSpan><Value>60</Value><DefaultUnitFormat>min</DefaultUnitFormat></TimeSpan>"
        "</Interval></TimeSeriesSetup></Metadata></ImageDocument>")
    m._image = SimpleNamespace(metadata=md, channel_names=["A", "B"])
    parsed = m._parse_metadata()
    assert parsed["time_increment"] == 60
    assert parsed["time_increment_unit"] == "second"
