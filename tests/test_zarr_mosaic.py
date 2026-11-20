# tests/test_zarr_mosaic.py
"""ZarrMosaicManager: a folder of OME-Zarr stores, each placed at its NGFF translation."""
from __future__ import annotations

import logging
import pickle
import shutil
import sys

import numpy as np
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.cli import zarrsmith as cli
from zarrsmith.mosaic_manager import MOSAIC_READERS, ZarrMosaicManager, detect_microscope

SHAPE = (2, 4, 16, 16)  # c, z, y, x of every store
VOX = (2.0, 1.0, 1.0)


def _store(path, value, position=None, axes="czyx", shape=SHAPE, data_type="intensity"):
    data = np.full(shape, value, np.uint16)
    md = {"axes": axes, "scales": [VOX[-sum(a in "zyx" for a in axes):]], "data_type": data_type}
    if "c" in axes:
        md["channel_names"] = ["DAPI", "GFP"]
    if position is not None:
        md["translations"] = position
    mm.ArrayManager(data, md).to_zarr(str(path), progress=False, drop_singleton=False)
    return path


@pytest.fixture
def stores(tmp_path):
    """a and b overlap by 4 columns (one field); c is far away; a label store is ignored."""
    folder = tmp_path / "stores"
    folder.mkdir()
    _store(folder / "a.ome.zarr", 10, {"z": 0.0, "y": 5.0, "x": 0.0})
    _store(folder / "b.ome.zarr", 30, {"z": 0.0, "y": 5.0, "x": 12.0})
    _store(folder / "c.ome.zarr", 50, {"z": 0.0, "y": 0.0, "x": 1000.0})
    _store(folder / "seg.ome.zarr", 1, {"z": 0.0, "y": 0.0, "x": 0.0}, axes="zyx", shape=SHAPE[1:],
           data_type="label")
    (folder / "notes.txt").write_text("not a store")
    return folder


def test_registered_and_detected(stores):
    assert MOSAIC_READERS["zarr"] is ZarrMosaicManager
    assert detect_microscope(stores) == "zarr"
    assert detect_microscope(stores / "a.ome.zarr") == "zarr"
    assert cli._resolve_mosaic_manager(str(stores))[1] == "zarr"


def test_scan_places_the_stores(stores):
    m = ZarrMosaicManager(stores)
    assert len(m.tiles) == 6  # 3 intensity stores x 2 channels; the label store is left out
    assert m.channels == ["DAPI", "GFP"]
    assert len(m.fields) == 2  # a + b touch, c is apart
    b = m.select(channel="GFP")[1]
    assert b.attrs["stack"] == "b" and b.origin == (0.0, 5.0, 12.0) and b.voxel_size == VOX
    assert b.shape == SHAPE[1:] and b.dtype == np.uint16
    assert m.tile(0).metadata["translations"] == [{"z": 0.0, "y": 5.0, "x": 0.0}]


def _field_of(m, stack):
    return next(t.field for t in m.tiles if t.attrs["stack"] == stack)


def test_fuse_touching_stores(stores):
    m = ZarrMosaicManager(stores)
    zm = m.fuse(stores / "fused.ome.zarr", field=_field_of(m, "a"), blend="max", progress=False,
                drop_singleton=False)
    data = np.asarray(zm.data[0])
    assert data.shape == (2, 4, 16, 28)  # 12 + 16 columns
    assert (data[..., :12] == 10).all() and (data[..., 12:] == 30).all()
    assert zm.metadata["translations"][0] == {"z": 0.0, "y": 5.0, "x": 0.0}

    # the fusion, written in the folder, is not taken for one more store
    assert len(ZarrMosaicManager(stores).tiles) == 6
    # but a fused store brought into another folder is a tile there
    other = stores.parent / "fields"
    other.mkdir()
    shutil.copytree(stores / "fused.ome.zarr", other / "fused.ome.zarr")
    moved = ZarrMosaicManager(other)
    assert len(moved.tiles) == 2 and moved.tiles[0].shape == (4, 16, 28)


def test_tile_array_is_lazy_and_picklable(stores):
    tile = ZarrMosaicManager(stores).tiles[0]
    clone = pickle.loads(pickle.dumps(tile.data))
    assert clone._array is None
    assert np.array_equal(clone[1:3, :4, ::2], tile.data[1:3, :4, ::2])
    assert clone[...].shape == SHAPE[1:] and clone[0].shape == SHAPE[2:]


def test_2d_and_timelapse_stores(tmp_path):
    folder = tmp_path / "stores"
    folder.mkdir()
    _store(folder / "flat.ome.zarr", 7, {"y": 5.0, "x": 6.0}, axes="yx", shape=(8, 9))
    _store(folder / "movie.ome.zarr", 3, {"z": 0.0, "y": 0.0, "x": 50.0}, axes="tzyx", shape=(3, 2, 8, 9))
    m = ZarrMosaicManager(folder)
    flat = [t for t in m.tiles if t.attrs["stack"] == "flat"][0]
    assert flat.shape == (1, 8, 9) and flat.origin == (0.0, 5.0, 6.0) and flat.voxel_size == (1.0, 1.0, 1.0)
    assert flat.data[:, 2:4].shape == (1, 2, 9) and (flat.data[0] == 7).all()
    assert flat.channel == "ch0"
    assert m.timepoints == [0, 1, 2]


def test_store_without_position_is_placed_at_the_origin(tmp_path, caplog):
    folder = tmp_path / "stores"
    folder.mkdir()
    _store(folder / "a.ome.zarr", 1)
    with caplog.at_level(logging.WARNING, logger="zarrsmith"):
        m = ZarrMosaicManager(folder)
    assert m.tiles[0].origin == (0.0, 0.0, 0.0)
    assert "no position" in caplog.text and "a.ome.zarr" in caplog.text


def test_visualize_at_the_positions(stores):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    v = ViewerModel()
    m = ZarrMosaicManager(stores)
    m.visualize(tiles=m.select(channel="DAPI"), viewer=v)
    assert sorted(tuple(layer.translate)[-1] for layer in v.layers) == [0.0, 12.0, 1000.0]


def test_cli_fuse_detects_the_stores(stores, tmp_path, monkeypatch):
    out = tmp_path / "field.ome.zarr"
    far = _field_of(ZarrMosaicManager(stores), "c")
    monkeypatch.setattr(sys, "argv", ["zarrsmith", "fuse", "-i", str(stores), "-z", str(out), "-b", "max",
                                      "-f", far])
    cli.main()
    fused = mm.ZarrManager(str(out))
    assert fused.metadata["translations"][0]["x"] == 1000.0
    assert (np.asarray(fused.data[0]) == 50).all()
