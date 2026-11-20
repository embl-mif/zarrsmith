"""Label tiles of a tiled segmentation: read from their stores, relabelled, fused."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.mosaic_manager import RelabeledArray, fuse_tiles, store_tiles

SHAPE = (4, 16, 24)  # z, y, x of each tile


def _label_store(path, labels, x0):
    """A segmentation tile as a tiled Cellpose run writes it: zyx labels at its translation."""
    md = {"axes": "zyx", "scales": [(1.0, 1.0, 1.0)], "data_type": "label",
          "translations": {"z": 0.0, "y": 0.0, "x": float(x0)}}
    mm.ArrayManager(labels, md).to_zarr(str(path), progress=False)
    return path


def _two_tiles(tmp_path):
    """Two tiles overlapping by 8 along x (x 16..24), each numbered from 1.

    Tile 1: object 1 at x 0..10, object 2 at x 12..24 (it crosses the seam).
    Tile 2: object 1 at x 16..24 (the same cell as tile 1's object 2), object 2 at x 26..36.
    """
    a = np.zeros(SHAPE, np.uint16)
    a[..., 0:10], a[..., 12:24] = 1, 2
    b = np.zeros(SHAPE, np.uint16)
    b[..., 0:8], b[..., 10:20] = 1, 2
    return _label_store(tmp_path / "t1.ome.zarr", a, 0), _label_store(tmp_path / "t2.ome.zarr", b, 16)


def test_store_tiles_reads_label_stores_at_their_position(tmp_path):
    s1, s2 = _two_tiles(tmp_path)
    (t1,), _ = store_tiles(s1)  # a zero translation is not written: placed at the origin
    (t2,), placed = store_tiles(s2)
    assert placed
    assert t1.origin == (0.0, 0.0, 0.0) and t2.origin == (0.0, 0.0, 16.0)
    assert t1.data.shape == SHAPE and t1.data[0, 0, 11] == 0 and t1.data[0, 0, 12] == 2


def test_relabeled_array_offsets_then_maps_and_keeps_the_background():
    data = np.array([[[0, 1, 2, 3]]], np.uint16)
    view = RelabeledArray(data, offset=100, mapping={102: 7})
    np.testing.assert_array_equal(view[:], [[[0, 101, 7, 103]]])
    assert view[:].dtype == np.uint32 and view.shape == (1, 1, 4)
    np.testing.assert_array_equal(view[0, 0, 2:], [7, 103])  # any slice


def test_relabeled_array_refuses_ids_that_do_not_fit():
    view = RelabeledArray(np.array([[[1]]], np.uint16), offset=2**32)
    with pytest.raises(ValueError, match="do not fit"):
        view[:]
    with pytest.raises(ValueError, match="integer dtype"):
        RelabeledArray(np.zeros((1, 1, 1)), dtype="float32")


def test_relabeled_array_is_picklable_when_its_data_is(tmp_path):
    s1, _ = _two_tiles(tmp_path)
    (tile,), _ = store_tiles(s1)
    view = RelabeledArray(tile.data, offset=1000, mapping={1002: 5})
    assert view.picklable
    copy = pickle.loads(pickle.dumps(view))
    np.testing.assert_array_equal(copy[:], view[:])
    assert not RelabeledArray(np.zeros((1, 1, 1), np.uint8)).picklable


def test_tiles_are_fused_into_one_numbering(tmp_path):
    """The tiled Cellpose merge: offset each tile by tile_index * step, apply the solver's
    mapping (tile 2's object 1 is tile 1's object 2), fuse without blending."""
    step = 1000
    mapping = {2001: 1002}
    tiles = []
    for index, store in enumerate(_two_tiles(tmp_path), start=1):
        (tile,), _ = store_tiles(store)
        tile.data = RelabeledArray(tile.data, offset=index * step, mapping=mapping)
        tiles.append(tile)

    zm = fuse_tiles(tiles, tmp_path / "merged.ome.zarr", blend="none", data_type="label",
                    levels=2, progress=False, workers=2)
    out = zm.data[0].compute()
    assert zm.metadata["data_type"] == "label" and out.dtype == np.uint32
    assert out.shape == (4, 16, 40)
    assert set(np.unique(out)) == {0, 1001, 1002, 2002}
    assert (out[..., 0:10] == 1001).all()
    assert (out[..., 12:24] == 1002).all()  # one cell across the seam at x 20
    assert (out[..., 26:36] == 2002).all()
    assert (out[..., 36:] == 0).all()
    assert set(np.unique(zm.data[1].compute())) <= {0, 1001, 1002, 2002}
