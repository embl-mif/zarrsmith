"""Mosaics with several timepoints: reference geometry, fusion, napari slider, command line."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

import zarrsmith.microscope_manager as mm
from zarrsmith.cli import zarrsmith as cli
from zarrsmith.mosaic_manager import LuxendoMosaicManager, MosaicManager, Tile, fuse_tiles, fusion, plan_fusion
from zarrsmith.mosaic_manager.tile import first_of_series, series_keys

from .test_luxendo_mosaic import _write_tile


def _tile(value, x, tp, channel="A", group=None, stack=None, shape=(4, 8, 8), y=0.0, z=0.0):
    attrs = {} if stack is None else {"stack": stack}
    return Tile(np.full(shape, value, np.uint16), (z, y, x), (1.0, 1.0, 1.0), channel, group, tp, attrs)


def _series(n_timepoints=3, drift=0.0):
    """Two overlapping tiles per timepoint (value = 10 * (t + 1) + tile), drifting in x over time."""
    return [_tile(10 * (t + 1) + i, i * 4 + t * drift, t, stack=f"s{i}")
            for t in range(n_timepoints) for i in range(2)]


class Fake(MosaicManager):
    tile_list: list = []

    def _scan(self):
        return list(type(self).tile_list)


def _mosaic(tiles):
    return type("M", (Fake,), {"tile_list": tiles})(".")


@pytest.fixture(autouse=True)
def small_regions(monkeypatch):
    """Several fusion regions even for tiny test data."""
    monkeypatch.setattr(fusion, "_REGION_3D", (4, 4))
    monkeypatch.setattr(fusion, "_REGION_2D", (4, 4))


# ---------------------------------------------------------------------------
# matching tiles across timepoints
# ---------------------------------------------------------------------------

def test_series_keys_match_positions_across_timepoints():
    tiles = _series(2)
    keys = series_keys(tiles)
    assert keys[0] == keys[2] and keys[1] == keys[3] and keys[0] != keys[1]
    assert first_of_series(tiles) == [0, 1, 0, 1]
    # without a stack name: by order within each timepoint
    plain = [_tile(1, 0, 0), _tile(2, 4, 0), _tile(3, 0, 1), _tile(4, 4, 1)]
    assert first_of_series(plain) == [0, 1, 0, 1]
    # the earliest timepoint comes first, whatever the order of the list
    assert first_of_series(list(reversed(tiles))) == [2, 3, 2, 3]
    # channel and group are part of a position
    assert len(set(series_keys([_tile(1, 0, 0, "A", "L", "s"), _tile(1, 0, 0, "A", "R", "s"),
                                _tile(1, 0, 0, "B", "L", "s")]))) == 3


def test_manager_inspection_over_timepoints():
    m = _mosaic(_series(3, drift=1.0))
    assert m.timepoints == [0, 1, 2]
    assert len(m.tiles) == 6 and len(m.positions) == 2
    assert all(t.timepoint == 0 for t in m.positions)
    assert [t.timepoint for t in m.select(timepoint=1)] == [1, 1]
    assert m.bounding_box() == ((0.0, 0.0, 0.0), (4.0, 8.0, 12.0))  # drift of later timepoints left out
    # ids name a position: the same at every timepoint, and they find every timepoint
    assert {m.tile_id(t) for t in m.tiles} == {"A/s0", "A/s1"}
    assert sorted(t.timepoint for t in m.tiles_by_ids(["A/s0"])) == [0, 1, 2]
    assert sorted(t.timepoint for t in m.tiles_by_ids(["s1"])) == [0, 1, 2]
    # readers without stack names too
    plain = _mosaic([_tile(1, 0, t) for t in range(2)])
    assert {plain.tile_id(t) for t in plain.tiles} == {"A/0"}


def test_stage_drift_neither_joins_nor_splits_fields():
    # two separate fields at timepoint 0; at timepoint 1 the second drifts onto the first
    tiles = [_tile(1, 0, 0, stack="a"), _tile(2, 100, 0, stack="b"),
             _tile(3, 0, 1, stack="a"), _tile(4, 2, 1, stack="b")]
    m = _mosaic(tiles)
    assert m.fields == ["field0", "field1"]
    assert [t.field for t in m.tiles] == ["field0", "field1", "field0", "field1"]
    # one field at timepoint 0; at timepoint 1 a tile drifts away from it
    tiles = [_tile(1, 0, 0, stack="a"), _tile(2, 4, 0, stack="b"),
             _tile(3, 0, 1, stack="a"), _tile(4, 500, 1, stack="b")]
    assert _mosaic(tiles).fields == ["field0"]


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

def test_timelapse_output_has_a_t_axis(tmp_path):
    tiles = [t for c in ("A", "B") for t in
             [_tile(10 * (tp + 1), 0, tp, c, stack="s0") for tp in range(3)]]
    zm = fuse_tiles(tiles, tmp_path / "o.zarr", blend="max", levels=2, downscale_factor=2, progress=False)
    assert zm.metadata["axes"] == "tczyx"
    assert zm.data[0].shape == (3, 2, 4, 8, 8)
    assert zm.data[1].shape == (3, 2, 2, 4, 4)  # time and channels are not downsampled
    data = zm.data[0].compute()
    assert [int(data[t].max()) for t in range(3)] == [10, 20, 30]
    assert [int(data[t].min()) for t in range(3)] == [10, 20, 30]
    # the original timepoints and the reference are recorded
    attrs = zm.root.attrs[fusion.PROVENANCE_KEY]
    assert attrs == {"timepoints": [0, 1, 2], "reference_timepoint": 0}


def test_single_timepoint_keeps_its_axes_and_attributes(tmp_path):
    zm = fuse_tiles([_tile(5, 0, 0), _tile(6, 8, 0)], tmp_path / "o.zarr", blend="max", progress=False)
    assert zm.metadata["axes"] == "zyx"  # a single channel is dropped, as for zarrsmith convert
    assert fusion.PROVENANCE_KEY not in zm.root.attrs
    plan = plan_fusion([_tile(5, 0, 0)])
    assert plan.axes == "czyx" and plan.summary().startswith("1 ch x 4 x 8 x 8 px (czyx)")
    assert "reference" not in plan.summary()


@pytest.mark.parametrize("blend,kw", [("sigmoid", {}), ("max", {}), ("sum", {}), ("midline", {}),
                                      ("sigmoid", {"projection": "max"}), ("midline", {"projection": "sum"}),
                                      ("sigmoid", {"ds": (2, 2, 2)}), ("max", {"crop": {"x": slice(10, 50)}})])
def test_every_frame_matches_a_fusion_of_that_timepoint_alone(tmp_path, blend, kw):
    """Same stage positions at every timepoint: a frame is exactly the single-timepoint result."""
    rng = np.random.default_rng(0)
    tiles = []
    for tp in range(3):
        for i, (group, x) in enumerate((("L", 0), ("L", 16), ("R", 32), ("R", 48))):
            data = rng.integers(0, 4000, size=(8, 24, 24), dtype=np.uint16)
            tiles.append(Tile(data, (0, 0, x), (1, 1, 1), "A", group, tp, {"stack": f"s{i}"}))

    zm = fuse_tiles(tiles, tmp_path / "all.zarr", blend=blend, progress=False, drop_singleton=False, **kw)
    assert zm.metadata["axes"][:2] == "tc"
    stack = zm.data[0].compute()
    assert stack.shape[0] == 3
    for tp in range(3):
        one = fuse_tiles([t for t in tiles if t.timepoint == tp], tmp_path / f"{tp}.zarr", blend=blend,
                         progress=False, drop_singleton=False, **kw)
        np.testing.assert_array_equal(stack[tp], one.data[0].compute())


def test_reference_timepoint_sets_the_geometry(tmp_path):
    tiles = _series(3, drift=5.0)  # the reported positions drift by 5 voxels per timepoint
    # reference 0: tiles at x = 0 and 4 at every timepoint, 12 wide, whatever the later positions say
    zm = fuse_tiles(tiles, tmp_path / "r0.zarr", blend="max", progress=False)
    data = zm.data[0].compute()
    assert zm.metadata["axes"] == "tzyx" and data.shape == (3, 4, 8, 12)
    for t in range(3):
        assert (data[t, ..., :4] == 10 * (t + 1)).all()      # first tile only
        assert (data[t, ..., 4:] == 10 * (t + 1) + 1).all()  # second tile, brighter in the overlap

    # the positions of the reference timepoint count: at timepoint 2 the tiles still overlap the same
    # way (drift is common), but the origin is the one of that timepoint
    plan = plan_fusion(tiles, reference_timepoint=2)
    assert plan.reference_timepoint == 2 and plan.origin == (0.0, 0.0, 10.0)
    assert "reference timepoint 2" in plan.summary()

    # a tile that moves relative to the other one: the geometry is not affected unless it is the reference
    moved = _series(2)
    moved[3] = _tile(21, 40, 1, stack="s1")
    assert plan_fusion(moved, reference_timepoint=0).shape == (2, 1, 4, 8, 12)
    assert plan_fusion(moved, reference_timepoint=1).shape == (2, 1, 4, 8, 48)


def test_transitions_come_from_the_reference_at_every_timepoint(tmp_path):
    # L/R tiles whose overlap would change at timepoint 1 if its own positions were used
    tiles = [_tile(100, 0, 0, group="L", stack="l"), _tile(200, 4, 0, group="R", stack="r"),
             _tile(100, 0, 1, group="L", stack="l"), _tile(200, 7, 1, group="R", stack="r")]
    data = fuse_tiles(tiles, tmp_path / "o.zarr", blend="midline", progress=False).data[0].compute()
    assert data.shape == (2, 4, 8, 12)
    np.testing.assert_array_equal(data[0], data[1])
    assert (data[1, ..., :6] == 100).all() and (data[1, ..., 6:] == 200).all()  # cut at x = 6


def test_timepoint_selection(tmp_path):
    tiles = _series(4)
    two = fuse_tiles(tiles, tmp_path / "two.zarr", blend="max", timepoints=[3, 1], progress=False)
    data = two.data[0].compute()
    assert two.metadata["axes"] == "tzyx" and data.shape[0] == 2
    assert data[0, 0, 0, 0] == 20 and data[1, 0, 0, 0] == 40  # increasing order
    assert two.root.attrs[fusion.PROVENANCE_KEY] == {"timepoints": [1, 3], "reference_timepoint": 0}

    # a single timepoint of a timelapse has no t axis, and says where it comes from
    one = fuse_tiles(tiles, tmp_path / "one.zarr", blend="max", timepoints=[2], progress=False)
    assert one.metadata["axes"] == "zyx" and one.data[0][0, 0, 0].compute() == 30
    assert one.root.attrs[fusion.PROVENANCE_KEY] == {"timepoints": [2], "reference_timepoint": 0}

    with pytest.raises(ValueError, match=r"Timepoints \[9\] not found; available: \[0, 1, 2, 3\]"):
        plan_fusion(tiles, timepoints=[9])
    with pytest.raises(ValueError, match="No timepoint selected"):
        plan_fusion(tiles, timepoints=[])
    with pytest.raises(ValueError, match=r"Reference timepoint 9 not found"):
        plan_fusion(tiles, reference_timepoint=9)


def test_parts_fused_separately_share_the_geometry(tmp_path):
    tiles = _series(4, drift=3.0)
    first = plan_fusion(tiles, timepoints=[0, 1])
    second = plan_fusion(tiles, timepoints=[2, 3])  # default reference: first of the tiles, not of the selection
    assert second.reference_timepoint == 0
    assert first.origin == second.origin and first.shape == second.shape
    # also when the reference timepoint is not fused
    only = fuse_tiles(tiles, tmp_path / "o.zarr", blend="max", timepoints=[3], reference_timepoint=1, progress=False)
    assert only.root.attrs[fusion.PROVENANCE_KEY]["reference_timepoint"] == 1
    assert only.data[0].shape == (4, 8, 12)


def test_missing_tile_stays_blank(tmp_path, caplog):
    tiles = _series(3)
    del tiles[5]  # second tile of the last timepoint: an interrupted acquisition
    with caplog.at_level(logging.WARNING, logger="zarrsmith"):
        plan = plan_fusion(tiles)
    assert plan.n_missing == 1 and "1 tile(s) missing" in plan.summary()
    assert "missing at some timepoints" in caplog.text
    data = fuse_tiles(tiles, tmp_path / "o.zarr", blend="max", progress=False).data[0].compute()
    assert data.shape == (3, 4, 8, 12)
    assert (data[2, ..., :8] == 30).all() and (data[2, ..., 8:] == 0).all()
    assert (data[1, ..., 8:] == 21).all()


def test_tile_without_reference_counterpart_is_not_fused(tmp_path, caplog):
    tiles = _series(2) + [_tile(99, 500, 1, stack="extra")]
    with caplog.at_level(logging.WARNING, logger="zarrsmith"):
        plan = plan_fusion(tiles)
    assert plan.shape == (2, 1, 4, 8, 12) and not any(t.attrs["stack"] == "extra" for t in plan.tiles)
    assert "1 tile(s) have no counterpart at reference timepoint 0" in caplog.text


def test_tile_of_another_shape_is_refused():
    tiles = _series(2)
    tiles[3] = _tile(1, 4, 1, stack="s1", shape=(5, 8, 8))
    with pytest.raises(ValueError, match=r"'s1'.*timepoint 1.*does not match reference timepoint 0.*timepoints="):
        plan_fusion(tiles)
    assert plan_fusion(tiles, timepoints=[0]).shape == (1, 4, 8, 12)  # the way out the message gives


def test_other_voxel_size_or_unit_is_refused():
    tiles = _series(2)
    tiles[3] = Tile(np.zeros((4, 8, 8), np.uint16), (0, 0, 4), (2.0, 1.0, 1.0), "A", None, 1, {"stack": "s1"})
    with pytest.raises(ValueError, match="does not match reference"):
        plan_fusion(tiles)
    tiles[3] = Tile(np.zeros((4, 8, 8), np.uint16), (0, 0, 4), (1.0, 1.0, 1.0), "A", None, 1, {"stack": "s1"}, units="mm")
    with pytest.raises(ValueError, match="does not match reference"):
        plan_fusion(tiles)


def test_skip_by_id_or_tile(tmp_path):
    m = _mosaic(_series(2))
    # an id blanks the position at every timepoint; the output keeps its size
    plan = plan_fusion(m.tiles, skip=m.tiles_by_ids(["s1"]))
    assert plan.n_skipped == 2 and plan.shape == (2, 1, 4, 8, 12) and len(plan.tiles) == 2
    data = m.fuse(tmp_path / "id.zarr", skip=["s1"], blend="max", progress=False).data[0].compute()
    assert (data[:, ..., 8:] == 0).all() and (data[:, ..., :8] > 0).all()
    # a Tile blanks only its own timepoint
    data = m.fuse(tmp_path / "tile.zarr", skip=[m.tiles[3]], blend="max", progress=False).data[0].compute()
    assert (data[1, ..., 8:] == 0).all() and (data[0, ..., 8:] == 11).all()
    # skipping the reference tile keeps the geometry and the other timepoints
    data = m.fuse(tmp_path / "ref.zarr", skip=[m.tiles[1]], blend="max", progress=False).data[0].compute()
    assert (data[0, ..., 8:] == 0).all() and (data[1, ..., 8:] == 21).all()
    with pytest.raises(ValueError, match="Every tile is skipped"):
        plan_fusion(m.tiles, skip=m.tiles)


def test_fields_and_channels_over_time(tmp_path):
    tiles = [_tile(10 * (t + 1) + i, i * 100, t, c, stack=f"s{i}")
             for t in range(2) for i in range(2) for c in ("A", "B")]
    m = _mosaic(tiles)
    assert m.fields == ["field0", "field1"]
    zm = m.fuse(tmp_path / "f1.zarr", field="field1", channels=["B"], blend="max", progress=False)
    assert zm.metadata["axes"] == "tzyx" and zm.data[0].shape == (2, 4, 8, 8)
    assert int(zm.data[0][0].max()) == 11 and int(zm.data[0][1].max()) == 21


def test_display_window_uses_every_timepoint(tmp_path):
    tiles = [_tile(100, 0, 0), _tile(10000, 0, 1)]
    out = tmp_path / "o.zarr"
    fuse_tiles(tiles, out, blend="max", progress=False)
    (low, high), = fusion.display_windows(out)
    assert low < 1000 and high > 5000


def test_chunks_are_given_in_output_axis_order(tmp_path):
    tiles = _series(2)
    zm = fuse_tiles(tiles, tmp_path / "o.zarr", chunks=(1, 1, 2, 4, 4), shards=None, drop_singleton=False, progress=False)
    assert zm.data[0].chunksize == (1, 1, 2, 4, 4)
    with pytest.raises(ValueError, match="chunks must have 5 values"):
        fuse_tiles(tiles, tmp_path / "p.zarr", chunks=(1, 2, 4, 4), progress=False)
    zm = fuse_tiles(tiles, tmp_path / "q.zarr", projection="max", drop_singleton=False, progress=False)
    assert zm.metadata["axes"] == "tcyx"


def test_regions_are_built_lazily_and_counted():
    tiles = _series(2)
    del tiles[3]
    plan = plan_fusion(tiles)
    placed = plan.layout["A"]
    blocks = [(0, "A", (0, 8, 0, 8), (0, 4), None, None, [0]), (0, "A", (0, 8, 8, 12), (0, 4), None, None, [1])]
    regions = fusion._RegionSeries(blocks, plan)
    assert len(regions) == 3 == len(list(regions))  # the second tile is missing at timepoint 1
    assert [r.time_index for r in regions] == [0, 0, 1]
    assert list(regions)[0].placed[0].offset == placed[0].offset
    assert list(regions)[2].placed[0].tile.timepoint == 1


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def _luxendo_timelapse(root, layout, n_timepoints=3, time_point=True, shape=(4, 8, 6)):
    raw = root / "raw"
    for t in range(n_timepoints):
        for stack, tx in (("1-x00-y00", 0.0), ("1-x01-y00", 8.0)):
            tp = t if time_point else None
            value = 100 * (t + 1) + int(tx)
            if layout == "one folder":
                _write_tile(raw / f"stack_{stack}_GFP-L", f"Cam_bottom_{t:05d}", "GFP-L", stack, tx, 0.0, 0.0,
                            shape=shape, value=value, time_point=tp)
            else:
                _write_tile(raw / f"stack_{stack}_GFP-L_tp{t}", "Cam_bottom_00000", "GFP-L", stack, tx, 0.0, 0.0,
                            shape=shape, value=value, time_point=tp)
    return root


@pytest.mark.parametrize("layout,time_point", [("one folder", True), ("one folder", False), ("folder per timepoint", True)])
def test_luxendo_timelapse(tmp_path, layout, time_point):
    m = LuxendoMosaicManager(_luxendo_timelapse(tmp_path / "acq", layout, time_point=time_point), progress=False)
    assert m.timepoints == [0, 1, 2] and len(m.tiles) == 6 and len(m.positions) == 2
    assert m.fields == ["field0"]
    zm = m.fuse(tmp_path / "o.zarr", blend="max", progress=False, workers=2, executor="thread")
    assert zm.metadata["axes"] == "tzyx"
    data = zm.data[0].compute()
    assert data.shape[0] == 3
    assert [int(data[t, 0, 0, 0]) for t in range(3)] == [100, 200, 300]
    assert [int(data[t].max()) for t in range(3)] == [108, 208, 308]


def test_luxendo_timelapse_in_worker_processes(tmp_path):
    m = LuxendoMosaicManager(_luxendo_timelapse(tmp_path / "acq", "one folder"), progress=False)
    serial = m.fuse(tmp_path / "a.zarr", blend="sigmoid", progress=False).data[0].compute()
    procs = m.fuse(tmp_path / "b.zarr", blend="sigmoid", progress=False, workers=2, executor="process").data[0].compute()
    np.testing.assert_array_equal(serial, procs)


def test_opera_timelapse_with_autofocus_drift(tmp_path):
    from .test_opera_mosaic import NX, NY, write_export

    from zarrsmith.mosaic_manager import OperaMosaicManager

    root = write_export(tmp_path / "plate", timepoints=(1, 2, 3), z_drift=1e-6)
    m = OperaMosaicManager(root, flatfield=False, progress=False)
    assert m.timepoints == [0, 1, 2]
    origins = {t.timepoint: t.origin[0] for t in m.select(channel="EGFP", stack="r01c01f01")}
    assert origins == {0: -2.0, 1: -1.0, 2: 0.0}  # the stage reports a different z at every timepoint
    # labelled once per position, per well
    assert set(m.fields) == {"A1_field0", "A1_field1", "B1_field0"}
    assert {t.field for t in m.select(stack="r01c01f01")} == {"A1_field0"}

    zm = m.fuse(tmp_path / "o.zarr", field="B1_field0", channels=["EGFP"], blend="max", progress=False)
    assert zm.metadata["axes"] == "tzyx"
    data = zm.data[0].compute()
    assert data.shape == (3, 3, NY, NX)  # all planes at every timepoint, despite the drift in z
    assert [int(data[t, 0, 0, 0]) - int(data[0, 0, 0, 0]) for t in range(3)] == [0, 100, 200]


# ---------------------------------------------------------------------------
# napari
# ---------------------------------------------------------------------------

def _napari():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    return ViewerModel


def test_visualize_scrolls_through_time_from_the_first_timepoint():
    ViewerModel = _napari()
    m = _mosaic(_series(5, drift=1.0))
    v = ViewerModel()
    m.visualize(viewer=v)  # the first layer of a viewer: napari starts it in the middle of every axis
    assert len(v.layers) == 2
    layer = v.layers[1]
    assert layer.data.shape == (5, 4, 8, 12)  # the tile moves 4 voxels over time
    assert v.dims.ndim == 4 and v.dims.axis_labels[0] == "T"
    assert v.dims.current_step[0] == 0
    assert v.dims.nsteps[0] == 5
    assert tuple(layer.translate) == (0.0, 0.0, 0.0, 4.0)  # where the tile is furthest left
    assert tuple(layer.scale) == (1.0, 1.0, 1.0, 1.0)
    assert all(lay.visible for lay in v.layers)
    assert layer.metadata["mosaic_tile_id"] == "A/s1"

    def shown():
        return int(np.asarray(layer._slice.image.raw).max())

    assert shown() == 11
    v.dims.set_current_step(0, 3)
    assert shown() == 41
    v.dims.set_current_step(0, 0)
    assert shown() == 11


def test_visualize_blank_where_a_timepoint_has_no_tile():
    ViewerModel = _napari()
    tiles = _series(3)
    del tiles[3]  # second tile, timepoint 1
    tiles.append(_tile(7, 4, 2, stack="other", shape=(2, 8, 8)))
    tiles[4] = _tile(9, 4, 2, stack="s1", shape=(2, 8, 8))  # another shape
    m = _mosaic(tiles)
    v = ViewerModel()
    m.visualize(tiles=[t for t in m.tiles if t.attrs["stack"] != "other"], viewer=v)
    data = v.layers[1].data
    assert data.shape == (3, 4, 8, 8)
    assert int(data[0].max()) == 11
    assert int(data[1].max()) == 0 and int(data[2].max()) == 0


def test_visualize_one_timepoint_of_a_timelapse_has_no_slider():
    ViewerModel = _napari()
    m = _mosaic(_series(3))
    v = ViewerModel()
    m.visualize(tiles=m.select(timepoint=1), viewer=v)
    assert v.dims.ndim == 3 and v.layers[0].data.shape == (4, 8, 8)
    assert int(v.layers[0].data.max()) == 20
    # "auto" samples every position once
    assert m.contrast_limits()["A"][1] <= 11


def test_visualize_reads_ahead_the_tiles_of_the_first_timepoint(tmp_path, monkeypatch):
    ViewerModel = _napari()
    from zarrsmith.mosaic_manager import TiffStackArray

    read = []
    monkeypatch.setattr(TiffStackArray, "prefetch", lambda self, z: read.append(self.label))
    tiles = [Tile(TiffStackArray([None, None], (2, 8, 8), np.uint16), (0, 0, 0), (1, 1, 1), "A", None, t, {"stack": "s"})
             for t in range(3)]
    for t in tiles:
        t.data.label = f"t{t.timepoint}"
    m = _mosaic(tiles)
    v = ViewerModel()
    v.add_image(np.zeros((8, 8)))  # the 2D view of the plugin, with something already shown
    m.visualize(viewer=v)
    assert read == ["t0"]


def _frame_box(frame):
    """``(start, stop)`` per axis of the non-zero voxels of one frame."""
    return [(int(np.flatnonzero(np.asarray(frame).any(axis=other))[0]),
             int(np.flatnonzero(np.asarray(frame).any(axis=other))[-1]) + 1)
            for other in ((1, 2), (0, 2), (0, 1))]


def test_visualize_shows_every_timepoint_at_its_own_position():
    ViewerModel = _napari()
    tiles = [_tile(10, 5, 0, stack="s", z=2, y=1),   # x 5, y 1, z 2
             _tile(20, 8, 1, stack="s", z=0, y=3),   # moves right, down and to a lower z
             _tile(30, 2, 2, stack="s", z=1, y=1)]   # the furthest left: it sets the layer's x
    v = ViewerModel()
    _mosaic(tiles).visualize(viewer=v)
    layer = v.layers[0]
    assert tuple(layer.translate) == (0.0, 0.0, 1.0, 2.0)
    assert layer.data.shape == (3, 4 + 2, 8 + 2, 8 + 6)
    data = layer.data.compute()
    assert data.dtype == np.uint16
    # in the layer, relative to its translate
    assert _frame_box(data[0]) == [(2, 6), (0, 8), (3, 11)] and (data[0][data[0] > 0] == 10).all()
    assert _frame_box(data[1]) == [(0, 4), (2, 10), (6, 14)] and (data[1][data[1] > 0] == 20).all()
    assert _frame_box(data[2]) == [(1, 5), (0, 8), (0, 8)] and (data[2][data[2] > 0] == 30).all()


def test_visualize_tile_that_does_not_move_is_not_padded():
    ViewerModel = _napari()
    tiles = [_tile(10, 4.3, t, stack="s", z=1.2) for t in range(3)]
    tiles[2] = _tile(30, 4.5, 2, stack="s", z=1.2)  # less than half a voxel away
    v = ViewerModel()
    _mosaic(tiles).visualize(viewer=v)
    assert v.layers[0].data.shape == (3, 4, 8, 8)
    assert tuple(v.layers[0].translate) == pytest.approx((0.0, 1.2, 0.0, 4.3))


def test_visualize_moving_tile_reads_a_plane_once_and_blank_areas_never(tmp_path):
    from zarrsmith.mosaic_manager.tiff_array import PlaneStackArray

    class Counting(PlaneStackArray):
        def __init__(self, label):
            super().__init__((3, 8, 8), np.uint16)
            self.label = label  # also keeps dask from taking the stacks for one another
            self.reads = []

        def _read(self, z):
            self.reads.append(z)
            return np.full(self.shape[1:], 5, np.uint16)

    stacks = [Counting(t) for t in range(2)]
    tiles = [Tile(stacks[0], (2, 0, 3), (1, 1, 1), "A", None, 0, {"stack": "s"}),
             Tile(stacks[1], (0, 2, 0), (1, 1, 1), "A", None, 1, {"stack": "s"})]
    data, origin, starts = MosaicManager._lazy_series({t.timepoint: t for t in tiles}, [0, 1])
    assert data.shape == (2, 5, 10, 11) and origin == (0.0, 0.0, 0.0)
    assert starts == {0: (2, 0, 3), 1: (0, 2, 0)}
    assert (np.asarray(data[0, 3])[0:8, 3:11] == 5).all()
    assert stacks[0].reads == [1] and stacks[1].reads == []  # plane 3 of the layer is plane 1 of the tile
    assert (np.asarray(data[0, 0]) == 0).all()  # below the tile at this timepoint
    assert stacks[0].reads == [1]


@pytest.mark.parametrize("z_start, expected", [(0.0, [("t0", 0)]), (2.0, [])])
def test_visualize_reads_ahead_the_plane_of_the_tile_at_its_position(monkeypatch, z_start, expected):
    ViewerModel = _napari()
    from zarrsmith.mosaic_manager import TiffStackArray

    read = []
    monkeypatch.setattr(TiffStackArray, "prefetch", lambda self, z: read.append((self.label, z)))
    tiles = [Tile(TiffStackArray([None, None], (2, 8, 8), np.uint16), (z, 0, 0), (1, 1, 1), "A", None, t, {"stack": "s"})
             for t, z in enumerate((z_start, 2.0 - z_start))]
    for t in tiles:
        t.data.label = f"t{t.timepoint}"
    v = ViewerModel()
    v.add_image(np.zeros((8, 8)))
    _mosaic(tiles).visualize(viewer=v)
    assert v.layers[1].data.shape == (2, 4, 8, 8)
    assert read == expected


def _dock(monkeypatch, viewer):
    from zarrsmith.napari import _mosaic_widget as mw

    monkeypatch.setattr(mw, "current_viewer", lambda: viewer)
    return mw, mw.mosaic_widget()


@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_widget_timepoint_options(app, monkeypatch, tmp_path):
    ViewerModel = _napari()
    mw, dock = _dock(monkeypatch, ViewerModel())
    fw = dock.fuse_widget
    assert fw.timepoints.native.isHidden() and fw.reference_timepoint.native.isHidden()

    # a single timepoint: nothing to choose, and the summary stays as it was
    m = _mosaic([_tile(1, 0, 0, stack="s0")])
    dock.state["mosaic"] = m
    dock.populate(m, tmp_path)
    assert fw.timepoints.native.isHidden()
    assert "t x" not in dock.plan_label.text() and "(czyx)" in dock.plan_label.text()

    m = _mosaic(_series(3))
    dock.state["mosaic"] = m
    dock.populate(m, tmp_path)
    assert not fw.timepoints.native.isHidden() and not fw.reference_timepoint.native.isHidden()
    assert list(fw.timepoints.value) == [0, 1, 2] and fw.reference_timepoint.value == 0
    assert "reference timepoint 0" in dock.plan_label.text()
    fw.ds_z.value, fw.ds_y.value, fw.ds_x.value = 1, 1, 1
    assert "3 t x 1 ch x 4 x 8 x 12 px (tczyx)" in dock.plan_label.text()

    fw.timepoints.value = [0, 2]
    assert "2 t x 1 ch" in dock.plan_label.text()
    fw.reference_timepoint.value = 2
    assert "reference timepoint 2" in dock.plan_label.text()
    fw.timepoints.value = [1]
    assert "1 ch x 4 x 8 x 12 px (czyx)" in dock.plan_label.text()
    fw.timepoints.value = []
    assert "No timepoint selected" in dock.plan_label.text()


def test_widget_shows_timepoint_zero_and_keeps_the_tile_map(app, monkeypatch, tmp_path, capsys):
    ViewerModel = _napari()
    viewer = ViewerModel()
    mw, dock = _dock(monkeypatch, viewer)
    m = _mosaic(_series(4, drift=1.0))
    dock.state["mosaic"] = m
    dock.populate(m, tmp_path)
    tile_map = mw.find_tile_map(viewer.layers, m.path)
    assert len(tile_map.data) == 2
    np.testing.assert_array_equal(mw.tile_footprints(m)["s1"], [[0, 4], [8, 12]])  # not widened by drift

    dock.show_widget()
    layers = mw.tile_layers(viewer.layers, m.path)
    assert len(layers) == 2 and layers[0].data.shape == (4, 4, 8, 11)
    assert viewer.dims.current_step[0] == 0 and viewer.dims.nsteps[0] == 4
    assert tile_map.visible and viewer.layers[-1] is tile_map

    # showing again after the slider moved starts over, with the same layers
    viewer.dims.set_current_step(0, 2)
    dock.show_widget()
    assert len(mw.tile_layers(viewer.layers, m.path)) == 2
    assert viewer.dims.current_step[0] == 0


def test_widget_fuses_with_reference_and_keeps_the_output_size(app, monkeypatch, tmp_path):
    """Skipping a tile on the edge, or leaving out the reference, must not change the geometry."""
    ViewerModel = _napari()
    viewer = ViewerModel()
    mw, dock = _dock(monkeypatch, viewer)
    calls = []

    class Now:
        """Runs the fusion when started, in this thread."""

        def __init__(self, tiles, out, kwargs):
            self.args = tiles, out, kwargs
            self.returned = self.errored = self

        def connect(self, f):
            self.done = f if f.__name__ == "_done" else getattr(self, "done", None)
            return f

        def start(self):
            tiles, out, kwargs = self.args
            calls.append(fuse_tiles(tiles, out, progress=False, **kwargs))
            self.done(calls[-1])

    monkeypatch.setattr(mw, "_fuse_worker", Now)
    m = _mosaic(_series(3, drift=2.0))
    dock.state["mosaic"] = m
    dock.populate(m, tmp_path)
    fw = dock.fuse_widget
    fw.ds_z.value, fw.ds_y.value, fw.ds_x.value = 1, 1, 1
    fw.blend.value = "max"
    fw.levels.value = 1
    fw.workers.value = 1
    fw.open_result.value = True
    m.visualize(viewer=viewer, contrast_limits={"A": (0, 100)})
    layers = mw.tile_layers(viewer.layers, m.path)

    # the second tile (right edge) is hidden and left blank; timepoint 0 (the reference) is not fused
    fw.tiles_to_fuse.value = "visible layers only"
    layers[1].visible = False
    fw.timepoints.value = [1, 2]
    fw.output_path.value = tmp_path / "out.ome.zarr"
    fw()

    zm = calls[-1]
    assert zm.metadata["axes"] == "tzyx" and zm.data[0].shape == (2, 4, 8, 12)
    data = zm.data[0].compute()
    assert (data[:, ..., 8:] == 0).all() and int(data[0].max()) == 20 and int(data[1].max()) == 30
    # the result opens on its first timepoint
    assert viewer.dims.current_step[-4] == 0

    # a single timepoint: a hidden tile on the edge leaves the output at its full size
    viewer.layers.clear()
    m = _mosaic([_tile(1, 0, 0, stack="s0"), _tile(2, 8, 0, stack="s1")])
    dock.state["mosaic"] = m
    dock.populate(m, tmp_path)
    fw.ds_z.value, fw.ds_y.value, fw.ds_x.value = 1, 1, 1
    fw.open_result.value = False
    m.visualize(viewer=viewer, contrast_limits={"A": (0, 100)})
    fw.tiles_to_fuse.value = "visible layers only"
    mw.tile_layers(viewer.layers, m.path)[1].visible = False
    fw.output_path.value = tmp_path / "single.ome.zarr"
    fw()
    assert calls[-1].data[0].shape == (4, 8, 16)


def test_batch_row_timepoints():
    _napari()
    from zarrsmith.napari import _mosaic_widget as mw

    values = dict(channels=["A"], groups=["L"], projection="none (3D)", blend="sigmoid", ds_z=1, ds_y=1, ds_x=1,
                  levels=1, workers=1, overwrite=False)
    common = dict(input_path="acq", microscope="luxendo", output="o.zarr", all_groups=["L"], skip_ids=[])
    row = mw.batch_row(values=values, **common)
    assert row["timepoints"] == "" and row["reference_timepoint"] == ""
    row = mw.batch_row(values=dict(values, timepoints=[0, 1, 2], reference_timepoint=1),
                       all_timepoints=[0, 1, 2], **common)
    assert row["timepoints"] == "" and row["reference_timepoint"] == "1"
    row = mw.batch_row(values=dict(values, timepoints=[0, 2], reference_timepoint=0),
                       all_timepoints=[0, 1, 2], **common)
    assert row["timepoints"] == "0 2" and row["reference_timepoint"] == "0"
    row = mw.batch_row(values=dict(values, timepoints=[0, 1, 2], reference_timepoint="all"),
                       all_timepoints=[0, 1, 2], **common)
    assert row["reference_timepoint"] == "all"
    assert {"timepoints", "reference_timepoint"} <= set(mw.MOSAIC_BATCH_COLUMNS)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["zarrsmith", *map(str, argv)])
    cli.main()


@pytest.fixture
def timelapse(tmp_path):
    return _luxendo_timelapse(tmp_path / "acq", "one folder")


def test_fuse_all_timepoints(timelapse, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out.ome.zarr"
    _run(monkeypatch, "fuse", "-i", timelapse, "-z", out, "-b", "max", "-rw", 1, "-nl", 2, "-df", 1, 2, 2)
    text = capsys.readouterr().out
    assert "3 timepoints: [0, 1, 2]" in text and "reference timepoint 0" in text
    z = mm.ZarrManager(str(out))
    assert z.metadata["axes"] == "tzyx" and z.data[0].shape[0] == 3 and len(z.data) == 2
    assert z.data[1].shape[:2] == (3, 4)  # factor 1 in z; time untouched


def test_fuse_timepoints_and_reference(timelapse, tmp_path, monkeypatch):
    out = tmp_path / "out.ome.zarr"
    _run(monkeypatch, "fuse", "-i", timelapse, "-z", out, "-b", "max", "-rw", 1, "-tp", 2, 1, "-rtp", 1, "-cs", 1, 1, 2, 4, 4,
         "-sh", "none")
    z = mm.ZarrManager(str(out))
    assert z.data[0].shape[0] == 2 and z.data[0].chunksize == (1, 2, 4, 4)
    assert z.root.attrs[fusion.PROVENANCE_KEY] == {"timepoints": [1, 2], "reference_timepoint": 1}
    assert int(z.data[0][0, 0, 0, 0].compute()) == 200


def test_fuse_refuses_t_in_subset_and_unknown_timepoints(timelapse, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--timepoints"):
        _run(monkeypatch, "fuse", "-i", timelapse, "-z", tmp_path / "a.zarr", "-s", "t=0:1")
    with pytest.raises(ValueError, match=r"Timepoints \[7\] not found"):
        _run(monkeypatch, "fuse", "-i", timelapse, "-z", tmp_path / "b.zarr", "-tp", 7)


def test_fuse_batch_timepoints(timelapse, tmp_path, monkeypatch):
    # empty cells make pandas read the numbers as floats (2.0)
    csv = tmp_path / "batch.csv"
    pd.DataFrame([
        {"input": str(timelapse), "output": str(tmp_path / "a.zarr"), "blend": "max", "read_workers": 1,
         "timepoints": "0 2", "reference_timepoint": 2},
        {"input": str(timelapse), "output": str(tmp_path / "b.zarr"), "blend": "max", "read_workers": 1,
         "timepoints": None, "reference_timepoint": None},
    ]).to_csv(csv, index=False)
    _run(monkeypatch, "fuse", "--batch", csv)
    assert zarr.open_group(str(tmp_path / "a.zarr"), mode="r").attrs[fusion.PROVENANCE_KEY] == {
        "timepoints": [0, 2], "reference_timepoint": 2}
    assert mm.ZarrManager(str(tmp_path / "b.zarr")).data[0].shape[0] == 3

    csv2 = tmp_path / "batch2.csv"
    pd.DataFrame([
        {"input": str(timelapse), "output": str(tmp_path / "c.zarr"), "blend": "max", "read_workers": 1, "timepoints": 1},
        {"input": str(timelapse), "output": str(tmp_path / "d.zarr"), "blend": "max", "read_workers": 1, "timepoints": None},
    ]).to_csv(csv2, index=False)
    _run(monkeypatch, "fuse", "--batch", csv2)
    assert mm.ZarrManager(str(tmp_path / "c.zarr")).metadata["axes"] == "zyx"
    assert Path(tmp_path / "d.zarr").exists()


# ---------------------------------------------------------------------------
# reference_timepoint="all": every timepoint at its own positions
# ---------------------------------------------------------------------------

def _moving_tiles():
    """Two tiles whose overlap changes: the second one moves 3 voxels right at timepoint 1."""
    return [_tile(100, 0, 0, stack="s0"), _tile(200, 4, 0, stack="s1"),
            _tile(100, 0, 1, stack="s0"), _tile(200, 7, 1, stack="s1")]


def test_own_positions_place_every_timepoint(tmp_path):
    zm = fuse_tiles(_moving_tiles(), tmp_path / "o.zarr", blend="max", reference_timepoint="all", progress=False)
    data = zm.data[0].compute()
    assert data.shape == (2, 4, 8, 15)  # the grid covers both timepoints
    assert (data[0, ..., :4] == 100).all() and (data[0, ..., 4:12] == 200).all() and (data[0, ..., 12:] == 0).all()
    assert (data[1, ..., :7] == 100).all() and (data[1, ..., 7:15] == 200).all()
    assert zm.root.attrs[fusion.PROVENANCE_KEY] == {"timepoints": [0, 1], "reference_timepoint": "all"}


def test_own_positions_summary_and_invalid_value():
    plan = plan_fusion(_moving_tiles(), reference_timepoint="all")
    assert plan.per_timepoint and "own positions per timepoint" in plan.summary()
    assert not plan_fusion(_moving_tiles()).per_timepoint
    with pytest.raises(ValueError, match="must be a timepoint or 'all'"):
        plan_fusion(_moving_tiles(), reference_timepoint="every")


@pytest.mark.parametrize("blend", ["sigmoid", "midline", "max", "sum"])
def test_own_positions_match_a_fusion_of_that_timepoint_alone(tmp_path, blend):
    """Transitions follow each timepoint: a frame equals that timepoint fused by itself, on the same grid."""
    tiles = [_tile(100, 0, 0, group="L", stack="l"), _tile(200, 4, 0, group="R", stack="r"),
             _tile(100, 0, 1, group="L", stack="l"), _tile(200, 7, 1, group="R", stack="r")]
    data = fuse_tiles(tiles, tmp_path / "all.zarr", blend=blend, reference_timepoint="all", progress=False).data[0].compute()
    for t in (0, 1):
        alone = [x for x in tiles if x.timepoint == t]
        frame = fuse_tiles(alone, tmp_path / f"t{t}.zarr", blend=blend, progress=False).data[0].compute()
        np.testing.assert_array_equal(data[t][..., :frame.shape[-1]], frame)
        assert (data[t][..., frame.shape[-1]:] == 0).all()


def test_own_positions_parts_line_up(tmp_path):
    kw = dict(blend="max", reference_timepoint="all", progress=False)
    whole = fuse_tiles(_moving_tiles(), tmp_path / "w.zarr", **kw).data[0].compute()
    part = fuse_tiles(_moving_tiles(), tmp_path / "p.zarr", timepoints=[1], **kw)
    assert part.metadata["axes"] == "zyx"
    np.testing.assert_array_equal(part.data[0].compute(), whole[1])
    assert part.root.attrs[fusion.PROVENANCE_KEY] == {"timepoints": [1], "reference_timepoint": "all"}


def test_own_positions_crop_is_on_the_common_grid(tmp_path):
    kw = dict(blend="max", reference_timepoint="all", progress=False)
    whole = fuse_tiles(_moving_tiles(), tmp_path / "w.zarr", **kw).data[0].compute()
    # x 12:15 only holds the second tile at timepoint 1: timepoint 0 stays blank there
    crop = fuse_tiles(_moving_tiles(), tmp_path / "c.zarr", crop={"x": slice(12, 15)}, **kw).data[0].compute()
    np.testing.assert_array_equal(crop, whole[..., 12:15])
    assert (crop[0] == 0).all() and (crop[1] == 200).all()
    with pytest.raises(ValueError, match="No tile intersects"):
        plan_fusion(_moving_tiles(), crop={"x": slice(12, 15)}, timepoints=[0], reference_timepoint="all")


def test_own_positions_missing_tile_stays_blank(tmp_path, caplog):
    tiles = _moving_tiles()
    del tiles[3]  # no second tile at timepoint 1
    with caplog.at_level(logging.WARNING, logger="zarrsmith"):
        plan = plan_fusion(tiles, reference_timepoint="all")
    assert plan.n_missing == 1 and "missing at some timepoints" in caplog.text
    data = fuse_tiles(tiles, tmp_path / "o.zarr", blend="max", reference_timepoint="all", progress=False).data[0].compute()
    assert (data[1, ..., :8] == 100).all() and (data[1, ..., 8:] == 0).all()


def test_own_positions_allow_another_shape_but_not_another_voxel_size(tmp_path):
    tiles = [_tile(100, 0, 0, stack="s"), _tile(200, 0, 1, stack="s", shape=(4, 8, 12))]
    with pytest.raises(ValueError, match="does not match reference timepoint"):
        plan_fusion(tiles)
    data = fuse_tiles(tiles, tmp_path / "o.zarr", blend="max", reference_timepoint="all", progress=False).data[0].compute()
    assert data.shape == (2, 4, 8, 12)
    assert (data[0, ..., 8:] == 0).all() and (data[1] == 200).all()
    other = Tile(np.ones((4, 8, 8), np.uint16), (0, 0, 0), (2.0, 1.0, 1.0), "A", None, 1, {"stack": "s"})
    with pytest.raises(ValueError, match="share one voxel size"):
        plan_fusion([_tile(100, 0, 0, stack="s"), other], reference_timepoint="all")


def test_own_positions_skip_by_id_and_by_tile(tmp_path):
    kw = dict(blend="max", reference_timepoint="all", progress=False)
    m = _mosaic(_moving_tiles())
    by_id = m.fuse(tmp_path / "i.zarr", skip=["s1"], **kw).data[0].compute()
    assert by_id.shape == (2, 4, 8, 15)  # the output keeps its size
    assert (by_id[..., :8] == 100).all() and (by_id[..., 8:] == 0).all()
    second = [t for t in m.tiles if t.attrs["stack"] == "s1" and t.timepoint == 1]
    one = m.fuse(tmp_path / "t.zarr", skip=second, **kw).data[0].compute()  # only timepoint 1
    assert (one[0, ..., 4:12] == 200).all() and (one[1, ..., 8:] == 0).all()


def test_own_positions_projection_with_worker_threads(tmp_path):
    kw = dict(blend="max", reference_timepoint="all", progress=False)
    single = fuse_tiles(_moving_tiles(), tmp_path / "s.zarr", projection="max", **kw).data[0].compute()
    assert single.shape == (2, 8, 15)
    threads = fuse_tiles(_moving_tiles(), tmp_path / "t.zarr", projection="max", workers=2, executor="thread", **kw)
    np.testing.assert_array_equal(threads.data[0].compute(), single)


def test_own_positions_in_worker_processes(tmp_path):
    acq = _luxendo_timelapse(tmp_path / "acq", "one folder")
    kw = dict(blend="max", reference_timepoint="all", progress=False)
    m = LuxendoMosaicManager(acq)
    one = m.fuse(tmp_path / "a.zarr", **kw).data[0].compute()
    two = m.fuse(tmp_path / "b.zarr", workers=2, executor="process", **kw).data[0].compute()
    np.testing.assert_array_equal(one, two)


def test_cli_own_positions(timelapse, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out.ome.zarr"
    _run(monkeypatch, "fuse", "-i", timelapse, "-z", out, "-b", "max", "-rw", 1, "-rtp", "all")
    assert "own positions per timepoint" in capsys.readouterr().out
    assert mm.ZarrManager(str(out)).root.attrs[fusion.PROVENANCE_KEY]["reference_timepoint"] == "all"
    with pytest.raises(SystemExit):
        _run(monkeypatch, "fuse", "-i", timelapse, "-z", tmp_path / "x.zarr", "-rtp", "every")


def test_batch_own_positions(timelapse, tmp_path, monkeypatch):
    csv = tmp_path / "batch.csv"
    pd.DataFrame([
        {"input": str(timelapse), "output": str(tmp_path / "a.zarr"), "blend": "max", "read_workers": 1,
         "reference_timepoint": "all"},
        {"input": str(timelapse), "output": str(tmp_path / "b.zarr"), "blend": "max", "read_workers": 1,
         "reference_timepoint": 2},
    ]).to_csv(csv, index=False)
    _run(monkeypatch, "fuse", "--batch", csv)
    for name, reference in (("a.zarr", "all"), ("b.zarr", 2)):
        record = zarr.open_group(str(tmp_path / name), mode="r").attrs[fusion.PROVENANCE_KEY]
        assert record["reference_timepoint"] == reference


def test_widget_own_positions(app, monkeypatch, tmp_path):
    ViewerModel = _napari()
    mw, dock = _dock(monkeypatch, ViewerModel())
    fw = dock.fuse_widget
    m = _mosaic(_moving_tiles())
    dock.state["mosaic"] = m
    dock.populate(m, tmp_path)
    assert fw.reference_timepoint.value == 0  # still the default
    fw.ds_z.value, fw.ds_y.value, fw.ds_x.value = 1, 1, 1
    fw.reference_timepoint.value = "all"
    assert "2 t x 1 ch x 4 x 8 x 15 px (tczyx)" in dock.plan_label.text()
    assert "own positions per timepoint" in dock.plan_label.text()
