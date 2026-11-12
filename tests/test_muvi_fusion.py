"""MuVi SPIM (Luxendo, two opposing cameras): reading, camera fusion and the transforms file."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from zarrsmith.cli.zarrsmith import main as cli_main
from zarrsmith.microscope_manager.zarr_manager import ZarrManager
from zarrsmith.mosaic_manager import (LuxendoMosaicManager, default_transforms_path, load_transforms, plan_fusion)

SHAPE = (8, 16, 12)  # z, y, x of one tile
VOX = (2.0, 0.5, 0.5)  # z, y, x
ROT45 = [[0.70710677, 0.0, 0.70710677], [0.0, 1.0, 0.0], [-0.70710677, 0.0, 0.70710677]]


def _write_view(raw, stack, channel, camera, data, ty, tz=-8.0, tx=0.0, rotation=45.0):
    """One MuVi view as Luxendo writes it: the right camera has a negative x scale and
    looks the other way (detection direction), both share the stage rotation."""
    folder = raw / f"stack_{stack}_channel_{channel}_obj_{camera}"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"Cam_{camera}_00000"
    with h5py.File(folder / f"{name}.lux.h5", "w") as f:
        f["Data"] = data
    sign = 1.0 if camera == "left" else -1.0
    z, y, x = data.shape
    info = {
        "sources": ["Luxendo MuVi-SPIM, Embedded v3.17.3"],
        "time_point": "0",
        "channel": channel.split("-")[0],
        "channel_description": channel.split("-", 1)[1],
        "stack": stack,
        "objective": camera,
        "camera": camera,
        "voxel_size_um": {"width": VOX[2], "height": VOX[1], "depth": VOX[0]},
        "image_size_vx": {"width": x, "height": y, "depth": z},
        "affine_to_sample": [
            {"matrix": np.eye(3).tolist(), "translation": [-(x - 1) / 2, -(y - 1) / 2, 0.0]},
            {"matrix": [[sign * VOX[2], 0, 0], [0, VOX[1], 0], [0, 0, VOX[0]]], "translation": [tx, ty, tz]},
            {"matrix": ROT45, "translation": [-6.7, 0.0, 125.7]},
        ],
        # sample frame: the stage rotation applied to +z (left) or -z (right)
        "detection_directions": [[sign * 0.7071068, 0.0, sign * 0.7071068]],
        "acquisition": [{"microscope_type": "MuVi-SPIM", "stage_positions": [
            {"type": "rotation", "name": "r", "start_deg": rotation, "end_deg": rotation}]}],
    }
    (folder / f"{name}.json").write_text(json.dumps({"processingInformation": info}))


def _sample(seed=0, shape=SHAPE):
    return np.random.default_rng(seed).integers(100, 1000, shape).astype(np.uint16)


def _muvi(tmp_path, left=None, right=None):
    """Stack 0 with two tiles (x00, x01: 4 voxels apart in y) and stack 1 with one,
    each seen by both cameras; the right view is the mirror image of the left one."""
    raw = tmp_path / "acq" / "raw"
    left = _sample() if left is None else left
    for stack, ty in (("0-x00-y00", 0.0), ("0-x01-y00", 2.0), ("1", 500.0)):
        for channel in ("2-DAPI", "0-GFP"):
            view = left if right is None else left
            _write_view(raw, stack, channel, "left", view, ty)
            _write_view(raw, stack, channel, "right", view[:, :, ::-1] if right is None else right, ty)
    return tmp_path / "acq"


def test_reader_cameras_flips_and_fields(tmp_path):
    m = LuxendoMosaicManager(_muvi(tmp_path), progress=False)
    assert m.cameras == ["left", "right"]
    assert m.fields == ["field0", "field1"]  # one per stack, not guessed from the positions
    assert m.groups == [None]  # illumination is not split
    left, right = (next(t for t in m.tiles if t.camera == c and t.attrs["stack"] == "0-x00-y00"
                        and t.channel == "DAPI") for c in ("left", "right"))
    assert left.flip == (False, False, False) and right.flip == (False, False, True)
    assert left.origin == right.origin  # same stage position, the right one mirrored about its centre
    assert left.attrs["detection_direction"][0] == pytest.approx(1.0, abs=1e-6)  # +z in the stage frame
    assert right.attrs["detection_direction"][0] == pytest.approx(-1.0, abs=1e-6)
    assert left.attrs["microscope"] == "MuVi-SPIM" and left.attrs["rotation_deg"] == 45.0
    # the napari view (lazy data) is flipped back as well
    np.testing.assert_array_equal(np.asarray(m._lazy(right)), np.asarray(m._lazy(left)))

    assert m.tile_id(right) == "DAPI/0-x00-y00/right"
    assert m.tiles_by_ids(["DAPI/0-x00-y00/right"]) == [right]
    assert set(map(id, m.tiles_by_ids(["DAPI/0-x00-y00"]))) == {id(left), id(right)}  # every camera
    assert len(m.tiles_by_ids(["0-x00-y00"])) == 4  # every channel and camera
    assert len(m.tiles_by_ids(["0-x00-y00/left"])) == 2  # every channel of one camera


def test_single_camera_luxendo_unchanged(tmp_path):
    """Without cameras (or with one), ids, fields and flips stay as before."""
    raw = tmp_path / "raw"
    _write_view(raw, "0-x00-y00", "2-DAPI", "left", _sample(), 0.0)
    m = LuxendoMosaicManager(tmp_path, progress=False)
    assert m.tile_id(m.tiles[0]) == "DAPI/0-x00-y00"
    assert plan_fusion(m.tiles).blend_by == "group"


def test_opposing_cameras_fuse_into_the_left_view(tmp_path):
    """The right view, flipped back, matches the left one: blending both gives the left view."""
    m = LuxendoMosaicManager(_muvi(tmp_path), progress=False)
    tiles = m.field_tiles("field0")
    plan = plan_fusion(tiles, channels=["DAPI"])
    assert plan.blend_by == "camera" and plan.midline_axis == "z"
    assert "2 cameras (blended along z, right -> left)" in plan.summary()

    both = m.fuse(tmp_path / "both.ome.zarr", field="field0", channels=["DAPI"], progress=False)
    only_left = m.fuse(tmp_path / "left.ome.zarr", tiles=[t for t in tiles if t.camera == "left"],
                       channels=["DAPI"], progress=False)
    np.testing.assert_array_equal(np.asarray(both.data[0]), np.asarray(only_left.data[0]))
    # two tiles stitched along y within each camera: 4 voxels apart
    assert np.asarray(both.data[0]).shape == (SHAPE[0], SHAPE[1] + 4, SHAPE[2])


def test_each_camera_gets_the_side_it_images_sharply(tmp_path):
    """Left camera (detection +z) wins at high z, right camera at low z."""
    left = np.full(SHAPE, 1000, np.uint16)
    right = np.full(SHAPE, 10, np.uint16)
    m = LuxendoMosaicManager(_muvi(tmp_path, left=left, right=right), progress=False)
    stack = [t for t in m.tiles if t.attrs["stack"] == "1" and t.channel == "DAPI"]
    out = np.asarray(m.fuse(tmp_path / "f.ome.zarr", tiles=stack, progress=False).data[0])
    profile = out[:, SHAPE[1] // 2, SHAPE[2] // 2]
    assert profile[0] < 100 and profile[-1] > 900  # right at low z, left at high z
    assert np.all(np.diff(profile.astype(int)) >= 0)  # a smooth switch in between

    mip = np.asarray(m.fuse(tmp_path / "mip.ome.zarr", tiles=stack, projection="max", progress=False).data[0])
    assert np.all(mip == 1000)  # projections: the maximum of the views


def test_cameras_and_groups_together_are_refused(tmp_path):
    m = LuxendoMosaicManager(_muvi(tmp_path), progress=False)
    tiles = m.field_tiles("field1")
    tiles[0].group = "L"
    with pytest.raises(ValueError, match="several cameras and several groups"):
        plan_fusion(tiles)


def test_several_rotations_are_refused(tmp_path):
    raw = tmp_path / "raw"
    _write_view(raw, "0", "2-DAPI", "left", _sample(), 0.0, rotation=45.0)
    _write_view(raw, "3", "2-DAPI", "left", _sample(), 2.0, rotation=90.0)
    m = LuxendoMosaicManager(tmp_path, progress=False)
    with pytest.raises(ValueError, match="several stage rotations"):
        plan_fusion(m.tiles)


def test_transforms_written_and_applied(tmp_path):
    m = LuxendoMosaicManager(_muvi(tmp_path), progress=False)
    out = tmp_path / "f.ome.zarr"
    m.fuse(out, field="field0", channels=["DAPI"], progress=False)
    path = default_transforms_path(out)
    assert path == tmp_path / "f.transforms.json"
    doc = load_transforms(path)
    assert doc["method"] == "stage" and doc["blend_by"] == "camera" and doc["axes"] == "zyx"
    assert len(doc["tiles"]) == 4  # 2 tiles x 2 cameras, one channel
    entry = next(e for e in doc["tiles"] if e["id"] == "DAPI/0-x01-y00/right")
    assert entry["flip"] == [False, False, True] and entry["offset"] == [0, 4, 0]
    affine = np.array(entry["affine"])
    # raw voxel x = 0 of the right camera is the last x of the fused (flipped) tile
    assert affine[2, 2] == -VOX[2]
    assert affine[2, 3] == pytest.approx(entry["placed_at"][2] + (SHAPE[2] - 1) * VOX[2])

    # edit a position, then fuse with the file: that tile moves
    entry["translation"][1] += 2 * VOX[1]
    path.write_text(json.dumps(doc))
    m2 = LuxendoMosaicManager(_muvi(tmp_path / "again"), progress=False)  # same layout, other folder
    m2.apply_transforms(path)
    moved = m2.tiles_by_ids(["DAPI/0-x01-y00/right"])[0]
    assert moved.origin[1] == pytest.approx(m2.tiles_by_ids(["DAPI/0-x01-y00/left"])[0].origin[1] + 2 * VOX[1])
    assert moved.attrs["transform_method"] == "stage" and "stage_origin" in moved.attrs
    m2.reset_transforms()
    assert moved.origin == m2.tiles_by_ids(["DAPI/0-x01-y00/left"])[0].origin


def test_cli_transforms_only_then_fuse_with_them(tmp_path, capsys):
    acq = _muvi(tmp_path)
    out = tmp_path / "cli.ome.zarr"
    cli_main(["fuse", "-i", str(acq), "-z", str(out), "-f", "field1", "-tfo"])
    transforms = tmp_path / "cli.transforms.json"
    assert transforms.is_file() and not out.exists()  # nothing fused
    assert "2 cameras" in capsys.readouterr().out

    doc = load_transforms(transforms)
    assert {e["camera"] for e in doc["tiles"]} == {"left", "right"}
    cli_main(["fuse", "-i", str(acq), "-z", str(out), "-f", "field1", "-tf", str(transforms), "-cam", "left"])
    zm = ZarrManager(str(out), mode="r")
    assert zm.metadata["channel_names"] == ["GFP", "DAPI"]  # folder order
    assert {e["camera"] for e in load_transforms(transforms)["tiles"]} == {"left"}  # rewritten for this fusion
