"""Image-based registration of the MuVi SPIM camera views (at the camera handover)."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy import ndimage as ndi

from zarrsmith.mosaic_manager import LuxendoMosaicManager, Tile, load_transforms
from zarrsmith.mosaic_manager.registration import (AffineArray, estimate_camera_registration, handover_z,
                                                   register_cameras, registered_tile, tile_grid)

from .test_muvi_fusion import _muvi

VOX = (2.0, 0.5, 0.5)
SHIFT = np.array([2.0, 1.5, -2.0])  # z, y, x (um): structure in the left view -> same in the right view


def _blobs(shape=(48, 96, 96), n=400, seed=0):
    """Nucleus-like spots (sigma ~2 um) on a dim background."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, np.float32)
    vol[tuple(rng.integers(0, s, n) for s in shape)] = 1.0
    vol = ndi.gaussian_filter(vol, np.array([2.0, 2.0, 2.0]) / np.array(VOX))
    return (100 + 4000 * vol / vol.max()).astype(np.uint16)


def _rigid(angle_deg=0.0, shift=SHIFT, centre=(0.0, 0.0, 0.0)):
    """Reference position -> moving position: rotation about z (in the yx plane) around
    ``centre``, then ``shift`` (tile frame, um)."""
    a = np.radians(angle_deg)
    m = np.eye(4)
    m[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    c = np.asarray(centre, float)
    m[:3, 3] = c - m[:3, :3] @ c + np.asarray(shift, float)
    return m


def _right_of(left, matrix=None):
    """The right camera's raw data: the left view moved by ``matrix`` (default: SHIFT),
    mirrored in x. Both views have their voxel [0, 0, 0] at the origin."""
    matrix = _rigid() if matrix is None else matrix
    v = np.diag(np.r_[VOX, 1.0])
    to_left = np.linalg.inv(v) @ np.linalg.inv(matrix) @ v  # moving voxel -> left voxel
    moved = ndi.affine_transform(left.astype(np.float32), to_left[:3, :3], offset=to_left[:3, 3], order=1,
                                 mode="nearest")
    return np.rint(moved).astype(np.uint16)[:, :, ::-1]


def _tiles(left, right, channel="DAPI"):
    kw = dict(origin=(0.0, 0.0, 0.0), voxel_size=VOX, channel=channel)
    return (Tile(data=left, camera="left", attrs={"stack": "0"}, **kw),
            Tile(data=right, camera="right", flip=(False, False, True), attrs={"stack": "0"}, **kw))


def _displacement_error(m, truth, shape):
    """Median distance between where ``m`` and ``truth`` send points of the volume (um)."""
    pts = np.stack(np.meshgrid(*[np.linspace(0, (s - 1) * v, 5) for s, v in zip(shape, VOX)],
                               indexing="ij"), -1).reshape(-1, 3)
    send = lambda t: (t[:3, :3] @ pts.T).T + t[:3, 3]  # noqa: E731
    return float(np.median(np.linalg.norm(send(m) - send(truth), axis=1)))


OPTS = dict(factors=(1, 1, 1), block_um=(30.0, 24.0, 24.0), band_um=80.0, workers=2)


def test_handover_is_the_middle_of_the_z_overlap():
    left, right = _tiles(np.zeros((10, 4, 4), np.uint16), np.zeros((10, 4, 4), np.uint16))
    assert handover_z(left, right) == pytest.approx(10.0)
    right.origin = (4.0, 0.0, 0.0)  # overlap 4..20
    assert handover_z(left, right) == pytest.approx(12.0)


@pytest.mark.slow
@pytest.mark.parametrize("model", ["translation", "rigid"])
def test_estimate_recovers_a_known_shift_of_the_mirrored_view(model):
    left = _blobs()
    r = estimate_camera_registration(*_tiles(left, _right_of(left)), model=model, **OPTS)
    assert _displacement_error(r.matrix, _rigid(), left.shape) < 0.3, r.summary()
    assert r.rotation[0] < 0.3
    assert r.misalignment() < 0.3 < r.misalignment(registered=False)
    assert r.z_range == pytest.approx((48.0 - 40.0, 48.0 + 40.0))
    assert r.to_dict()["moving_camera"] == "right"


@pytest.mark.slow
def test_estimate_recovers_a_rotation():
    left = _blobs(shape=(40, 160, 160), n=900)
    centre = (40.0, 40.0, 40.0)
    truth = _rigid(3.0, centre=centre)
    ref, mov = _tiles(left, _right_of(left, truth))
    rigid = estimate_camera_registration(ref, mov, **{**OPTS, "band_um": 70.0})
    angle, axis = rigid.rotation
    assert angle == pytest.approx(3.0, abs=0.3) and abs(axis[0]) > 0.95, rigid.summary()  # about z
    assert _displacement_error(rigid.matrix, truth, left.shape) < 0.3
    translation = estimate_camera_registration(ref, mov, model="translation", **{**OPTS, "band_um": 70.0})
    assert rigid.misalignment() < 0.3 < translation.misalignment()  # a translation leaves the rotation


def test_estimate_without_structure_raises():
    flat = np.full((48, 96, 96), 100, np.uint16)
    with pytest.raises(ValueError, match="blocks match|Nothing"):
        estimate_camera_registration(*_tiles(flat, flat), **OPTS)


def test_affine_array_reads_like_the_resampled_volume():
    vol = _blobs().astype(np.float32)
    m = np.eye(4)
    m[1:3, 1:3] = [[0.999, -0.04], [0.04, 0.999]]
    m[:3, 3] = [0.5, 3.0, -2.0]
    arr = AffineArray(vol, m, vol.shape)
    full = ndi.affine_transform(vol, m[:3, :3], offset=m[:3, 3], order=1, mode="nearest", prefilter=False)
    assert np.allclose(arr[...], full, atol=1e-3)
    assert np.allclose(arr[3:9, 10:50, 20:31], full[3:9, 10:50, 20:31], atol=1e-3)
    assert np.allclose(arr[5], full[5], atol=1e-3) and arr[5, 7].shape == (96,)
    assert pickle.loads(pickle.dumps(arr)).shape == vol.shape
    assert AffineArray(vol.astype(np.uint16), m, (2, 3, 4))[...].dtype == np.uint16


def test_registered_tile_translation_moves_rotation_resamples():
    left = _blobs()
    ref, mov = _tiles(left, _right_of(left))
    moved = registered_tile(mov, _rigid(), tile_grid(ref))
    assert np.allclose(moved.origin, -SHIFT) and moved.data is mov.data  # only moved
    rotated = registered_tile(mov, _rigid(0.5), tile_grid(ref))
    assert isinstance(rotated.data, AffineArray) and rotated.origin == ref.origin and rotated.flip == ref.flip
    assert rotated.attrs["stage_data"] is mov.data and rotated.attrs["registration"]["grid"]["shape"] == [48, 96, 96]
    # with no rotation, the resampled view of the right camera is the left view
    aligned = registered_tile(mov, _rigid() + np.diag([0, 1e-6, 0, 0]), tile_grid(ref))
    inner = (slice(4, -4), slice(8, -8), slice(8, -8))
    assert np.corrcoef(aligned.data[...][inner].ravel(), left[inner].ravel())[0, 1] > 0.98


@pytest.mark.slow
def test_register_cameras_reference_channel_applies_to_every_channel():
    left = _blobs()
    right = _right_of(left)
    dapi, gfp = _tiles(left, right, "DAPI"), _tiles(left // 2, right // 2, "GFP")
    tiles = [dapi[0], dapi[1], gfp[0], gfp[1]]
    out, results = register_cameras(tiles, reference_channel="DAPI", model="translation", **OPTS)
    assert list(results) == [("0", 0, None, "right", "DAPI")]  # measured once, on DAPI
    shift = results[("0", 0, None, "right", "DAPI")].shift
    assert results[("0", 0, None, "right", "DAPI")].channel == "DAPI"
    assert out[0] is tiles[0] and out[2] is tiles[2]  # reference camera untouched
    for moved, orig in ((out[1], tiles[1]), (out[3], tiles[3])):
        assert np.allclose(moved.origin, np.array(orig.origin) - shift)
        assert moved.attrs["stage_origin"] == orig.origin and moved.attrs["transform_method"] == "registration"
        assert moved.attrs["registration"]["channel"] == "DAPI" and moved.attrs["registration"]["timepoint"] == 0
    assert tiles[1].origin == (0.0, 0.0, 0.0)  # inputs are not changed
    out2, results2 = register_cameras(tiles, transforms={("0", 0, None, "right"): _rigid(shift=[1.0, 0, 0])})
    assert results2 == {} and out2[1].origin == (-1.0, 0.0, 0.0)  # known transform: not measured
    assert register_cameras(tiles[::2]) == (tiles[::2], {})  # one camera: nothing to do
    with pytest.raises(ValueError, match="Reference channel 'RFP'"):
        register_cameras(tiles, reference_channel="RFP")


@pytest.mark.slow
def test_register_cameras_every_channel_on_its_own():
    """``reference_channel="all"``: a channel offset differently (chromatic) gets its own
    transform; a channel without structure takes the best registered one."""
    left = _blobs()
    gfp_shift = SHIFT + [0.0, 1.0, 1.0]
    flat = np.full(left.shape, 100, np.uint16)
    dapi = _tiles(left, _right_of(left), "DAPI")
    gfp = _tiles(left, _right_of(left, _rigid(shift=gfp_shift)), "GFP")
    empty = _tiles(flat, flat, "empty")
    tiles = [*dapi, *gfp, *empty]
    out, results = register_cameras(tiles, reference_channel="all", model="translation", **OPTS)
    assert sorted(k[-1] for k in results) == ["DAPI", "GFP"]
    assert np.allclose(results[("0", 0, None, "right", "DAPI")].shift, SHIFT, atol=0.3)
    assert np.allclose(results[("0", 0, None, "right", "GFP")].shift, gfp_shift, atol=0.3)
    assert np.allclose(out[1].origin, -results[("0", 0, None, "right", "DAPI")].shift)
    assert np.allclose(out[3].origin, -results[("0", 0, None, "right", "GFP")].shift)
    assert out[5].attrs["registration"]["channel"] in ("DAPI", "GFP")  # fallback, no structure of its own
    assert out[5].origin in (out[1].origin, out[3].origin)


@pytest.mark.slow
def test_register_cameras_timepoints_follow_the_reference():
    """One transform measured at the reference timepoint holds for every timepoint;
    ``"all"`` measures each timepoint."""
    left = _blobs()
    later = SHIFT + [0.0, -1.0, 1.5]
    t0, t1 = _tiles(left, _right_of(left)), _tiles(left, _right_of(left, _rigid(shift=later)))
    for t in t1:
        t.timepoint = 1
    tiles = [*t0, *t1]
    opts = dict(model="translation", **OPTS)
    out, results = register_cameras(tiles, **opts)  # default: the first timepoint
    assert list(results) == [("0", 0, None, "right", "DAPI")]
    assert out[1].origin == out[3].origin and out[3].attrs["registration"]["timepoint"] == 0
    out, results = register_cameras(tiles, reference_timepoint=1, **opts)
    assert list(results) == [("0", 1, None, "right", "DAPI")]
    assert np.allclose(out[1].origin, -later, atol=0.3) and out[1].origin == out[3].origin
    out, results = register_cameras(tiles, reference_timepoint="all", **opts)
    assert sorted(k[1] for k in results) == [0, 1]
    assert np.allclose(out[1].origin, -SHIFT, atol=0.3) and np.allclose(out[3].origin, -later, atol=0.3)
    with pytest.raises(ValueError, match="Reference timepoint 5"):
        register_cameras(tiles, reference_timepoint=5, **opts)


@pytest.mark.slow
def test_mosaic_register_cameras_fuse_records_and_reset(tmp_path):
    left = _blobs()
    m = LuxendoMosaicManager(_muvi(tmp_path, left=left, right=_right_of(left)), progress=False)
    stage = {m.tile_id(t): (t.origin, t.flip, t.data) for t in m.tiles}
    results = m.register_cameras(field="field1", reference_channel="DAPI", **OPTS)
    reg = results[("1", 0, None, "right", "DAPI")]
    assert list(results) == [("1", 0, None, "right", "DAPI")] and reg.model == "rigid"
    for t in m.field_tiles("field1"):
        o, f, d = stage[m.tile_id(t)]
        if t.camera == "right":  # resampled on the left tile's grid
            assert isinstance(t.data, AffineArray) and t.flip == (False, False, False)
        else:
            assert t.data is d and t.origin == o
    assert all(t.data is stage[m.tile_id(t)][2] for t in m.field_tiles("field0"))  # other field untouched

    out = tmp_path / "f1.ome.zarr"
    m.fuse(out, field="field1", blend="sigmoid")
    doc = load_transforms(out.with_name("f1.transforms.json"))
    entry = next(e for e in doc["tiles"] if e["id"] == "DAPI/1/right")
    assert entry["method"] == "registration" and np.allclose(entry["registration"]["matrix"], reg.matrix)
    # the affine maps a raw voxel to where the registration puts it
    raw_vox = np.array([10.0, 20.0, 30.0, 1.0])
    stage_pos = np.array([*np.array(stage["DAPI/1/right"][0]), 0]) + np.r_[
        raw_vox[:3] * VOX * [1, 1, -1] + [0, 0, (left.shape[2] - 1) * VOX[2]], 1]
    expected = np.linalg.inv(reg.matrix) @ stage_pos
    assert np.allclose(np.array(entry["affine"]) @ raw_vox, expected, atol=0.6 * max(VOX))  # placement rounding

    # the file reproduces the registration on a fresh mosaic, reset goes back to the stage
    m2 = LuxendoMosaicManager(m.path, progress=False)
    m2.apply_transforms(out.with_name("f1.transforms.json"))
    t1 = next(t for t in m.tiles if m.tile_id(t) == "GFP/1/right")
    t2 = next(t for t in m2.tiles if m2.tile_id(t) == "GFP/1/right")
    assert np.allclose(t1.data.matrix, t2.data.matrix) and np.array_equal(t1.data[20:22], t2.data[20:22])
    m.reset_transforms()
    assert all((t.origin, t.flip) == stage[m.tile_id(t)][:2] and t.data is stage[m.tile_id(t)][2] for t in m.tiles)
    assert not any("registration" in t.attrs for t in m.tiles)


@pytest.mark.slow
def test_cli_register_cameras(tmp_path, capsys, monkeypatch):
    import functools

    from zarrsmith.cli.zarrsmith import main as cli_main
    from zarrsmith.mosaic_manager import registration

    left = _blobs()
    acq = _muvi(tmp_path, left=left, right=_right_of(left))
    out = tmp_path / "cli.ome.zarr"
    cli_main(["fuse", "-i", str(acq), "-z", str(out), "-f", "field1", "-rc", "-rch", "DAPI", "-rm", "translation"])
    assert "right -> left (translation, DAPI): shift" in capsys.readouterr().out
    doc = load_transforms(tmp_path / "cli.transforms.json")
    for e in doc["tiles"]:
        moved = np.array(e["stage_translation"]) - e["translation"]
        if e["camera"] == "right":  # default settings (downsampled 4x in yx): coarser than OPTS
            assert e["method"] == "registration" and np.allclose(moved, SHIFT, atol=1.0)
        else:
            assert e["method"] == "stage" and np.allclose(moved, 0)
    # rigid (default): the right view is resampled. The synthetic stack is too small for the
    # default block size (too few blocks for a rotation): use the fine settings of OPTS.
    monkeypatch.setattr(registration, "estimate_camera_registration",
                        functools.partial(registration.estimate_camera_registration, **OPTS))
    out = tmp_path / "rigid.ome.zarr"
    cli_main(["fuse", "-i", str(acq), "-z", str(out), "-f", "field1", "-rc"])
    entry = next(e for e in load_transforms(tmp_path / "rigid.transforms.json")["tiles"] if e["camera"] == "right")
    assert "grid" in entry["registration"] and entry["flip"] == [False, False, False]
