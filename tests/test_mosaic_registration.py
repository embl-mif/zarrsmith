import json

import numpy as np
import pytest
from scipy import ndimage as ndi

from zarrsmith import mosaic_manager as mos
from zarrsmith.mosaic_manager.registration import _find_pairs, _match, _reference_tiles, _structure


class _Mosaic(mos.MosaicManager):
    """Mosaic of tiles given directly (no reader)."""

    def __init__(self, tiles):
        self._given = tiles
        self.path = "synthetic"
        self._open_files = []
        self.tiles = tiles
        mos.label_fields(self.tiles)

    def _scan(self):
        return self._given


def _sample(shape, seed=0):
    """Blurred noise with bright spots: structure at several scales, as uint16."""
    rng = np.random.default_rng(seed)
    fine = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), 1.5)
    coarse = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), 5)
    arr = fine / fine.std() + coarse / coarse.std()
    return (500 + 100 * np.clip(arr, -4, None) + rng.normal(0, 5, shape)).astype(np.uint16)


def _grid(sample, rows, cols, size, overlap, errors=None, channel="A", timepoint=0, vox=(1.0, 1.0, 1.0)):
    """Tiles cut from ``sample``; ``errors[stack]`` is added to the true position to make the
    stage position. Returns ``(tiles, {stack: true (z, y, x) position})``."""
    tiles, truth = [], {}
    step = size - overlap
    for r in range(rows):
        for c in range(cols):
            stack = f"r{r}c{c}"
            true = np.array((0, r * step, c * step))
            cut = sample[:, true[1]:true[1] + size, true[2]:true[2] + size]
            error = np.array((errors or {}).get(stack, (0, 0, 0)), float)
            tiles.append(mos.Tile(cut, tuple((true + error) * vox), vox, channel, timepoint=timepoint,
                                  attrs={"stack": stack}))
            truth[stack] = true * np.array(vox)
    return tiles, truth


def _residual_error(tiles, truth):
    """Largest error left between tiles (voxels), once their common offset is removed."""
    found = {t.attrs["stack"]: np.array(t.origin) for t in tiles}
    offsets = np.array([found[k] - truth[k] for k in truth])
    return np.max(np.abs(offsets - offsets.mean(axis=0)))


ERRORS = {"r0c1": (0, 4, -3), "r1c0": (0, -5, 2), "r1c1": (0, 3, 5)}


def test_recovers_stage_errors_in_2d(tmp_path):
    sample = _sample((1, 400, 400))
    tiles, truth = _grid(sample, 2, 2, 220, 40, ERRORS)
    mosaic = _Mosaic(tiles)
    before = _residual_error(tiles, truth)
    reg = mosaic.register(tmp_path / "reg.json", progress=False)
    assert reg.n_used == 4
    mosaic.apply_registration(reg)
    assert before > 3
    assert _residual_error(tiles, truth) < 0.5


def test_recovers_stage_errors_in_3d_with_z(tmp_path):
    sample = _sample((80, 300, 300))
    errors = {"r0c1": (2, 3, -3)}
    tiles, truth = _grid(sample, 1, 2, 180, 60, errors)
    reg = mos.register_tiles(tiles, tmp_path / "reg.json", max_shift=(0.1, 0.06, 0.06), progress=False)
    mosaic = _Mosaic(tiles)
    mosaic.apply_registration(reg)
    assert _residual_error(tiles, truth) < 0.5


def test_correct_stage_positions_stay(tmp_path):
    sample = _sample((1, 400, 400))
    tiles, truth = _grid(sample, 2, 2, 220, 40)
    reg = mos.register_tiles(tiles, tmp_path / "reg.json", progress=False)
    assert np.max(np.abs(list(reg.shifts.values()))) < 0.5


def test_blank_tile_keeps_its_stage_position(tmp_path):
    sample = _sample((1, 400, 400))
    tiles, truth = _grid(sample, 2, 2, 220, 40, ERRORS)
    blank = next(t for t in tiles if t.attrs["stack"] == "r1c1")
    blank.data = np.full(blank.shape, 500, np.uint16)
    stage = blank.origin
    mosaic = _Mosaic(tiles)
    reg = mosaic.register(tmp_path / "reg.json", progress=False)
    mosaic.apply_registration(reg)
    assert reg.n_used == 2
    assert np.allclose(blank.origin, stage, atol=1e-2)
    assert "low NCC" in reg.summary() or "no measurement" in reg.summary()
    rest = [t for t in tiles if t is not blank]
    assert _residual_error(rest, {t.attrs["stack"]: truth[t.attrs["stack"]] for t in rest}) < 0.5


def test_reference_channel_and_timepoint_apply_to_all(tmp_path):
    sample = _sample((1, 300, 400))
    errors = {"r0c1": (0, 4, -3)}
    tiles = []
    for channel, timepoint, data in (("A", 0, sample), ("B", 0, sample * 0 + 7),
                                     ("A", 1, sample * 0 + 9), ("B", 1, sample * 0 + 11)):
        tiles += _grid(data, 1, 2, 220, 40, errors, channel=channel, timepoint=timepoint)[0]
    mosaic = _Mosaic(tiles)
    reg = mosaic.register(tmp_path / "reg.json", reference_channel="A", reference_timepoint=0, progress=False)
    assert reg.n_used == 1
    mosaic.apply_registration(reg)
    moved = {(t.channel, t.timepoint): t.origin for t in tiles if t.attrs["stack"] == "r0c1"}
    assert len(set(moved.values())) == 1
    with pytest.raises(ValueError, match="stage positions"):  # measured from where the stage put them
        mosaic.register(tmp_path / "again.json", progress=False)
    mosaic.clear_registration()
    with pytest.raises(ValueError, match="Reference channel"):
        mosaic.register(tmp_path / "other.json", reference_channel="C", progress=False)
    with pytest.raises(ValueError, match="Reference timepoint"):
        mosaic.register(tmp_path / "other.json", reference_timepoint=5, progress=False)


def test_apply_twice_clear_and_fuse(tmp_path):
    sample = _sample((1, 300, 400))
    tiles, _ = _grid(sample, 1, 2, 220, 40, {"r0c1": (0, 6, -4)})
    mosaic = _Mosaic(tiles)
    stage = [t.origin for t in tiles]
    without = np.asarray(mosaic.fuse(tmp_path / "stage.ome.zarr", projection="max", blend="max",
                                     progress=False).data[0])
    reg = mosaic.register(tmp_path / "reg.json", progress=False)
    mosaic.apply_registration(reg)
    once = [t.origin for t in tiles]
    mosaic.apply_registration(reg)
    assert [t.origin for t in tiles] == once and once != stage

    fused = np.asarray(mosaic.fuse(tmp_path / "registered.ome.zarr", projection="max", blend="max",
                                   progress=False).data[0])
    assert np.squeeze(fused).shape != np.squeeze(without).shape  # the 6 voxel y error is gone
    truth = sample[0, :220, :400].astype(float)
    a = np.squeeze(fused)[:220, :400].astype(float)
    assert np.corrcoef(a.ravel(), truth.ravel())[0, 1] > 0.95

    mosaic.clear_registration()
    assert [t.origin for t in tiles] == stage
    assert all("stage_origin" not in t.attrs for t in tiles)
    again = np.asarray(mosaic.fuse(tmp_path / "stage2.ome.zarr", projection="max", blend="max",
                                   progress=False).data[0])
    assert np.array_equal(without, again)


def test_no_registration_leaves_tiles_alone():
    tiles, _ = _grid(_sample((1, 100, 200)), 1, 2, 80, 10)
    stage = [t.origin for t in tiles]
    mos.plan_fusion(tiles)
    assert [t.origin for t in tiles] == stage and all("stage_origin" not in t.attrs for t in tiles)


def test_file_round_trip_and_solve_without_reads(tmp_path):
    sample = _sample((1, 400, 400))
    tiles, truth = _grid(sample, 2, 2, 220, 40, ERRORS)
    path = tmp_path / "reg.json"
    reg = mos.register_tiles(tiles, path, progress=False)
    loaded = mos.Registration.load(path)
    assert loaded.shifts == reg.shifts
    assert loaded.reference_channel == "A" and loaded.n_used == reg.n_used

    for t in tiles:
        t.data = None  # any read would now fail
    loaded.solve(min_ncc=1.01)
    assert loaded.n_used == 0
    assert np.allclose(list(loaded.shifts.values()), 0)
    loaded.solve(min_ncc=0.2)
    assert loaded.n_used == reg.n_used
    assert "Rejected" not in loaded.summary()


def test_interrupted_run_resumes_without_reading_again(tmp_path, monkeypatch):
    from zarrsmith.mosaic_manager import registration as module

    sample = _sample((1, 400, 400))
    tiles, _ = _grid(sample, 2, 2, 220, 40, ERRORS)
    path = tmp_path / "reg.json"
    real = module._measure_pair
    calls = []

    def stop_after_two(job):
        if len(calls) == 2:
            raise KeyboardInterrupt
        calls.append((job.a, job.b))
        return real(job)

    monkeypatch.setattr(module, "_measure_pair", stop_after_two)
    with pytest.raises(KeyboardInterrupt):
        mos.register_tiles(tiles, path, progress=False)
    saved = json.loads(path.read_text())
    assert len(saved["links"]) == 2 and saved["shifts"] == {}
    with pytest.raises(ValueError, match="no solution"):
        _Mosaic(tiles).apply_registration(mos.Registration.load(path))

    remaining = []

    def count(job):
        remaining.append((job.a, job.b))
        return real(job)

    monkeypatch.setattr(module, "_measure_pair", count)
    reg = mos.register_tiles(tiles, path, progress=False)
    assert len(remaining) == 2 and not set(remaining) & set(calls)
    assert reg.n_used == 4


def test_file_of_other_settings_or_acquisition_is_refused(tmp_path):
    sample = _sample((1, 400, 400))
    tiles, _ = _grid(sample, 2, 2, 220, 40, ERRORS)
    path = tmp_path / "reg.json"
    reg = mos.register_tiles(tiles, path, progress=False)
    with pytest.raises(ValueError, match="another registration"):
        mos.register_tiles(tiles, path, max_shift=0.03, progress=False)

    elsewhere, _ = _grid(sample, 2, 2, 220, 40)
    for t in elsewhere:
        t.origin = tuple(v + 100 for v in t.origin)
    with pytest.raises(ValueError, match="another acquisition"):
        _Mosaic(elsewhere).apply_registration(reg)


def test_retry_rejected_measures_only_those(tmp_path, monkeypatch):
    from zarrsmith.mosaic_manager import registration as module

    sample = _sample((1, 400, 400))
    tiles, _ = _grid(sample, 2, 2, 220, 40, ERRORS)
    blank = next(t for t in tiles if t.attrs["stack"] == "r1c1")
    data, blank.data = blank.data, np.full(blank.shape, 500, np.uint16)
    path = tmp_path / "reg.json"
    assert mos.register_tiles(tiles, path, progress=False).n_used == 2
    blank.data = data
    real = module._measure_pair
    retried = []

    def count(job):
        retried.append(job.a)
        return real(job)

    monkeypatch.setattr(module, "_measure_pair", count)
    assert mos.register_tiles(tiles, path, retry_rejected=True, progress=False).n_used == 4
    assert len(retried) == 2


def test_budget_limits_reads_and_slabs_find_the_signal(tmp_path):
    # signal only in the deepest planes: the scout must send the slabs there
    shape = (512, 128, 256)
    sample = np.full(shape, 500, np.uint16)
    rng = np.random.default_rng(1)
    sample = (sample + rng.normal(0, 5, shape)).astype(np.uint16)
    sample[384:] = _sample((128, 128, 256), seed=2)
    a = sample[:, :, :160]
    b = sample[:, :, 100:]
    tiles = [mos.Tile(a, (0, 0, 0), (1, 1, 1), "A", attrs={"stack": "a"}),
             mos.Tile(b, (0, 0, 103), (1, 1, 1), "A", attrs={"stack": "b"})]
    plan = mos.plan_registration(tiles, budget_mb=12)
    assert plan.limit_gb < plan.full_gb
    reg = mos.register_tiles(tiles, tmp_path / "reg.json", budget_mb=12, progress=False)
    link = reg.links[0]
    assert link["slabs_available"] > 1
    assert link["read_mb"] <= 12 * 1.5
    assert all(e["z"][0] >= 320 for e in link["estimates"])
    assert reg.n_used == 1
    assert abs((reg.shifts["|b|0"][2] - reg.shifts["|a|0"][2]) - (-3)) < 0.5


def test_structure_tells_tissue_from_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(500, 20, (64, 64, 64))
    assert _structure(noise) == pytest.approx(1, abs=0.2)
    assert _structure(_sample((64, 64, 64))) > 3


def test_budget_is_per_pair_whatever_the_neighbours():
    tiles, _ = _grid(_sample((512, 400, 400)), 2, 2, 220, 40)
    plan = mos.plan_registration(tiles, budget_mb=2)
    assert len(plan.pairs) == 4
    assert plan.limit_gb == pytest.approx(4 * 2 / 1e3)  # 2 MB for each of the 4 pairs


def test_no_signal_stops_after_the_probes(tmp_path):
    rng = np.random.default_rng(3)
    noise = rng.normal(500, 5, (512, 128, 256)).astype(np.uint16)
    tiles = [mos.Tile(noise[:, :, :160], (0, 0, 0), (1, 1, 1), "A", attrs={"stack": "a"}),
             mos.Tile(noise[:, :, 100:], (0, 0, 100), (1, 1, 1), "A", attrs={"stack": "b"})]
    reg = mos.register_tiles(tiles, tmp_path / "reg.json", budget_mb=12, progress=False)
    assert reg.links[0]["signal"] is False and reg.links[0]["slabs_tried"] == 0
    assert reg.links[0]["read_mb"] < 12
    assert "no signal" in reg.summary()


def test_slabs_without_a_match_still_respect_the_budget(tmp_path, monkeypatch):
    from zarrsmith.mosaic_manager import registration as module

    sample = _sample((512, 128, 256))
    tiles = [mos.Tile(sample[:, :, :160], (0, 0, 0), (1, 1, 1), "A", attrs={"stack": "a"}),
             mos.Tile(sample[:, :, 100:], (0, 0, 100), (1, 1, 1), "A", attrs={"stack": "b"})]
    monkeypatch.setattr(module, "_match", lambda *args: None)
    reg = mos.register_tiles(tiles, tmp_path / "reg.json", budget_mb=12, progress=False)
    assert 1 <= reg.links[0]["slabs_tried"] < reg.links[0]["slabs_available"]
    assert reg.links[0]["read_mb"] <= 12 * 1.5


def test_failed_reads_are_tried_again(tmp_path, monkeypatch):
    from zarrsmith.mosaic_manager import registration as module

    sample = _sample((1, 300, 400))
    tiles, _ = _grid(sample, 1, 2, 220, 40, {"r0c1": (0, 4, -3)})
    real = module._measure_once
    failures = [OSError("network"), OSError("network")]

    def flaky(job):
        if failures:
            raise failures.pop()
        return real(job)

    monkeypatch.setattr(module, "_measure_once", flaky)
    monkeypatch.setattr(module, "_RETRY_SECONDS", 0.0)
    assert mos.register_tiles(tiles, tmp_path / "reg.json", progress=False).n_used == 1
    assert not failures

    def broken(job):
        raise OSError("gone")

    monkeypatch.setattr(module, "_measure_once", broken)
    with pytest.raises(OSError, match="gone"):
        mos.register_tiles(tiles, tmp_path / "other.json", progress=False)


def test_pairs_are_faces_within_one_field():
    tiles, _ = _grid(_sample((1, 400, 400)), 2, 2, 220, 40)
    far = mos.Tile(np.zeros((1, 220, 220), np.uint16), (0, 5000, 5000), (1, 1, 1), "A", attrs={"stack": "far"})
    tiles.append(far)
    mos.label_fields(tiles)
    refs, _, _ = _reference_tiles(tiles, None, None)
    pairs = _find_pairs(refs, (0.0, 0.06, 0.06))
    assert len(pairs) == 4  # no corners, nothing with the other field
    assert not any("far" in a or "far" in b for a, b in pairs)


def test_match_prefers_the_real_shift_over_a_slow_background():
    sample = _sample((1, 200, 300)).astype(np.float32)
    ramp = np.linspace(0, 2000, 300, dtype=np.float32)[None, None, :]
    a = sample[:, :, :120] + ramp[:, :, :120]
    b = sample[:, :, 7:127] + ramp[:, :, :120]
    found = _match(a, b, np.zeros(3), np.array([0.0, 20.0, 20.0]))
    assert found["s"][2] == pytest.approx(7, abs=0.5) and found["s"][1] == pytest.approx(0, abs=0.5)
    assert found["ncc"] > found["ncc_stage"]


def test_plan_reads_no_pixels():
    tiles, _ = _grid(_sample((1, 400, 400)), 2, 2, 220, 40)
    for t in tiles:
        t.data = type("NoRead", (), {"shape": t.shape, "dtype": t.dtype, "__getitem__": lambda s, k: 1 / 0})()
    plan = mos.plan_registration(tiles)
    assert len(plan.pairs) == 4 and "4 pairs" in plan.summary()


# ---------------------------------------------------------------------------
# several groups: tilt inside the tiles, one offset between the groups
# ---------------------------------------------------------------------------

def _tilted(sample, y0, x0, shape, tilt, z_extra=0):
    """Tile cut from ``sample`` whose planes are tilted: a voxel at height h shows at plane
    h + tilt_y * u_y + tilt_x * u_x (u: distance to the tile centre), plus ``z_extra`` planes."""
    nz, ny, nx = shape
    rows = np.arange(ny) - (ny - 1) / 2
    columns = np.arange(nx) - (nx - 1) / 2
    moves = np.rint(tilt[0] * rows[:, None] + tilt[1] * columns[None, :]).astype(int) + z_extra
    margin = 12
    tile = np.empty(shape, sample.dtype)
    for j in range(ny):
        for i in range(nx):
            tile[:, j, i] = sample[margin - moves[j, i]:margin - moves[j, i] + nz, y0 + j, x0 + i]
    return tile


def _two_sides(channels=("A",), tilts=((0.03, 0.0), (-0.01, 0.01)), seed=0, depth=72, errors=None):
    """Left and right blocks of 3 x 2 tiles, tilted differently. Channel B is ``z_extra`` planes off
    on the right side only. Returns ``(tiles, sample, truth)``; ``truth`` has the y/x of every stack."""
    size, step = 120, 90
    sample = _sample((depth + 24, 3 * step + size, 4 * step + size), seed=seed)
    tiles, truth = [], {}
    for channel in channels:
        extra = 3 if channel == "B" else 0
        for side, tilt, columns in (("L", tilts[0], (0, 1)), ("R", tilts[1], (2, 3))):
            for row in range(3):
                for column in columns:
                    stack = f"{side}{row}{column}"
                    y0, x0 = row * step, column * step
                    error = np.array((errors or {}).get(stack, (0, 0, 0)), float)
                    tile = _tilted(sample, y0, x0, (depth, size, size), tilt, extra if side == "R" else 0)
                    tiles.append(mos.Tile(tile, tuple(np.array([0, y0, x0]) + error), (1, 1, 1), channel, group=side,
                                          attrs={"stack": stack}))
                    truth[stack] = np.array([0, y0, x0], float)
    return tiles, sample, truth


def _seam_score(mosaic, tiles, a, b, channel="A"):
    """NCC over the voxels two placed tiles share."""
    ta = next(t for t in tiles if t.attrs["stack"] == a and t.channel == channel)
    tb = next(t for t in tiles if t.attrs["stack"] == b and t.channel == channel)
    lo = np.ceil(np.maximum(ta.extent[0], tb.extent[0])).astype(int)
    hi = np.floor(np.minimum(ta.extent[1], tb.extent[1])).astype(int)

    def cut(t):
        start = np.rint(lo - np.array(t.origin)).astype(int)
        stop = start + hi - lo
        return np.asarray(t.data[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]], float)

    x, y = cut(ta), cut(tb)
    keep = (x > 0) & (y > 0)
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


TWO_SIDE_ERRORS = {"L01": (0, 3, -2), "L10": (0, -2, 2), "R02": (0, 2, 3), "R13": (0, -3, -1), "R22": (0, 1, 2)}


def test_tilted_array_moves_columns_by_whole_planes():
    data = np.arange(1, 8 * 6 * 4 + 1, dtype=np.uint16).reshape(8, 6, 4)
    tilted = mos.TiltedArray(data, (2.0, 1.0, 1.0), (0.8, 0.0))  # 0.4 planes per row, downwards with y
    full = tilted[:, :, :]
    assert full.dtype == data.dtype and full.shape == tilted.shape and tilted.shape[1:] == (6, 4)
    assert full.shape[0] == 8 + np.ptp(tilted._columns(slice(0, 6), slice(0, 4)))
    # every voxel value is kept, in its own column, in order and with no interpolation
    for j in range(6):
        column = full[:, j, 0]
        assert np.array_equal(column[column > 0], data[:, j, 0])
    # the top row moves opposite to the bottom row
    moves = tilted._columns(slice(0, 6), slice(0, 4))
    assert moves[0, 0] > 0 > moves[-1, 0]
    assert np.flatnonzero(full[:, 0, 0])[0] == tilted.lead + moves[0, 0]
    # any slice equals the same slice of the whole
    assert np.array_equal(tilted[2:7, 1:5, 1:3], full[2:7, 1:5, 1:3])
    assert np.array_equal(tilted[3, 2, 1], full[3, 2, 1]) and np.array_equal(tilted[-1, :, 2], full[-1, :, 2])
    assert np.array_equal(tilted[..., 1], full[..., 1])
    assert tilted[0:0, :, :].size == 0
    with pytest.raises(IndexError, match="step"):
        tilted[::2]


def test_tilted_array_can_be_sent_to_workers_and_read_lazily():
    import pickle

    import dask.array as da

    reads = []

    class Reading(np.ndarray):
        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)

    data = np.ones((8, 6, 4), np.uint16).view(Reading)
    tilted = mos.TiltedArray(data, (1, 1, 1), (0.5, 0.0))
    assert not reads  # nothing is read until it is sliced
    assert not tilted.picklable  # as its data
    h5 = type("Picklable", (), {"picklable": True, "shape": (8, 6, 4), "dtype": np.uint16})()
    assert mos.TiltedArray(h5, (1, 1, 1), (0.5, 0.0)).picklable
    again = pickle.loads(pickle.dumps(mos.TiltedArray(np.asarray(data), (1, 1, 1), (0.5, 0.0))))
    assert np.array_equal(again[:, :, :], mos.TiltedArray(np.asarray(data), (1, 1, 1), (0.5, 0.0))[:, :, :])
    lazy = da.from_array(mos.TiltedArray(np.asarray(data), (1, 1, 1), (0.5, 0.0)), chunks=(4, 3, 2))
    assert lazy.compute().shape == mos.TiltedArray(data, (1, 1, 1), (0.5, 0.0)).shape


def test_tilts_are_found_per_group_and_do_not_add_up(tmp_path):
    tiles, _, truth = _two_sides(errors=TWO_SIDE_ERRORS)
    mosaic = _Mosaic(tiles)
    reg = mosaic.register(tmp_path / "reg.json", max_shift=(0.1, 0.1, 0.1), progress=False)
    field = tiles[0].field
    assert reg.tilts[f"{field}|L"][0] == pytest.approx(0.03, abs=0.006)
    assert reg.tilts[f"{field}|R"][0] == pytest.approx(-0.01, abs=0.006)
    assert reg.tilts[f"{field}|R"][1] == pytest.approx(0.01, abs=0.006)
    # z stays at the stage: the offsets between rows came from the tilt and did not build up
    assert np.max(np.abs(np.array(list(reg.shifts.values()))[:, 0])) < 1.0
    mosaic.apply_registration(reg)
    found = {t.attrs["stack"]: np.array(t.origin) for t in tiles}
    offsets = np.array([found[k][1:] - truth[k][1:] for k in truth])
    assert np.max(np.abs(offsets - offsets.mean(axis=0))) < 1.0
    assert "Tilt of block" in reg.summary()


def test_seams_between_groups_match_after_registration(tmp_path):
    tiles, _, _ = _two_sides(errors=TWO_SIDE_ERRORS)
    mosaic = _Mosaic(tiles)
    seams = [("L01", "R02"), ("L11", "R12"), ("L21", "R22")]
    same = [("L00", "L10"), ("R12", "R22")]
    before = [_seam_score(mosaic, tiles, a, b) for a, b in seams + same]
    reg = mosaic.register(tmp_path / "reg.json", max_shift=(0.1, 0.1, 0.1), progress=False)
    mosaic.apply_registration(reg)
    after = [_seam_score(mosaic, tiles, a, b) for a, b in seams + same]
    assert min(after) > 0.9, after
    assert min(after) > max(before), (before, after)


def test_every_channel_gets_its_own_z_offset_between_groups(tmp_path):
    tiles, _, _ = _two_sides(("A", "B"), errors=TWO_SIDE_ERRORS)
    mosaic = _Mosaic(tiles)
    reg = mosaic.register(tmp_path / "reg.json", reference_channel="A", max_shift=(0.1, 0.1, 0.1), progress=False)
    right = f"{tiles[0].field}|R"
    assert reg.z_offsets[right]["B"] - reg.z_offsets[right]["A"] == pytest.approx(-3, abs=0.6)  # B sits 3 planes higher
    assert sum(len(v) for v in reg.channel_links.values()) == 3  # the seams, not all pairs
    mosaic.apply_registration(reg)
    for channel in "AB":
        assert _seam_score(mosaic, tiles, "L11", "R12", channel) > 0.9
    # y and x are the same in all channels
    origins = {(t.attrs["stack"], t.channel): t.origin for t in tiles}
    assert origins[("R12", "A")][1:] == origins[("R12", "B")][1:]
    # a channel without a match falls back on the reference, and says so
    loaded = mos.Registration.load(tmp_path / "reg.json")
    assert loaded.z_offsets == reg.z_offsets
    loaded.channel_links["B"] = [dict(link, estimates=[]) for link in loaded.channel_links["B"]]
    loaded.solve()
    assert loaded.z_offsets[right]["B"] == loaded.z_offsets[right]["A"]
    assert "no trusted match" in loaded.summary()


def test_apply_and_clear_with_tilt_restore_the_tiles(tmp_path):
    tiles, _, _ = _two_sides(errors=TWO_SIDE_ERRORS)
    mosaic = _Mosaic(tiles)
    stage = [t.origin for t in tiles]
    data = [t.data for t in tiles]
    reg = mosaic.register(tmp_path / "reg.json", max_shift=(0.1, 0.1, 0.1), progress=False)
    mosaic.apply_registration(reg)
    assert all(isinstance(t.data, mos.TiltedArray) for t in tiles)
    once = [t.origin for t in tiles]
    mosaic.apply_registration(reg)  # the tilt does not stack up
    assert [t.origin for t in tiles] == once
    assert all(t.data.data is d for t, d in zip(tiles, data))
    with pytest.raises(ValueError, match="stage positions"):
        mosaic.register(tmp_path / "again.json", progress=False)
    mosaic.clear_registration()
    assert [t.origin for t in tiles] == stage
    assert all(t.data is d for t, d in zip(tiles, data))
    assert not any("stage_data" in t.attrs for t in tiles)


def test_fusion_uses_the_tilt_and_keeps_voxel_values(tmp_path):
    tiles, sample, _ = _two_sides(errors=TWO_SIDE_ERRORS)
    mosaic = _Mosaic(tiles)
    reg = mosaic.register(tmp_path / "reg.json", max_shift=(0.1, 0.1, 0.1), progress=False)
    mosaic.apply_registration(reg)
    fused = np.asarray(mosaic.fuse(tmp_path / "out.ome.zarr", blend="max", progress=False).data[0])
    assert fused.ndim == 3 and fused.shape[0] > 72  # a few planes deeper, not resampled
    assert np.isin(np.unique(fused)[:2000], np.append(np.unique(sample), 0)).all()  # no new values
    # the seam: a plane through the middle follows the sample, across both sides
    middle = fused[fused.shape[0] // 2, 40:200, 150:290].astype(float)
    best = 0.0
    for z in range(0, sample.shape[0]):
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                best = max(best, float(np.corrcoef(middle.ravel(),
                                                   sample[z, 40 + 2 + dy:200 + 2 + dy, 150 + 2 + dx:290 + 2 + dx].ravel())[0, 1]))
    assert best > 0.7


def test_one_group_moves_whole_tiles_without_tilt(tmp_path):
    sample = _sample((80, 300, 300))
    tiles, truth = _grid(sample, 1, 2, 180, 60, {"r0c1": (2, 3, -3)})
    mosaic = _Mosaic(tiles)
    data = [t.data for t in tiles]
    reg = mos.register_tiles(tiles, tmp_path / "reg.json", max_shift=(0.1, 0.06, 0.06), progress=False)
    assert reg.tilts == {} and reg.z_offsets == {}
    mosaic.apply_registration(reg)
    assert all(t.data is d for t, d in zip(tiles, data))  # the data is untouched
    assert _residual_error(tiles, truth) < 0.5


def test_tiles_without_a_link_follow_the_trend(tmp_path):
    # a stage with a wrong scale, as a wrong pixel size gives; one corner tile has nothing to match
    sample = _sample((1, 620, 620))
    scale = 0.03
    errors = {f"r{r}c{c}": (0, -scale * r * 160, -scale * c * 160) for r in range(4) for c in range(4)}
    tiles, truth = _grid(sample, 4, 4, 220, 60, errors)
    corner = next(t for t in tiles if t.attrs["stack"] == "r3c3")
    corner.data = np.full(corner.shape, 500, np.uint16)
    mosaic = _Mosaic(tiles)
    reg = mosaic.register(tmp_path / "reg.json", progress=False)
    mosaic.apply_registration(reg)
    assert not any(corner_key in key for key in [k for k, o in reg.outcome.items() if o["status"] == "used"]
                   for corner_key in ("|r3c3|",))
    assert _residual_error(tiles, truth) < 1.0  # also the corner, far from its stage position
    assert np.abs(np.array(corner.origin) - np.array(corner.attrs["stage_origin"])).max() > 3


def test_fine_pass_measures_at_full_resolution(tmp_path, monkeypatch):
    from zarrsmith.mosaic_manager import registration as module

    sample = _sample((64, 400, 620), seed=4)
    a = mos.Tile(sample[:, :, :340], (0, 0, 0), (1, 1, 1), "A", attrs={"stack": "a"})
    b = mos.Tile(sample[:, :, 240:], (0, 0, 243), (1, 1, 1), "A", attrs={"stack": "b"})
    monkeypatch.setattr(module, "_WORK_VOXELS", 2**19)  # forces downsampled matching on this small data
    reg = mos.register_tiles([a, b], tmp_path / "reg.json", progress=False)
    link = reg.links[0]
    assert np.max(link["estimates"][0]["voxel"][1:]) > 1
    assert "fine" in link and link["fine"]["ncc"] > 0.5
    relative = np.array(reg.shifts["|b|0"]) - np.array(reg.shifts["|a|0"])
    assert relative[2] == pytest.approx(-3, abs=0.3)
    # the whole matching stays within the budget, and the pass is skipped when it does not fit
    assert reg.links[0]["read_mb"] < module.BUDGET_MB


def test_fine_pass_is_left_out_at_full_resolution(tmp_path):
    tiles, _ = _grid(_sample((1, 300, 400)), 1, 2, 220, 40, {"r0c1": (0, 4, -3)})
    reg = mos.register_tiles(tiles, tmp_path / "reg.json", progress=False)
    assert "fine" not in reg.links[0]


def test_files_of_the_earlier_version(tmp_path):
    tiles, _ = _grid(_sample((1, 300, 400)), 1, 2, 220, 40, {"r0c1": (0, 4, -3)})
    path = tmp_path / "reg.json"
    reg = mos.register_tiles(tiles, path, progress=False)
    content = json.loads(path.read_text())
    content["version"] = 1
    for key in ("fields", "channel_links", "tilts", "z_offsets"):
        content.pop(key)
    for link in content["links"]:
        link.pop("centres"), link.pop("overlap")
    path.write_text(json.dumps(content))
    assert mos.Registration.load(path).shifts.keys() == reg.shifts.keys()  # thin tiles of one group still work
    for name, shift in mos.Registration.load(path).shifts.items():
        assert shift == pytest.approx(reg.shifts[name])

    tiles, _, _ = _two_sides(errors=TWO_SIDE_ERRORS)
    path = tmp_path / "groups.json"
    mos.register_tiles(tiles, path, max_shift=(0.1, 0.1, 0.1), progress=False)
    content = json.loads(path.read_text())
    content["version"] = 1
    for link in content["links"]:
        link.pop("centres"), link.pop("overlap")
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="earlier version"):
        mos.Registration.load(path)
