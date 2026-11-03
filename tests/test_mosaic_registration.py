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
