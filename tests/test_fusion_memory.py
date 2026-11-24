"""Fusion workers each hold a whole shard: their number is limited by the free memory."""
from __future__ import annotations

import numpy as np
import pytest

from zarrsmith.mosaic_manager import Tile, fuse_tiles
from zarrsmith.mosaic_manager import fusion
from zarrsmith.utils import ngff


def test_worker_memory_counts_the_shard_twice_and_the_canvases():
    region, piece = (16, 256, 256), (16, 128, 128)
    expected = 2 * 16 * 256 * 256 * 2 + 16 * 128 * 128 * 4 * fusion._CANVASES_PER_PIECE
    assert fusion._worker_bytes(region, piece, np.uint16, flat=False) == expected
    # projections: the z of the region and piece is reduced while reading
    assert fusion._worker_bytes((16, 256, 256), (16, 128, 128), np.uint16, flat=True) == \
        2 * 256 * 256 * 2 + 128 * 128 * 4 * fusion._CANVASES_PER_PIECE


def test_workers_are_lowered_to_what_fits(monkeypatch):
    region = piece = (64, 1024, 1024)  # 128 MB of uint16: 256 MB + 768 MB of canvases per worker
    per_worker = fusion._worker_bytes(region, piece, np.uint16, flat=False)
    monkeypatch.setattr(ngff, "_available_memory_mb", lambda: 2 * 3 * per_worker / 1024**2)  # budget: 3 workers
    with pytest.warns(UserWarning, match="3 of the 8 workers fit"):
        assert fusion._workers_that_fit(8, region, piece, np.uint16, flat=False) == 3
    monkeypatch.setattr(ngff, "_available_memory_mb", lambda: 1.0)  # nothing fits: still one worker
    with pytest.warns(UserWarning):
        assert fusion._workers_that_fit(4, region, piece, np.uint16, flat=False) == 1


def test_workers_that_fit_are_kept(monkeypatch, recwarn):
    monkeypatch.setattr(ngff, "_available_memory_mb", lambda: 10**6)
    assert fusion._workers_that_fit(8, (16, 256, 256), (16, 256, 256), np.uint16, flat=False) == 8
    assert not [w for w in recwarn if "fusion worker" in str(w.message)]


def test_fusion_still_runs_when_workers_are_lowered(tmp_path, monkeypatch):
    a = np.full((4, 32, 48), 10, np.uint16)
    b = np.full((4, 32, 48), 20, np.uint16)
    tiles = [Tile(a, (0, 0, 0), (1, 1, 1), "ch"), Tile(b, (0, 0, 40), (1, 1, 1), "ch")]
    monkeypatch.setattr(ngff, "_available_memory_mb", lambda: 1e-3)
    with pytest.warns(UserWarning, match="of the 4 workers fit"):
        zm = fuse_tiles(tiles, tmp_path / "out.ome.zarr", workers=4, executor="thread", progress=False,
                        transforms_path=False)
    out = zm.data[0].compute()
    assert out.shape == (4, 32, 88) and out[0, 0, 0] == 10 and out[0, 0, -1] == 20


def test_fusion_shards_default_to_their_own_target(tmp_path, monkeypatch):
    """fuse_tiles targets FUSION_SHARD_TARGET_MB (256 MB) unless shard_target_mb is given."""
    import zarr

    assert fusion.FUSION_SHARD_TARGET_MB == 256
    tiles = [Tile(np.full((16, 256, 272), v, np.uint16), (0, 0, x0), (1, 1, 1), "ch")
             for v, x0 in ((10, 0), (20, 256))]  # fused: 16 x 256 x 528 uint16, ~4.3 MB
    chunks = (1, 4, 64, 64)  # c, z, y, x: 32 KB chunks, so shards are made of many

    def shard_mb(name, **kw):
        fuse_tiles(tiles, tmp_path / name, progress=False, transforms_path=False, chunks=chunks,
                   drop_singleton=False, **kw)
        arr = zarr.open_group(str(tmp_path / name), mode="r")["0"]
        assert arr.shards is not None
        return np.prod(arr.shards) * arr.dtype.itemsize / 2**20

    monkeypatch.setattr(fusion, "FUSION_SHARD_TARGET_MB", 1.0)  # the default is what fuse_tiles uses
    default = shard_mb("default.ome.zarr")
    explicit = shard_mb("explicit.ome.zarr", shard_target_mb=4.0)  # an explicit value wins
    # "auto" shards come close to the target (whole chunks), not exactly
    assert default < 1.5 and 2.5 < explicit < 5 and explicit > 2 * default


def _plan(shape=(16, 256, 528), **tile_attrs):
    tiles = [Tile(np.full(shape, v, np.uint16), (0, 0, x0), (2.0, 0.5, 0.5), "ch", attrs=dict(tile_attrs))
             for v, x0 in ((10, 0), (20, 128))]
    return fusion.plan_fusion(tiles)


def test_automatic_workers_follow_the_cores_the_regions_and_the_memory():
    """workers=0: the rule shared by fuse_tiles, zarrsmith fuse and the napari widget."""
    plan = _plan()
    block = (1, 16, 64, 64)
    region, piece, n_regions = fusion.fusion_regions(plan, block)
    assert n_regions >= 1 and all(r % b == 0 for r, b in zip(region, block[1:]))
    big = 10**7  # MB: memory is no limit
    assert fusion.fusion_workers(plan, block, 0, cores=64, free_mb=big) == min(16, n_regions)  # at most 16
    assert fusion.fusion_workers(plan, block, 0, n_regions=100, cores=8, free_mb=big) == 4  # half of the cores
    assert fusion.fusion_workers(plan, block, 0, n_regions=2, cores=64, free_mb=big) == 2  # no more than regions
    assert fusion.fusion_workers(plan, block, 0, n_regions=100, cores=64, free_mb=1e-6) == 1  # memory
    assert fusion.fusion_workers(plan, block, 3, cores=64, free_mb=big) == 3  # a number given is kept


def test_default_fusion_levels_are_those_of_convert():
    from zarrsmith.utils.chunks import default_num_levels

    plan = _plan()
    md = {"axes": plan.axes, "size": [plan.shape], "dtype": str(plan.dtype)}
    assert fusion.default_fusion_levels(plan) == default_num_levels(md, 2)


def test_plan_voxel_size_is_none_where_a_tile_did_not_record_it():
    from zarrsmith.utils.axes import UNRECORDED_SCALE_KEY

    assert _plan().voxel_size == {"z": 2.0, "y": 0.5, "x": 0.5}
    assert _plan(**{UNRECORDED_SCALE_KEY: ("z",)}).voxel_size == {"z": None, "y": 0.5, "x": 0.5}
