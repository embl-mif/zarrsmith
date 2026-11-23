"""Tile plans (zarrsmith.utils.tiling) and writing tiles into an output laid out from one."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from zarrsmith.microscope_manager import ZarrManager
from zarrsmith.utils.allocate import allocate_zarr
from zarrsmith.utils.tiling import plan_tiles
from tests.test_allocate_pyramid import FORMAT_IDS, FORMATS, _make_template, _zyx


def test_an_axis_without_a_tile_size_is_not_split():
    plan = plan_tiles({"y": 100, "x": 50}, tile_size={"y": 40}, overlap=8)
    assert [t.bounds["y"] for t in plan.tiles] == [(0, 40), (32, 72), (64, 100)]
    assert {t.bounds["x"] for t in plan.tiles} == {(0, 50)}
    assert plan.stride == {"y": 32, "x": 50} and plan.overlap == {"y": 8, "x": 0}

    whole = plan_tiles({"t": 3, "z": 10, "y": 100, "x": 50})
    assert len(whole.tiles) == 1 and whole.tiles[0].bounds == {"z": (0, 10), "y": (0, 100), "x": (0, 50)}
    # a tile size larger than the ROI is one tile as well
    assert len(plan_tiles({"y": 100, "x": 50}, tile_size=500).tiles) == 1


def test_a_too_small_last_tile_is_folded_into_the_one_before():
    plan = plan_tiles({"y": 65, "x": 8}, tile_size={"y": 32})
    assert [t.bounds["y"] for t in plan.tiles] == [(0, 32), (32, 65)]
    plan = plan_tiles({"y": 70, "x": 8}, tile_size={"y": 32}, halo=8)  # last tile (64, 70) < halo
    assert [t.bounds["y"] for t in plan.tiles] == [(0, 32), (32, 70)]


def test_tiles_are_numbered_from_1_in_z_y_x_order():
    plan = plan_tiles({"z": 4, "y": 4, "x": 4}, tile_size=2)
    assert [t.index for t in plan.tiles] == list(range(1, 9))
    assert plan.tiles[1].grid == {"z": 0, "y": 0, "x": 1} and plan.tiles[0].tile_id == "tile_000001"
    assert plan.neighbors(plan.tiles[0]) == {"z": plan.tiles[4], "y": plan.tiles[2], "x": plan.tiles[1]}
    assert plan.neighbors(plan.tiles[-1]) == {"z": None, "y": None, "x": None}


def test_cores_cover_the_roi_once_and_start_on_the_stride():
    plan = plan_tiles({"y": 103, "x": 90}, tile_size=30, overlap=6, roi={"y": (3, None), "x": (10, 80)})
    assert plan.origin == {"y": 3, "x": 10} and plan.output_shape == {"y": 100, "x": 70}
    covered = np.zeros((100, 70), dtype=int)
    for tile in plan.tiles:
        region = plan.write_region(tile)
        covered[region["y"], region["x"]] += 1
        assert region["y"].start % plan.stride["y"] == 0 and region["x"].start % plan.stride["x"] == 0
    assert (covered == 1).all()


def test_physical_units_are_converted_with_the_level_geometry():
    plan = plan_tiles({"y": 200, "x": 200}, tile_size=20.0, overlap=2.0, halo=1.1,
                      roi={"y": (20.0, 70.0)}, scale={"y": 0.5, "x": 0.5},
                      translation={"y": 10.25, "x": 0.0}, physical=True)
    assert plan.tile_size == {"y": 40, "x": 40} and plan.overlap == {"y": 4, "x": 4}
    assert plan.halo == {"y": 3, "x": 3}
    # pixel i is centred on 10.25 + 0.5 i, so covers [10 + 0.5 i, 10.5 + 0.5 i): [20, 70) um is pixels 20..119
    assert plan.roi["y"] == (20, 120)


def test_rows_describe_every_tile():
    rows = plan_tiles({"z": 4, "y": 10, "x": 10}, tile_size={"y": 6}, overlap=2, level=1).rows()
    assert rows[0] == {"tile_index": 1, "tile_id": "tile_000001", "level": 1,
                       "z_min": 0, "z_max": 4, "y_min": 0, "y_max": 6, "x_min": 0, "x_max": 10,
                       "tile_size_z": 4, "tile_overlap_z": 0, "tile_size_y": 6, "tile_overlap_y": 2,
                       "tile_size_x": 10, "tile_overlap_x": 0}


def test_allocate_kwargs_make_each_core_a_whole_write_block():
    plan = plan_tiles({"t": 5, "z": 8, "y": 100, "x": 100}, tile_size={"y": 40, "x": 40}, overlap=8)
    # default: one timepoint per block, so jobs can be split per timepoint
    v3 = plan.allocate_kwargs("tzyx", zarr_format=3)
    assert v3["shards"] == (1, 8, 32, 32) and v3["chunk_size"] == (1, 8, 32, 32)
    v2 = plan.allocate_kwargs("tzyx", zarr_format=2)
    assert v2["chunk_size"] == (1, 8, 32, 32) and v2["shards"] is None
    # split="" keeps every timepoint of a tile in one block
    whole = plan.allocate_kwargs("tzyx", split="")
    assert whole["shards"] == (5, 8, 32, 32) and whole["chunk_size"] == (1, 8, 32, 32)
    assert plan.allocate_kwargs("tzyx", zarr_format=2, split="")["chunk_size"] == (5, 8, 32, 32)
    # no t axis: nothing to split
    assert plan.allocate_kwargs("zyx")["shards"] == (8, 32, 32)
    # inner chunks divide the stride
    big = plan_tiles({"y": 4000, "x": 4000}, tile_size=1100, overlap=100).allocate_kwargs("yx")
    assert big["shards"] == (1000, 1000) and big["chunk_size"] == (250, 250)

    # channels can be split too
    assert plan.allocate_kwargs("tczyx", split="tc", sizes={"c": 3})["shards"] == (1, 1, 8, 32, 32)
    # cropping t sizes the block to the kept timepoints
    cropped = plan.allocate_kwargs("tzyx", split="", crop={"t": (1, 4)})
    assert cropped["shards"][0] == 3 and cropped["crop"] == {"t": (1, 4), "z": (0, 8), "y": (0, 100), "x": (0, 100)}
    with pytest.raises(ValueError, match="untiled axes"):
        plan.allocate_kwargs("tzyx", split="y")
    with pytest.raises(ValueError, match="untiled axes"):
        plan.allocate_kwargs("tzyx", crop={"y": (0, 10)})


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_jobs_split_per_tile_and_timepoint(tmp_path, zarr_format, ngff):
    template = tmp_path / "raw.ome.zarr"
    _make_template(template, zarr_format, ngff)  # 2 timepoints
    plan = ZarrManager(str(template)).plan_tiles(tile_size={"y": 24, "x": 24}, overlap=4)

    def allocate(name, **kw):
        out = tmp_path / name
        allocate_zarr(out, template=template, data_type="label", dtype="uint32",
                      **plan.allocate_kwargs("tzyx", zarr_format=zarr_format, **kw))
        return out, ZarrManager(str(out), mode="r+")

    out, seg = allocate("split.ome.zarr")  # split="t" is the default
    for t in range(2):
        for tile in plan.tiles:
            region = plan.write_region(tile)
            shape = (1,) + tuple(region[ax].stop - region[ax].start for ax in "zyx")
            seg.write_label_region(np.full(shape, 10 * t + tile.index, dtype=np.uint32),
                                   t=slice(t, t + 1), aligned=True, **region)
    written = np.asarray(ZarrManager(str(out)).data[0])
    first = plan.tiles[0]
    assert written[0, 0, 0, 0] == first.index and written[1, 0, 0, 0] == 10 + first.index
    assert (written > 0).all()

    # with split="" a block holds both timepoints, so a per-timepoint write is refused
    _, whole = allocate("whole.ome.zarr", split="")
    region = plan.write_region(first)
    shape = (1,) + tuple(region[ax].stop - region[ax].start for ax in "zyx")
    with pytest.raises(ValueError, match="not aligned"):
        whole.write_label_region(np.ones(shape, dtype=np.uint32), t=slice(0, 1), aligned=True, **region)

    # a crop of t keeps those timepoints
    _, cropped = allocate("cropped.ome.zarr", crop={"t": (1, 2)})
    assert cropped.metadata["size"][0][0] == 1


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_tiles_write_their_cores_in_parallel_safe_blocks(tmp_path, zarr_format, ngff):
    """The tiled Cellpose flow: plan tiles, allocate the output from the plan, and every
    tile writes its core with aligned=True; together they fill the ROI exactly."""
    template = tmp_path / "raw.ome.zarr"
    _make_template(template, zarr_format, ngff)  # tczyx (2, 2, 8, 64, 64), scale (2, .5, .5), origin (10, 20, 30)
    raw = ZarrManager(str(template))
    plan = raw.plan_tiles(tile_size={"y": 24, "x": 24}, overlap=4, roi={"y": (4, 60)})
    assert plan.counts == {"z": 1, "y": 3, "x": 3}

    out = tmp_path / "seg.ome.zarr"
    allocate_zarr(out, template=template, data_type="label", dtype="uint32",
                  **plan.allocate_kwargs("tzyx", zarr_format=zarr_format))
    seg = ZarrManager(str(out), mode="r+")
    assert seg.metadata["size"] == [(2, 8, 56, 64)]
    assert seg.metadata["translations"] == [_zyx(10.0, 20.0 + 4 * 0.5, 30.0)]

    for tile in plan.tiles:
        region = plan.write_region(tile)
        shape = (2,) + tuple(region[ax].stop - region[ax].start for ax in "zyx")
        seg.write_label_region(np.full(shape, tile.index, dtype=np.uint32), aligned=True, **region)

    written = np.asarray(ZarrManager(str(out)).data[0])
    expected = np.zeros((56, 64), dtype=np.uint32)
    for tile in plan.tiles:
        region = plan.write_region(tile)
        expected[region["y"], region["x"]] = tile.index
    assert (written == expected).all() and (expected > 0).all()


# ---------------------------------------------------------------------------
# max_tile_voxels: cut only when a tile would not fit
# ---------------------------------------------------------------------------

def _core_volume(plan):
    return sum(np.prod([hi - lo for lo, hi in plan.core(t).values()]) for t in plan.tiles)


def test_an_image_that_fits_stays_one_tile():
    plan = plan_tiles({"z": 50, "y": 1000, "x": 1000}, halo=8, max_tile_voxels=60_000_000)
    assert len(plan.tiles) == 1


def test_a_large_volume_is_cut_in_yx_into_few_aligned_tiles():
    shape = {"t": 3, "z": 96, "y": 6000, "x": 5000}
    plan = plan_tiles(shape, overlap=32, halo=16, max_tile_voxels=400_000_000)
    assert plan.counts == {"z": 1, "y": 3, "x": 3}  # z short: kept whole
    assert plan.stride["y"] % 128 == 0 and plan.stride["x"] % 128 == 0  # inner chunks of 128 fit
    assert plan.max_load_voxels <= 400_000_000  # 3 x 3 even tiles: about 350 M voxels each
    assert _core_volume(plan) == 96 * 6000 * 5000  # cores still cover the image once
    # the budget is per timepoint: t is not tiled
    assert all(set(t.bounds) == {"z", "y", "x"} for t in plan.tiles)


def test_rounding_does_not_cost_a_row_of_tiles():
    """A 301 x 1880 x 2566 stack, ~300 M voxels per job: 2 x 4 tiles fit (about 234 M
    voxels each), with less area cut than 3 x 3; rounding an even share of the budget
    down used to give 3 x 4."""
    shape = {"t": 1, "c": 4, "z": 301, "y": 1880, "x": 2566}
    plan = plan_tiles(shape, overlap=64, halo=20, max_tile_voxels=300_000_000)
    assert plan.counts == {"z": 1, "y": 2, "x": 4}
    assert plan.max_load_voxels <= 300_000_000
    assert _core_volume(plan) == 301 * 1880 * 2566


def _key(counts, lengths):
    area = sum((counts[ax] - 1) * np.prod([lengths[o] for o in lengths if o != ax]) for ax in lengths)
    return area, int(np.prod(list(counts.values())))


@pytest.mark.parametrize("shape, overlap, halo, budget", [
    ({"y": 37, "x": 53}, 3, 2, 300),
    ({"y": 60, "x": 41}, 0, 5, 500),
    ({"y": 45, "x": 45}, 6, 1, 900),
    ({"z": 9, "y": 23, "x": 31}, 2, 1, 700),
    ({"z": 14, "y": 17, "x": 12}, 1, 2, 1200),
])
def test_the_search_finds_the_best_counts(shape, overlap, halo, budget):
    """Same answer as trying every tile count along every axis."""
    from zarrsmith.utils import tiling

    plan = plan_tiles(shape, overlap=overlap, halo=halo, max_tile_voxels=budget)
    assert plan.max_load_voxels <= budget
    best = None
    for counts in itertools.product(*(range(1, n - overlap + 1) for n in shape.values())):
        opts = [tiling._axis_option(0, n, n, c, overlap, halo, tiling.DEFAULT_INNER_CHUNK[ax])
                for (ax, n), c in zip(shape.items(), counts)]
        if np.prod([o[2] for o in opts]) <= budget:
            key = _key({ax: o[1] for ax, o in zip(shape, opts)}, shape)
            best = key if best is None or key < best else best
    assert _key(plan.counts, shape) == best


def test_resize_shrinks_the_tiles():
    shape = {"z": 96, "y": 6000, "x": 5000}
    plain = plan_tiles(shape, max_tile_voxels=400_000_000)
    doubled = plan_tiles(shape, max_tile_voxels=400_000_000, resize=2.0)  # e.g. Cellpose, diameter 15
    assert len(doubled.tiles) > len(plain.tiles)
    assert doubled.max_load_voxels * 8 <= 400_000_000


def test_a_long_axis_is_cut_too():
    plan = plan_tiles({"z": 4000, "y": 512, "x": 512}, max_tile_voxels=256 * 512 * 512)
    assert plan.counts["z"] > 1 and len(plan.tiles) == 16  # 16 tiles of 256 x 512 x 512 voxels' worth
    assert plan.max_load_voxels <= 256 * 512 * 512


def test_small_budgets_drop_the_alignment():
    plan = plan_tiles({"y": 1000, "x": 1000}, overlap=4, max_tile_voxels=10_000)
    assert plan.max_load_voxels <= 10_000
    assert _core_volume(plan) == 1000 * 1000


def test_a_given_tile_size_is_kept_and_counted():
    plan = plan_tiles({"z": 200, "y": 4000, "x": 4000}, tile_size={"z": 40}, max_tile_voxels=100_000_000)
    assert plan.tile_size["z"] == 40 and plan.counts["y"] > 1
    assert plan.max_load_voxels <= 100_000_000
    with pytest.raises(ValueError, match="more than max_tile_voxels"):
        plan_tiles({"y": 1000, "x": 1000}, tile_size=500, max_tile_voxels=1000)
    with pytest.raises(ValueError, match="too small"):
        plan_tiles({"y": 1000, "x": 1000}, overlap=20, halo=10, max_tile_voxels=1000)


def test_zarr_manager_plans_within_a_budget(tmp_path):
    path = tmp_path / "raw.ome.zarr"
    _make_template(path, *FORMATS[1])  # zyx (8, 64, 64) per timepoint and channel
    plan = ZarrManager(str(path)).plan_tiles(max_tile_voxels=8 * 32 * 32)
    assert len(plan.tiles) > 1 and plan.max_load_voxels <= 8 * 32 * 32
