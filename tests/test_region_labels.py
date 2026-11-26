"""Region.to_label_image: labels segmented from a region read with ZarrManager.read_region,
written as a standalone label image placed at the region and following the source image."""
from __future__ import annotations

import dask.array as da
import numpy as np
import pytest

import zarrsmith.microscope_manager as mm


def _image(path, *, axes="zyx", shape=(8, 64, 96), levels=3, zarr_format=3, translation=(10.0, 5.0, 3.0)):
    data = np.random.default_rng(0).integers(0, 1000, shape).astype(np.uint16)
    spatial = [a for a in axes if a in "zyx"]
    sc = dict(zip("zyx", (2.0, 0.5, 0.5)))
    tr = dict(zip("zyx", translation))
    md = {"axes": axes, "size": [shape], "chunksize": [tuple(min(s, 16) for s in shape)], "dtype": "uint16",
          "data_type": "intensity", "scales": [tuple(sc[a] for a in spatial)], "units": ("micrometer",) * len(spatial),
          "translations": [{a: tr[a] for a in spatial}]}
    if "c" in axes:
        md["channel_names"] = [f"ch{i}" for i in range(shape[axes.index("c")])]
    m = mm.ArrayManager([da.from_array(data)], md)
    if levels > 1:
        m.build_pyramid(num_levels=levels, downscale_factor=(1, 2, 2))
    m.to_zarr(str(path), zarr_format=zarr_format, ngff_version="0.5" if zarr_format == 3 else "0.4",
              drop_singleton=False, progress=False)
    return mm.ZarrManager(str(path), mode="r")


def _labels(block):
    """Stand-in segmentation of a block: ids by threshold, so the expected labels are known."""
    return (block > 500).astype(np.uint16) * 7


@pytest.mark.parametrize("given", ["block", "core"])
def test_labels_of_a_tile_are_placed_at_the_tile(tmp_path, given):
    src = _image(tmp_path / "img.ome.zarr")
    region = src.read_region(0, z=(0, 8), y=(40, 64), x=(30, 90), halo=8)
    labels = _labels(region.data if given == "block" else region.core)
    out = region.to_label_image(labels, tmp_path / "tile.ome.zarr", num_levels=1)
    assert out.metadata["data_type"] == "label" and len(out.data) == 1
    got = np.asarray(out.data[0])
    assert got.dtype == np.uint32 and np.array_equal(got, _labels(region.core))
    assert out.metadata["translations"][0] == {"z": 10.0, "y": 5.0 + 40 * 0.5, "x": 3.0 + 30 * 0.5}
    assert tuple(out.metadata["scales"][0]) == (2.0, 0.5, 0.5)


def test_pyramid_and_format_follow_the_source(tmp_path):
    src = _image(tmp_path / "img.ome.zarr", levels=4, zarr_format=2)
    region = src.read_region(1)  # the whole of level 1
    out = region.to_label_image(_labels(region.data), tmp_path / "labels.ome.zarr")
    assert out.metadata["zarr_format"] == 2
    assert [tuple(s) for s in out.metadata["scales"]] == [tuple(s) for s in src.metadata["scales"][1:]]
    for level in range(1, len(out.data)):  # sampled: no new ids
        assert set(np.unique(np.asarray(out.data[level]))) <= {0, 7}


def test_one_timepoint_of_a_time_series(tmp_path):
    src = _image(tmp_path / "t.ome.zarr", axes="tczyx", shape=(3, 2, 8, 32, 32), levels=1)
    region = src.read_region(0, t=1, y=(0, 16))
    block = region.data[0, 0]  # one timepoint, one channel: zyx
    out = region.to_label_image(_labels(block), tmp_path / "t_labels.ome.zarr")
    assert out.metadata["axes"] == "zyx" and np.asarray(out.data[0]).shape == (8, 16, 32)
    with_t = region.to_label_image(_labels(region.data[:, 0]), tmp_path / "t_labels2.ome.zarr")
    assert with_t.metadata["axes"] == "tzyx"


def test_refuses_wrong_shapes_and_ids_that_do_not_fit(tmp_path):
    src = _image(tmp_path / "img.ome.zarr", levels=1)
    region = src.read_region(0, y=(0, 32), x=(0, 32), halo=4)
    with pytest.raises(ValueError, match="expected the core"):
        region.to_label_image(np.zeros((8, 10, 10), np.uint16), tmp_path / "a.ome.zarr")
    with pytest.raises(ValueError, match="do not fit"):
        region.to_label_image(np.full((8, 32, 32), 70000, np.uint32), tmp_path / "b.ome.zarr", dtype="uint16")
    with pytest.raises(ValueError, match="integer dtype"):
        region.to_label_image(np.zeros((8, 32, 32)), tmp_path / "c.ome.zarr", dtype="float32")
