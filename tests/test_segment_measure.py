"""project, threshold, mask, measure, filter-labels (library and command line), and
fuse-labels --tiles_csv."""
from __future__ import annotations

import json

import dask.array as da
import numpy as np
import pandas as pd
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.cli import zarrsmith as cli
from zarrsmith.utils.labels import (filter_labels, fit_shape, mask_channel, parse_limits, project, resolve_channel,
                                    threshold_labels)
from zarrsmith.utils.measure import measure_labels


def _image(path, axes="czyx", shape=(2, 8, 64, 96), levels=3, translated=True):
    data = np.full(shape, 100, np.uint16)
    index = {a: i for i, a in enumerate(axes)}
    for k, (y, x) in enumerate(((10, 10), (30, 60), (50, 20), (40, 80))):  # four bright cubes in channel 0
        sl = [slice(None)] * len(axes)
        sl[index["c"]] = 0
        sl[index["y"]], sl[index["x"]] = slice(y, y + 8), slice(x, x + 8 + 4 * k)  # different sizes
        sl[index["z"]] = slice(2, 6)
        data[tuple(sl)] = 1000
    spatial = [a for a in axes if a in "zyx"]
    md = {"axes": axes, "size": [shape], "chunksize": [tuple(1 if a in "tc" else min(n, 16) for a, n in zip(axes, shape))],
          "dtype": "uint16", "data_type": "intensity", "scales": [tuple({"z": 2.0, "y": 0.5, "x": 0.5}[a] for a in spatial)],
          "units": ("micrometer",) * len(spatial), "channel_names": ["DAPI", "GFP"], "channel_colors": ["0000FF", "00FF00"]}
    if translated:
        md["translations"] = [{a: {"z": 10.0, "y": 5.0, "x": 3.0}[a] for a in spatial}]
    m = mm.ArrayManager([da.from_array(data)], md)
    m.build_pyramid(num_levels=levels, downscale_factor=(1, 2, 2))
    m.to_zarr(str(path), drop_singleton=False, progress=False)
    return mm.ZarrManager(str(path), mode="r")


def test_fit_shape_and_resolve_channel():
    a = np.ones((4, 5))
    assert fit_shape(a, (3, 7)).shape == (3, 7) and fit_shape(a, (3, 7))[:, 5:].sum() == 0
    assert fit_shape(da.ones((4, 5)), (6, 3)).shape == (6, 3)
    assert resolve_channel("GFP", ["DAPI", "GFP"], 2) == 1 and resolve_channel("0", ["DAPI", "GFP"], 2) == 0
    with pytest.raises(ValueError, match="Unknown channel"):
        resolve_channel("RFP", ["DAPI", "GFP"], 2)
    assert parse_limits("solidity:0.8:;area::100") == {"solidity": (0.8, None), "area": (None, 100.0)}


def test_project_drops_the_axis_everywhere(tmp_path):
    image = _image(tmp_path / "img.ome.zarr")
    out = project(image)
    assert out.metadata["axes"] == "cyx"
    for src, got in zip(image.data, out.data):
        assert np.array_equal(np.asarray(src).max(axis=1), np.asarray(got))
    assert [tuple(s) for s in out.metadata["scales"]] == [tuple(s)[1:] for s in image.metadata["scales"]]
    assert all("z" not in t for t in out.metadata["translations"])
    assert project(image, method="mean").data[0].dtype == np.float32


def test_threshold_labels_every_level_with_the_same_ids(tmp_path):
    image = _image(tmp_path / "img.ome.zarr")
    labels = threshold_labels(image, "DAPI", level=0)
    assert labels.metadata["axes"] == "zyx" and labels.metadata["data_type"] == "label"
    ids0 = set(np.unique(np.asarray(labels.data[0]))) - {0}
    assert len(ids0) == 4 and len(labels.data) == 3
    assert set(np.unique(np.asarray(labels.data[2]))) - {0} <= ids0
    from_level = threshold_labels(image, 0, level=1, levels="from_level")
    assert len(from_level.data) == 2
    assert [tuple(s) for s in from_level.metadata["scales"]] == [tuple(s) for s in image.metadata["scales"][1:]]


def test_measure_filter_and_mask(tmp_path):
    image = _image(tmp_path / "img.ome.zarr")
    labels_am = threshold_labels(image, "DAPI")
    labels_am.to_zarr(str(tmp_path / "labels.ome.zarr"), drop_singleton=False, progress=False)
    labels = mm.ZarrManager(str(tmp_path / "labels.ome.zarr"), mode="r")

    table = measure_labels(image, labels, channels=["DAPI"], dataset_id="img")
    assert list(table.columns[:2]) == ["dataset_id", "label"] and len(table) == 4
    assert "mean_intensity_ch0_DAPI" in table.columns and (table["mean_intensity_ch0_DAPI"] == 1000).all()

    smallest = table["area"].min()
    filtered, kept = filter_labels(labels, table, {"area": (smallest + 1, None)})
    assert len(kept) == 3 and sorted(kept["label"]) == [1, 2, 3]
    assert set(kept["original_label"]) == set(table["label"][table["area"] > smallest])
    for level in filtered.data:
        assert set(np.unique(np.asarray(level))) <= {0, 1, 2, 3}

    masked = mask_channel(image, labels, "DAPI")
    assert masked.metadata["channel_names"] == ["DAPI", "GFP", "DAPI_clean"]
    level0 = np.asarray(masked.data[0])
    lab0 = np.asarray(labels.data[0])
    assert np.array_equal(level0[2], np.where(lab0 != 0, level0[0], 0))
    only = mask_channel(image, labels, 0, append=False)
    assert only.metadata["channel_names"] == ["DAPI_clean"] and only.data[0].shape[0] == 1


@pytest.mark.slow
def test_command_line_chain(tmp_path):
    img = tmp_path / "img.ome.zarr"
    _image(img, axes="tczyx", shape=(2, 2, 8, 64, 96))
    cli.main(["project", "-i", str(img), "-z", str(tmp_path / "mip.ome.zarr")])
    cli.main(["threshold", "-i", str(tmp_path / "mip.ome.zarr"), "-z", str(tmp_path / "lab.ome.zarr"), "-c", "DAPI"])
    cli.main(["measure", "-i", str(tmp_path / "mip.ome.zarr"), "-l", str(tmp_path / "lab.ome.zarr"),
              "-oc", str(tmp_path / "f.csv")])
    table = pd.read_csv(tmp_path / "f.csv")
    assert set(table["timepoint"]) == {0, 1} and (table.groupby("timepoint").size() == 4).all()
    cli.main(["filter-labels", "-l", str(tmp_path / "lab.ome.zarr"), "-tb", str(tmp_path / "f.csv"),
              "-z", str(tmp_path / "kept.ome.zarr"), "-oc", str(tmp_path / "kept.csv"), "--min_area", "1"])
    cli.main(["mask", "-i", str(tmp_path / "mip.ome.zarr"), "-l", str(tmp_path / "kept.ome.zarr"), "-c", "0",
              "-z", str(tmp_path / "clean.ome.zarr")])
    clean = mm.ZarrManager(str(tmp_path / "clean.ome.zarr"), mode="r")
    assert clean.metadata["axes"] == "tcyx" and clean.metadata["channel_names"][-1] == "DAPI_clean"
    with pytest.raises(FileExistsError):  # outputs are not replaced without --overwrite
        cli.main(["project", "-i", str(img), "-z", str(tmp_path / "mip.ome.zarr")])


def test_fuse_labels_offsets_from_the_tiles_csv(tmp_path):
    folder = tmp_path / "tiles"
    folder.mkdir()
    shape = (4, 16, 24)
    a, b = np.zeros(shape, np.uint16), np.zeros(shape, np.uint16)
    a[..., 0:10], a[..., 12:24] = 1, 2
    b[..., 0:8], b[..., 10:20] = 1, 2
    for name, data, x0 in (("img__tile_000001_labels.ome.zarr", a, 0.0), ("img__tile_000002_labels.ome.zarr", b, 16.0)):
        md = {"axes": "zyx", "scales": [(1.0, 1.0, 1.0)], "data_type": "label", "translations": {"z": 0.0, "y": 0.0, "x": x0}}
        mm.ArrayManager(data, md).to_zarr(str(folder / name), progress=False)
    pd.DataFrame({"dataset_id": ["img__tile_000001", "img__tile_000002"], "tile_index": [1, 2],
                  "segmented_tile_path": [f"/work/x/{p.name}" for p in sorted(folder.iterdir())]}
                 ).to_csv(tmp_path / "tiles.csv", index=False)
    (tmp_path / "mapping.json").write_text(json.dumps({"2001": 1002}))
    out = tmp_path / "merged.ome.zarr"
    cli.main(["fuse-labels", "-i", str(folder), "-z", str(out), "--tiles_csv", str(tmp_path / "tiles.csv"),
              "--label_offset_step", "1000", "-lm", str(tmp_path / "mapping.json"), "-nl", "1"])
    merged = np.asarray(mm.ZarrManager(str(out), mode="r").data[0])
    assert set(np.unique(merged)) == {0, 1001, 1002, 2002}
    assert (tmp_path / "merged.label_offsets.csv").exists()
    with pytest.raises(SystemExit):  # the step goes with the tiles CSV
        cli.main(["fuse-labels", "-i", str(folder), "-z", str(out), "--tiles_csv", str(tmp_path / "tiles.csv")])
