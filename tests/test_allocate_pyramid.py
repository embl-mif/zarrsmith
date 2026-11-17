"""Translations, extra attributes, ``allocate``, in-place pyramids and aligned region writes,
on both zarr v2 / NGFF 0.4 and zarr v3 / NGFF 0.5."""
from __future__ import annotations

import json
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Label, Multiscales, Reader

from zarrsmith.cli.zarrsmith import main
from zarrsmith.microscope_manager import ZarrManager
from zarrsmith.utils.allocate import allocate_zarr
from zarrsmith.utils.downsampling import downsample, downsample_numpy
from zarrsmith.utils.translation import pyramid_translations
from zarrsmith.utils.ngff import ZarrWriteConfig
from zarrsmith.utils.to_zarr import to_zarr

FORMATS = [(2, "0.4"), (3, "0.5")]
FORMAT_IDS = ["v2-0.4", "v3-0.5"]


def _zyx(z, y, x):
    """A translation as zarrsmith stores it: one dict per level, keyed by spatial axis."""
    return {"z": z, "y": y, "x": x}


def _make_template(path: Path, zarr_format: int, ngff: str, *, translation=(10.0, 20.0, 30.0), shards=None):
    """A tczyx uint16 image (2, 2, 8, 64, 64) with one level, scale (2, 0.5, 0.5)."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 1000, size=(2, 2, 8, 64, 64), dtype=np.uint16)
    metadata = {
        "axes": "tczyx",
        "size": [data.shape],
        "chunksize": [(1, 1, 4, 32, 32)],
        "scales": [(2.0, 0.5, 0.5)],
        "translations": None if translation is None else [translation],
        "units": ("micrometer",) * 3,
        "time_increment": 60.0,
        "time_increment_unit": "second",
        "channel_names": ["A", "B"],
        "channel_colors": ["FF0000", "00FF00"],
        "dtype": "uint16",
        "name": "template",
    }
    to_zarr(
        path,
        [da.from_array(data, chunks=(1, 1, 4, 32, 32))],
        metadata,
        config=ZarrWriteConfig(
            zarr_format=zarr_format, ngff_version=ngff, shards=shards,
            drop_singleton=False, progress=False,
        ),
    )
    return data


def _napari_nodes(path: Path):
    """The nodes ome-zarr-py's reader (used by napari-ome-zarr) finds in ``path``."""
    return list(Reader(parse_url(str(path)))())


def _transforms(node, level: int):
    return {t["type"]: t[t["type"]] for t in node.metadata["coordinateTransformations"][level]}


# ---------------------------------------------------------------------------
# Downsampling methods
# ---------------------------------------------------------------------------

def test_mean_rounds_integers_and_pads_edges():
    arr = np.array([[0, 2, 4], [2, 4, 6]], dtype=np.uint8)
    out = downsample_numpy(arr, (2, 2), (0, 1), "mean")
    # [[0,2],[2,4]] -> 2 ; [[4,4],[6,6]] (edge-padded) -> 5
    np.testing.assert_array_equal(out, [[2, 5]])
    assert out.dtype == np.uint8


def test_mode_keeps_the_most_frequent_label_and_smallest_on_ties():
    arr = np.array([[1, 1, 3, 4], [2, 1, 4, 3]], dtype=np.uint32)
    out = downsample_numpy(arr, (2, 2), (0, 1), "mode")
    np.testing.assert_array_equal(out, [[1, 3]])


@pytest.mark.parametrize("method", ["mean", "mode", "nearest"])
def test_dask_and_numpy_downsampling_agree(method):
    arr = np.random.default_rng(1).integers(0, 5, size=(3, 13, 17), dtype=np.uint16)
    expected = downsample_numpy(arr, (2, 2, 2), (0, 1, 2), method)
    got = downsample(da.from_array(arr, chunks=(2, 5, 7)), (2, 2, 2), (0, 1, 2), method).compute()
    np.testing.assert_array_equal(got, expected)


def test_pyramid_translations_follow_the_ngff_06_convention():
    scales = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]
    # The 0.6 multiscales example: 0.5 base pixel, levels at 0.25 and 0.75.
    assert pyramid_translations({"y": 0.0, "x": 0.0}, scales[0], scales, "yx", "mean") == [
        {"y": 0.0, "x": 0.0}, {"y": 0.25, "x": 0.25}, {"y": 0.75, "x": 0.75}]
    assert pyramid_translations({"y": 10.0}, scales[0], scales, "yx", "nearest") == [
        {"y": 10.0, "x": 0.0}] * 3
    assert pyramid_translations(None, scales[0], scales, "yx", "nearest") is None
    assert pyramid_translations(None, scales[0], scales, "yx", "mode")[2] == {"y": 0.75, "x": 0.75}


# ---------------------------------------------------------------------------
# Translations and extra attributes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_translation_is_written_after_scale_and_survives_a_copy(tmp_path, zarr_format, ngff):
    src = tmp_path / "src.ome.zarr"
    _make_template(src, zarr_format, ngff)

    datasets = zarr.open_group(str(src), mode="r").attrs.asdict()
    if zarr_format == 3:
        datasets = datasets["ome"]
    transforms = datasets["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    assert [t["type"] for t in transforms] == ["scale", "translation"]
    assert transforms[1]["translation"] == [0.0, 0.0, 10.0, 20.0, 30.0]

    z = ZarrManager(str(src))
    assert z.metadata["translations"] == [_zyx(10.0, 20.0, 30.0)]
    z.to_zarr(tmp_path / "copy.ome.zarr", ngff_version=ngff, zarr_format=zarr_format,
              drop_singleton=False, progress=False)
    assert ZarrManager(str(tmp_path / "copy.ome.zarr")).metadata["translations"] == [_zyx(10.0, 20.0, 30.0)]


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_no_translation_is_written_when_there_is_none(tmp_path, zarr_format, ngff):
    src = tmp_path / "src.ome.zarr"
    _make_template(src, zarr_format, ngff, translation=None)
    z = ZarrManager(str(src))
    assert z.metadata["translations"] is None
    node = _napari_nodes(src)[0]
    assert "translation" not in _transforms(node, 0)


def test_crop_moves_the_translation(tmp_path):
    src = tmp_path / "src.ome.zarr"
    _make_template(src, 3, "0.5")
    z = ZarrManager(str(src))
    z.subset_dataset(Z=slice(2, 6), Y=slice(16, 48))
    assert z.metadata["translations"][0] == pytest.approx(_zyx(10.0 + 2 * 2.0, 20.0 + 16 * 0.5, 30.0))


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_lazy_pyramid_with_mean_writes_translations(tmp_path, zarr_format, ngff):
    src = tmp_path / "src.ome.zarr"
    data = _make_template(src, zarr_format, ngff)
    z = ZarrManager(str(src))
    z.build_pyramid(num_levels=2, method=None)  # None: mean for this intensity image
    out = tmp_path / "out.ome.zarr"
    z.to_zarr(out, ngff_version=ngff, zarr_format=zarr_format, drop_singleton=False,
              shards=None, progress=False)
    back = ZarrManager(str(out))
    assert back.metadata["translations"] == [_zyx(10.0, 20.0, 30.0), pytest.approx(_zyx(11.0, 20.25, 30.25))]
    np.testing.assert_array_equal(
        np.asarray(back.data[1]), downsample_numpy(data, (2, 2, 2), (2, 3, 4), "mean")
    )


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_extra_attributes_sit_next_to_the_ngff_metadata(tmp_path, zarr_format, ngff):
    out = tmp_path / "out.ome.zarr"
    allocate_zarr(out, template=None, size=(4, 32, 32), axes="zyx", zarr_format=zarr_format,
                  extra_attributes={"t_range": [0, 10]})
    attrs = zarr.open_group(str(out), mode="r").attrs.asdict()
    assert attrs["t_range"] == [0, 10]
    if zarr_format == 3:
        assert "t_range" not in attrs["ome"]
        assert "multiscales" in attrs["ome"]
    else:
        assert "multiscales" in attrs and "ome" not in attrs

    z = ZarrManager(str(out))
    assert z.metadata["extra_attributes"] == {"t_range": [0, 10]}
    z.to_zarr(tmp_path / "copy.ome.zarr", ngff_version=ngff, zarr_format=zarr_format, progress=False)
    assert zarr.open_group(str(tmp_path / "copy.ome.zarr"), mode="r").attrs["t_range"] == [0, 10]


# ---------------------------------------------------------------------------
# allocate -> write regions -> pyramid -> read like napari
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_allocate_write_pyramid_read_back(tmp_path, zarr_format, ngff):
    template = tmp_path / "raw.ome.zarr"
    data = _make_template(template, zarr_format, ngff, shards=None if zarr_format == 2 else (1, 1, 8, 64, 64))
    out = tmp_path / "prob.ome.zarr"

    # The format follows the template without --zarr_format.
    root = allocate_zarr(out, template=template, dtype="float32", num_channels=1,
                         channel_names=["foreground"], crop={"z": (0, 8)})
    assert root.metadata.zarr_format == zarr_format
    level0 = root["0"]
    assert level0.shape == (2, 1, 8, 64, 64)
    if zarr_format == 2:
        assert level0.shards is None
        assert level0.metadata.dimension_separator == "/"
        assert type(level0.compressors[0]).__name__ == "Blosc"

    # Fill level 0 tile by tile, every tile on the block grid.
    z = ZarrManager(str(out), mode="r+")
    block = level0.shards or level0.chunks
    values = data[:, :1].astype(np.float32) / 1000
    for t in range(2):
        for y0 in range(0, 64, block[3]):
            for x0 in range(0, 64, block[4]):
                tile = values[t:t + 1, :, :, y0:y0 + block[3], x0:x0 + block[4]]
                z.write_image_region(tile, t=slice(t, t + 1), y=slice(y0, y0 + block[3]),
                                     x=slice(x0, x0 + block[4]), aligned=True)

    z.build_pyramid_inplace(num_levels=3, progress=False)
    assert len(z.data) == 3
    np.testing.assert_allclose(np.asarray(z.data[0]), values)
    expected1 = downsample_numpy(values, (2, 2, 2), (2, 3, 4), "mean")
    np.testing.assert_allclose(np.asarray(z.data[1]), expected1, rtol=1e-6)

    node = _napari_nodes(out)[0]
    assert node.load(Multiscales) is not None
    assert [tuple(level.shape) for level in node.data] == [(2, 1, 8, 64, 64), (2, 1, 4, 32, 32), (2, 1, 2, 16, 16)]
    level1 = _transforms(node, 1)
    assert level1["scale"] == [60.0, 1.0, 4.0, 1.0, 1.0]
    # mean: level 1 pixel centres sit half a level-0 pixel further.
    assert level1["translation"] == pytest.approx([0.0, 0.0, 11.0, 20.25, 30.25])
    assert _transforms(node, 0)["translation"] == [0.0, 0.0, 10.0, 20.0, 30.0]


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_label_allocation_loads_as_labels(tmp_path, zarr_format, ngff):
    template = tmp_path / "raw.ome.zarr"
    _make_template(template, zarr_format, ngff)
    out = tmp_path / "seg.ome.zarr"
    allocate_zarr(out, template=template, data_type="label", dtype="uint32")

    attrs = zarr.open_group(str(out), mode="r").attrs.asdict()
    meta = attrs["ome"] if zarr_format == 3 else attrs
    assert meta["image-label"]["version"] == ngff

    z = ZarrManager(str(out), mode="r+")
    assert z.metadata["axes"] == "tzyx"
    labels = np.zeros((2, 8, 64, 64), dtype=np.uint32)
    labels[:, 2:6, 10:40, 10:40] = 5
    z.write_label_region(labels, aligned=True)
    z.build_pyramid_inplace(num_levels=2, method="mode", progress=False)
    assert set(np.unique(np.asarray(z.data[1]))) == {0, 5}

    node = _napari_nodes(out)[0]
    assert node.load(Label) is not None  # napari-ome-zarr opens it as a labels layer
    # mode moves the origin like mean
    assert _transforms(node, 1)["translation"] == pytest.approx([0.0, 11.0, 20.25, 30.25])

    with pytest.raises(ValueError, match="label"):
        z.build_pyramid_inplace(num_levels=2, method="mean", progress=False)


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_nested_label_group_pyramid(tmp_path, zarr_format, ngff):
    template = tmp_path / "raw.ome.zarr"
    _make_template(template, zarr_format, ngff)
    z = ZarrManager(str(template), mode="r+")
    md = dict(z.metadata, axes="tzyx", size=[(2, 8, 64, 64)], chunksize=[(1, 4, 32, 32)], dtype="uint16")
    z.create_empty_group("nuclei", md, data_type="label")
    z.write_label_region(np.ones((2, 8, 64, 64), dtype=np.uint16), group="labels/nuclei")
    z.build_pyramid_inplace(group="labels/nuclei", num_levels=3, progress=False)
    assert [arr.shape for arr in z.labels["nuclei"].data] == [(2, 8, 64, 64), (2, 4, 32, 32), (2, 2, 16, 16)]
    labels_node = [n for n in _napari_nodes(template) if n.load(Label) is not None]
    assert labels_node
    # nearest keeps the origin
    assert _transforms(labels_node[0], 2)["translation"] == [0.0, 10.0, 20.0, 30.0]


def test_v3_template_converts_to_v2_output(tmp_path):
    template = tmp_path / "raw.ome.zarr"
    _make_template(template, 3, "0.5", shards=(1, 1, 8, 64, 64))
    out = tmp_path / "v2.ome.zarr"
    root = allocate_zarr(out, template=template, zarr_format=2)
    assert root.metadata.zarr_format == 2
    assert root["0"].shards is None
    assert root["0"].metadata.dimension_separator == "/"
    attrs = root.attrs.asdict()
    assert "ome" not in attrs and attrs["multiscales"][0]["version"] == "0.4"
    z = ZarrManager(str(out))
    assert z.metadata["translations"] == [_zyx(10.0, 20.0, 30.0)]
    assert z.metadata["channel_names"] == ["A", "B"]


def test_v2_refuses_explicit_shards_and_drops_auto(tmp_path):
    with pytest.raises(ValueError, match="zarr v2 has no shards"):
        allocate_zarr(tmp_path / "a.ome.zarr", size=(4, 32, 32), axes="zyx", zarr_format=2, shards=(4, 32, 32))
    with pytest.warns(UserWarning, match="not supported"):
        root = allocate_zarr(tmp_path / "b.ome.zarr", size=(4, 32, 32), axes="zyx", zarr_format=2, shards="auto")
    assert root["0"].shards is None


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_rerunning_pyramid_overwrites_cleanly(tmp_path, zarr_format, ngff):
    out = tmp_path / "img.ome.zarr"
    allocate_zarr(out, size=(1, 16, 128, 128), axes="tzyx", zarr_format=zarr_format,
                  chunk_size=(1, 8, 32, 32), extra_attributes={"t_range": [3, 4]})
    z = ZarrManager(str(out), mode="r+")
    z.write_image_region(np.full((1, 16, 128, 128), 7, dtype=np.uint16))
    z.build_pyramid_inplace(num_levels=4, progress=False)
    assert len(z.data) == 4
    z.build_pyramid_inplace(num_levels=2, downscale_factor=(1, 2, 2), progress=False)
    assert len(z.data) == 2
    assert z.data[1].shape == (1, 16, 64, 64)
    assert not (out / "2").exists() and not (out / "3").exists()
    assert sorted(k for k in z.root.array_keys()) == ["0", "1"]
    assert z.metadata["extra_attributes"] == {"t_range": [3, 4]}
    assert int(np.asarray(z.data[1]).min()) == 7
    # one level only: every coarser level goes away
    z.build_pyramid_inplace(num_levels=1, progress=False)
    assert len(z.data) == 1 and sorted(z.root.array_keys()) == ["0"]


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_aligned_region_writes(tmp_path, zarr_format, ngff):
    out = tmp_path / "img.ome.zarr"
    shards = (8, 64, 64) if zarr_format == 3 else None
    allocate_zarr(out, size=(16, 128, 128), axes="zyx", zarr_format=zarr_format,
                  chunk_size=(8, 32, 32), shards=shards)
    z = ZarrManager(str(out), mode="r+")
    block = (8, 64, 64) if zarr_format == 3 else (8, 32, 32)

    z.write_image_region(np.ones(block, dtype=np.uint16), z=slice(0, block[0]), y=slice(0, block[1]),
                         x=slice(block[2], 2 * block[2]), aligned=True)
    with pytest.raises(ValueError, match="not aligned"):
        z.write_image_region(np.ones((8, 16, 16), dtype=np.uint16), z=slice(0, 8), y=slice(8, 24),
                             x=slice(0, 16), aligned=True)
    # without aligned=True the same write goes through
    z.write_image_region(np.ones((8, 16, 16), dtype=np.uint16), z=slice(0, 8), y=slice(8, 24), x=slice(0, 16))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_cli_allocate_and_pyramid(tmp_path, zarr_format, ngff, capsys):
    template = tmp_path / "raw.ome.zarr"
    _make_template(template, zarr_format, ngff)
    out = tmp_path / "seg.ome.zarr"
    main([
        "allocate", "-t", str(template), "-o", str(out), "--data_type", "label", "--dtype", "uint32",
        "--crop", "t=1:2;y=32:64", "--attrs", json.dumps({"t_range": [1, 2]}),
    ])
    z = ZarrManager(str(out))
    assert z.root.metadata.zarr_format == zarr_format
    assert z.metadata["size"] == [(1, 8, 32, 64)]
    assert z.metadata["translations"] == [_zyx(10.0, 20.0 + 32 * 0.5, 30.0)]
    assert z.metadata["extra_attributes"] == {"t_range": [1, 2]}

    main(["pyramid", "-z", str(out), "--num_levels", "3", "--method", "mode", "--num_workers", "2"])
    z = ZarrManager(str(out))
    assert len(z.data) == 3
    assert z.metadata["data_type"] == "label"


def test_cli_allocate_explicit_geometry(tmp_path):
    out = tmp_path / "new.ome.zarr"
    main(["allocate", "-o", str(out), "--axes", "czyx", "--size", "2", "10", "100", "120",
          "--scale", "2", "0.5", "0.5", "--translation", "1", "2", "3", "--zarr_format", "2",
          "--num_levels", "2"])
    z = ZarrManager(str(out))
    assert z.root.metadata.zarr_format == 2
    assert z.metadata["size"] == [(2, 10, 100, 120), (2, 5, 50, 60)]
    assert z.metadata["scales"] == [(2.0, 0.5, 0.5), (4.0, 1.0, 1.0)]
    assert z.metadata["translations"][0] == _zyx(1.0, 2.0, 3.0)
    assert z.metadata["units"] == ("micrometer",) * 3


def test_cli_allocate_refuses_template_and_size(tmp_path):
    template = tmp_path / "raw.ome.zarr"
    _make_template(template, 3, "0.5")
    with pytest.raises(SystemExit):
        main(["allocate", "-t", str(template), "-o", str(tmp_path / "x.zarr"), "--axes", "zyx",
              "--size", "1", "2", "3"])


# ---------------------------------------------------------------------------
# NGFF 0.6 reading, sharded pyramids, multiscales type/metadata
# ---------------------------------------------------------------------------

def _make_v06_store(path: Path) -> np.ndarray:
    """A hand-written NGFF 0.6 zyx image: scale + translation wrapped in a sequence."""
    data = np.arange(4 * 16 * 16, dtype=np.uint16).reshape(4, 16, 16)
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    arr = root.create_array("0", shape=data.shape, dtype=data.dtype, chunks=(2, 8, 8),
                            dimension_names=("z", "y", "x"))
    arr[:] = data
    axes = [{"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"]
    root.attrs["ome"] = {
        "version": "0.6",
        "multiscales": [{
            "name": "v06",
            "coordinateSystems": [{"name": "physical", "axes": axes}],
            "datasets": [{
                "path": "0",
                "coordinateTransformations": [{
                    "type": "sequence",
                    "input": {"path": "0"},
                    "output": {"name": "physical"},
                    "transformations": [
                        {"type": "scale", "scale": [2.0, 0.5, 0.5]},
                        {"type": "translation", "translation": [1.0, 2.0, 3.0]},
                    ],
                }],
            }],
        }],
    }
    return data


def test_ngff_06_store_is_read_with_its_axes_scale_and_translation(tmp_path):
    src = tmp_path / "v06.ome.zarr"
    _make_v06_store(src)
    z = ZarrManager(str(src))
    assert z.metadata["axes"] == "zyx"
    assert z.metadata["ngff_version"] == "0.6"
    assert z.metadata["zarr_format"] == 3
    assert z.metadata["scales"] == [(2.0, 0.5, 0.5)]
    assert z.metadata["translations"] == [_zyx(1.0, 2.0, 3.0)]
    assert z.metadata["units"] == ("micrometer",) * 3


def test_ngff_06_store_is_not_modified_in_place(tmp_path):
    src = tmp_path / "v06.ome.zarr"
    _make_v06_store(src)
    z = ZarrManager(str(src), mode="r+")
    with pytest.raises(ValueError, match="NGFF 0.6"):
        z.build_pyramid_inplace(num_levels=2, progress=False)
    with pytest.raises(ValueError, match="NGFF 0.6"):
        z.create_empty_group("seg", dict(z.metadata), data_type="label")
    assert sorted(z.root.array_keys()) == ["0"]


def test_ngff_06_template_allocates_a_05_output(tmp_path):
    src = tmp_path / "v06.ome.zarr"
    _make_v06_store(src)
    root = allocate_zarr(tmp_path / "out.ome.zarr", template=src)
    assert root.attrs["ome"]["version"] == "0.5"
    assert ZarrManager(str(tmp_path / "out.ome.zarr")).metadata["translations"] == [_zyx(1.0, 2.0, 3.0)]


def test_sharded_levels_are_built_chunk_by_chunk(tmp_path):
    out = tmp_path / "img.ome.zarr"
    allocate_zarr(out, size=(16, 128, 128), axes="zyx", zarr_format=3,
                  chunk_size=(4, 16, 16), shards=(8, 64, 64))
    data = np.random.default_rng(3).integers(0, 65535, size=(16, 128, 128), dtype=np.uint16)
    z = ZarrManager(str(out), mode="r+")
    z.write_image_region(data)
    z.build_pyramid_inplace(num_levels=3, progress=False)
    assert z.zarr_data[1].shards is not None and z.zarr_data[1].shards != z.zarr_data[1].chunks

    # Independent float64 reference: 2x2x2 block means, rounded.
    ref1 = np.rint(data.reshape(8, 2, 64, 2, 64, 2).astype(np.float64).mean(axis=(1, 3, 5)))
    np.testing.assert_array_equal(np.asarray(z.data[1]), ref1.astype(np.uint16))
    ref2 = np.rint(ref1.astype(np.uint16).reshape(4, 2, 32, 2, 32, 2).astype(np.float64).mean(axis=(1, 3, 5)))
    np.testing.assert_array_equal(np.asarray(z.data[2]), ref2.astype(np.uint16))


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_pyramid_records_its_method_and_stays_valid(tmp_path, zarr_format, ngff):
    from tests.test_ngff_validation import assert_valid_ome_zarr

    template = tmp_path / "raw.ome.zarr"
    _make_template(template, zarr_format, ngff)
    z = ZarrManager(str(template), mode="r+")
    md = dict(z.metadata, axes="tzyx", size=[(2, 8, 64, 64)], chunksize=[(1, 4, 32, 32)], dtype="uint16")
    z.create_empty_group("nuclei", md, data_type="label")
    z.build_pyramid_inplace(num_levels=3, progress=False)
    z.build_pyramid_inplace(group="labels/nuclei", num_levels=3, method="mode", progress=False)

    ms = z._get_multiscales(z.root)[0]
    assert ms["type"] == "mean"
    assert ms["metadata"]["kwargs"] == {"method": "mean", "factors": [2, 2, 2]}
    assert z._get_multiscales(z.root["labels/nuclei"])[0]["type"] == "mode"
    assert z.labels["nuclei"].metadata["data_type"] == "label"
    assert_valid_ome_zarr(template, ngff)

    z.build_pyramid_inplace(num_levels=1, progress=False)
    ms = z._get_multiscales(z.root)[0]
    assert "type" not in ms and "metadata" not in ms


def test_legacy_label_type_marker_is_kept(tmp_path):
    out = tmp_path / "seg.ome.zarr"
    allocate_zarr(out, size=(8, 32, 32), axes="zyx", zarr_format=3, data_type="label", dtype="uint32")
    root = zarr.open_group(str(out), mode="r+")
    ome = root.attrs["ome"]
    ome.pop("image-label")
    ome["multiscales"][0]["type"] = "label"  # how older stores marked labels
    root.attrs["ome"] = ome
    z = ZarrManager(str(out), mode="r+")
    z.build_pyramid_inplace(num_levels=2, progress=False)
    assert z._get_multiscales(z.root)[0]["type"] == "label"
    assert z.metadata["data_type"] == "label"


def test_mean_of_16_bit_data_matches_float64():
    arr = np.random.default_rng(4).integers(0, 65535, size=(32, 32, 32), dtype=np.uint16)
    got = downsample_numpy(arr, (2, 2, 2), (0, 1, 2), "mean")
    ref = np.rint(arr.reshape(16, 2, 16, 2, 16, 2).astype(np.float64).mean(axis=(1, 3, 5)))
    np.testing.assert_array_equal(got, ref.astype(np.uint16))


# ---------------------------------------------------------------------------
# One downsampling method per store: lazy pyramids, allocate and region writes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
@pytest.mark.parametrize("data_type,expected", [("intensity", "mean"), ("label", "nearest")])
def test_lazy_pyramid_defaults_by_data_type_and_records_it(tmp_path, zarr_format, ngff, data_type, expected):
    from tests.test_ngff_validation import assert_valid_ome_zarr
    from zarrsmith.microscope_manager import ArrayManager

    data = np.random.default_rng(1).integers(0, 50, size=(8, 32, 32)).astype(np.uint16)
    metadata = {"axes": "zyx", "scales": [(2.0, 0.5, 0.5)], "units": ("micrometer",) * 3,
                "translations": [_zyx(10.0, 20.0, 30.0)], "data_type": data_type, "dtype": "uint16"}
    manager = ArrayManager([da.from_array(data, chunks=(4, 16, 16))], metadata)
    manager.build_pyramid(num_levels=2)
    assert manager.metadata["downsampling_method"] == expected
    np.testing.assert_array_equal(np.asarray(manager.data[1]), downsample_numpy(data, (2, 2, 2), (0, 1, 2), expected))

    out = tmp_path / "img.ome.zarr"
    manager.to_zarr(str(out), zarr_format=zarr_format, ngff_version=ngff, progress=False)
    z = ZarrManager(str(out))
    ms = z._get_multiscales(z.root)[0]
    assert ms["type"] == expected
    assert ms["metadata"]["kwargs"] == {"method": expected, "factors": [2, 2, 2]}
    assert z.metadata["downsampling_method"] == expected
    assert_valid_ome_zarr(out, ngff)


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_region_writes_refresh_levels_with_the_allocated_method(tmp_path, zarr_format, ngff):
    out = tmp_path / "prob.ome.zarr"
    allocate_zarr(out, size=(8, 32, 32), axes="zyx", zarr_format=zarr_format, dtype="float32",
                  num_levels=2, translation=(10.0, 20.0, 30.0))
    z = ZarrManager(str(out), mode="r+")
    ms = z._get_multiscales(z.root)[0]
    assert ms["type"] == "mean"  # what allocate placed the level translations for

    values = np.random.default_rng(2).random((8, 32, 32)).astype(np.float32)
    z.write_image_region(values)
    level1 = np.asarray(zarr.open_group(str(out), mode="r")["1"][:])
    np.testing.assert_allclose(level1, downsample_numpy(values, (2, 2, 2), (0, 1, 2), "mean"), rtol=1e-6)


def test_region_writes_keep_nearest_for_stores_without_a_method(tmp_path):
    out = tmp_path / "old.ome.zarr"
    allocate_zarr(out, size=(8, 32, 32), axes="zyx", zarr_format=3, dtype="float32", num_levels=2)
    root = zarr.open_group(str(out), mode="r+")
    ome = root.attrs["ome"]
    ome["multiscales"][0].pop("type")
    ome["multiscales"][0].pop("metadata")
    root.attrs["ome"] = ome

    z = ZarrManager(str(out), mode="r+")
    assert z.metadata["downsampling_method"] is None
    values = np.random.default_rng(3).random((8, 32, 32)).astype(np.float32)
    z.write_image_region(values)
    level1 = np.asarray(zarr.open_group(str(out), mode="r")["1"][:])
    np.testing.assert_array_equal(level1, values[::2, ::2, ::2])


@pytest.mark.parametrize("data_type,dtype,level1", [
    ("intensity", "float32", [0.0, 1.0, 0.25, 0.25]),  # averaged: half a level-0 pixel further
    ("label", "uint32", None),                          # nearest: no offset, nothing written
])
def test_allocate_without_a_position_places_averaged_levels(tmp_path, data_type, dtype, level1):
    out = tmp_path / "out.ome.zarr"
    allocate_zarr(out, size=(2, 8, 32, 32), axes="tzyx", scale=(2.0, 0.5, 0.5), zarr_format=3,
                  dtype=dtype, data_type=data_type, num_levels=2)
    ms = ZarrManager(str(out))._get_multiscales(zarr.open_group(str(out), mode="r"))[0]
    transforms = [ds["coordinateTransformations"] for ds in ms["datasets"]]
    if level1 is None:
        assert all(len(t) == 1 for t in transforms)
    else:
        assert transforms[0][1]["translation"] == [0.0, 0.0, 0.0, 0.0]
        assert transforms[1][1]["translation"] == pytest.approx(level1)
