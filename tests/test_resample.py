from __future__ import annotations

import math

import dask.array as da
import numpy as np
import pytest
import zarr
from scipy.ndimage import gaussian_filter1d

import zarrsmith.microscope_manager as mm
from zarrsmith.utils import ngff
from zarrsmith.utils.resample import (
    _batches,
    _fitted_array_kwargs,
    resample_array,
    resample_levels,
    resample_region,
)


def _smooth_image(shape=(24, 96, 96), dtype=np.uint16):
    rng = np.random.default_rng(0)
    image = rng.random(shape)
    for axis in range(image.ndim):
        image = gaussian_filter1d(image, 2, axis=axis)
    image = (image - image.min()) / np.ptp(image)
    return (image * 60000).astype(dtype)


def _metadata(scales, axes="zyx", **extra):
    return {"axes": axes, "scales": scales, **extra}


@pytest.mark.parametrize(
    "factors",
    [
        {0: 0.25},  # upsample a lot more than the chunk size allows
        {1: 2.0, 2: 2.0},  # integer downsampling
        {0: 0.7, 1: 1.3, 2: 3.7},  # mixed, non-integer
        {2: 13.0},  # a factor larger than the chunks
    ],
)
def test_result_does_not_depend_on_chunks(factors):
    image = _smooth_image()
    reference = resample_array(da.from_array(image, chunks=-1), factors).compute()

    for chunks in [(4, 16, 16), (1, 7, 31), (24, 96, 5)]:
        result = resample_array(da.from_array(image, chunks=chunks), factors)
        computed = result.compute()
        assert result.shape == reference.shape
        assert sum(result.chunks[0]) == reference.shape[0]
        # Only float rounding may differ: at most one grey level out of 60000.
        assert np.abs(computed.astype(int) - reference.astype(int)).max() <= 1


def test_blocks_match_their_announced_chunks():
    image = _smooth_image()
    result = resample_array(da.from_array(image, chunks=(1, 7, 31)), {0: 0.3, 1: 4.1, 2: 1.7})
    # compute_chunk_sizes would hide a mismatch: check block by block.
    for index in np.ndindex(result.numblocks):
        expected = tuple(chunks[i] for chunks, i in zip(result.chunks, index))
        assert result.blocks[index].compute().shape == expected


@pytest.mark.parametrize("n, factor", [(1, 0.5), (1, 3.0), (2, 5.0), (3, 3.0), (5, 100.0), (7, 0.1)])
def test_tiny_axes(n, factor):
    arr = da.from_array(np.arange(n, dtype=np.uint16) * 10, chunks=1)
    result = resample_array(arr, {0: factor}).compute()
    assert result.shape == (int(np.ceil(n / factor)),)


def test_linear_ramp_is_reproduced_exactly():
    ramp = np.arange(32, dtype=np.float32)
    result = resample_array(da.from_array(ramp, chunks=5), {0: 0.25}).compute()
    assert result.shape == (128,)
    expected = np.minimum(np.arange(128) * 0.25, 31)
    np.testing.assert_allclose(result, expected)


def test_downsampling_is_anti_aliased():
    stripes = np.tile([0.0, 100.0], 64).astype(np.float32)
    result = resample_array(da.from_array(stripes, chunks=16), {0: 4.0}).compute()
    assert np.abs(result[4:-4] - 50).max() < 15  # plain subsampling would give all 0


def test_integer_dtype_is_preserved_and_flat_image_stays_flat():
    image = np.full((16, 32, 32), 1234, dtype=np.uint16)
    result = resample_array(da.from_array(image, chunks=8), {0: 0.7, 1: 1.3, 2: 2.5})
    assert result.dtype == np.uint16
    computed = result.compute()
    assert computed.dtype == np.uint16
    assert (computed == 1234).all()


def test_large_integers_keep_their_precision():
    value = 2**31 + 1  # not representable in float32
    image = np.full((8, 8), value, dtype=np.uint32)
    assert (resample_array(da.from_array(image, chunks=4), {0: 1.5}).compute() == value).all()


def test_labels_only_get_existing_values():
    labels = np.zeros((16, 64, 64), dtype=np.uint32)
    labels[2:10, 8:30, 8:30] = 7
    labels[4:14, 30:60, 30:60] = 100_000

    data, _ = resample_levels(
        [da.from_array(labels, chunks=(4, 16, 16))],
        _metadata([(1.0, 1.0, 1.0)], is_label=True),
        (0.5, 1.6, 2.7),
    )

    computed = data[0].compute()
    assert data[0].dtype == np.uint32
    assert set(np.unique(computed)) == {0, 7, 100_000}


def test_metadata_and_pyramid_are_rebuilt():
    image = np.zeros((16, 64, 64), dtype=np.uint16)
    levels = [
        da.from_array(image, chunks=(4, 16, 16)),
        da.from_array(image[:, ::2, ::2], chunks=(4, 16, 16)),
        da.from_array(image[:, ::4, ::4], chunks=(4, 16, 16)),
    ]
    metadata = _metadata(
        [(2.0, 0.5, 0.5), (2.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        size=[level.shape for level in levels],
        chunksize=[level.chunksize for level in levels],
    )

    data, new = resample_levels(levels, metadata, {"z": 1.0, "y": 0.25, "x": 0.25})

    assert [level.shape for level in data] == [(32, 128, 128), (32, 64, 64), (32, 32, 32)]
    assert new["scales"] == [(1.0, 0.25, 0.25), (1.0, 0.5, 0.5), (1.0, 1.0, 1.0)]
    assert new["size"] == [level.shape for level in data]
    assert all(level.chunksize == (4, 16, 16) for level in data)
    assert metadata["scales"][0] == (2.0, 0.5, 0.5)  # the input is left alone


def test_coarsest_sufficient_level_is_used():
    def must_not_read(block):
        raise AssertionError("the finest level should not be read")

    image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    level0 = da.from_array(image, chunks=16).map_blocks(must_not_read, dtype=np.uint16, meta=np.empty((0, 0), np.uint16))
    level1 = da.from_array(image[::2, ::2], chunks=16)

    data, metadata = resample_levels(
        [level0, level1],
        _metadata([(1.0, 1.0), (2.0, 2.0)], axes="yx"),
        2.0,
        fast=True,
    )

    np.testing.assert_array_equal(data[0].compute(), image[::2, ::2])
    assert metadata["scales"][0] == (2.0, 2.0)


def test_coarse_level_is_not_used_if_one_axis_needs_more_detail():
    image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    levels = [da.from_array(image, chunks=16), da.from_array(image[::2, ::2], chunks=16)]

    data, _ = resample_levels(levels, _metadata([(1.0, 1.0), (2.0, 2.0)], axes="yx"), (2.0, 1.0), fast=True)

    assert data[0].shape == (16, 32)
    assert data[0][:, 1].compute().tolist() != data[0][:, 0].compute().tolist()


@pytest.mark.parametrize(
    "target, expected",
    [
        (1.0, (1.0, 1.0, 1.0)),
        ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
        ({"z": 4.0}, (4.0, 0.5, 0.5)),
    ],
)
def test_target_scale_forms(target, expected):
    data = [da.zeros((8, 8, 8), chunks=4, dtype=np.uint8)]
    _, metadata = resample_levels(data, _metadata([(2.0, 0.5, 0.5)]), target)
    assert metadata["scales"][0] == expected


def test_dict_ignores_spatial_axes_the_dataset_does_not_have():
    data = [da.zeros((8, 8), chunks=4, dtype=np.uint8)]
    _, metadata = resample_levels(data, _metadata([(1.0, 1.0)], axes="yx"), {"z": 2.0, "x": 2.0})
    assert metadata["scales"][0] == (1.0, 2.0)


@pytest.mark.parametrize("target", [0.0, -1.0, (1.0, -1.0, 1.0), {"z": 0.0}, {"t": 2.0}])
def test_invalid_target_scale_is_refused(target):
    data = [da.zeros((8, 8, 8), chunks=4, dtype=np.uint8)]
    with pytest.raises(ValueError):
        resample_levels(data, _metadata([(1.0, 1.0, 1.0)]), target)


def test_same_scale_keeps_the_data():
    image = _smooth_image((8, 16, 16))
    data, _ = resample_levels([da.from_array(image, chunks=4)], _metadata([(1.0, 1.0, 1.0)]), 1.0)
    np.testing.assert_array_equal(data[0].compute(), image)


def test_non_spatial_axes_are_untouched():
    image = _smooth_image((2, 3, 8, 32, 32))
    data, metadata = resample_levels(
        [da.from_array(image, chunks=(1, 1, 4, 16, 16))],
        _metadata([(1.0, 1.0, 1.0)], axes="tczyx"),
        (0.5, 2.0, 2.0),
    )
    assert data[0].shape == (2, 3, 16, 16, 16)
    assert metadata["axes"] == "tczyx"
    separate = resample_array(da.from_array(image[1, 2], chunks=(4, 16, 16)), {0: 0.5, 1: 2.0, 2: 2.0})
    np.testing.assert_array_equal(data[0][1, 2].compute(), separate.compute())


def test_array_manager_end_to_end(tmp_path):
    image = _smooth_image((16, 64, 64))
    metadata = {
        "size": [image.shape],
        "chunksize": [(8, 32, 32)],
        "scales": [(2.0, 0.5, 0.5)],
        "units": ("micrometer", "micrometer", "micrometer"),
        "time_increment": None,
        "time_increment_unit": None,
        "channel_names": [],
        "channel_colors": [],
        "dtype": "uint16",
        "axes": "zyx",
        "data_type": "intensity",
        "name": "image",
    }
    manager = mm.ArrayManager([da.from_array(image, chunks=(8, 32, 32))], metadata)
    manager.resample_dataset({"z": 1.0})

    path = tmp_path / "resampled.zarr"
    manager.to_zarr(path)

    reread = mm.ZarrManager(path)
    assert reread.data[0].shape == (32, 64, 64)
    assert tuple(reread.metadata["scales"][0]) == (1.0, 0.5, 0.5)
    assert reread.data[0].chunksize == (8, 32, 32)
    np.testing.assert_array_equal(reread.data[0].compute(), manager.data[0].compute())


def test_zarr_manager_resamples_image_and_labels(tmp_path, image_pyramid, metadata):
    source = tmp_path / "source.zarr"
    mm.ArrayManager(image_pyramid, metadata).to_zarr(str(source), drop_singleton=False)

    z = mm.ZarrManager(str(source), mode="a")
    z.create_empty_group("nuclei", metadata, is_label=True)
    label_group = z.root["labels"]["nuclei"]
    label_group["0"][...] = 9

    z = mm.ZarrManager(str(source), mode="r")
    z.resample_dataset({"z": 1.0, "y": 0.25, "x": 0.25})

    assert z.data is z.raw.data
    assert z.raw.zarr_data is None
    assert z.data[0].shape == (2, 2, 8, 32, 32)
    assert z.labels["nuclei"].metadata["scales"][0] == (1.0, 0.25, 0.25)
    assert z.labels["nuclei"].data[0].shape[-3:] == (8, 32, 32)

    output = tmp_path / "resampled.zarr"
    z.to_zarr(str(output))

    reread = mm.ZarrManager(str(output))
    assert reread.data[0].shape == (2, 2, 8, 32, 32)
    assert tuple(reread.labels["nuclei"].metadata["scales"][0]) == (1.0, 0.25, 0.25)
    assert (reread.labels["nuclei"].data[0].compute() == 9).all()


# --- level 0 by default, leftover chunks --------------------------------------

def test_images_read_level_0_unless_fast():
    image = _smooth_image((32, 32))
    levels = [da.from_array(image, chunks=16), da.from_array(image[::2, ::2], chunks=16)]
    metadata = _metadata([(1.0, 1.0), (2.0, 2.0)], axes="yx")

    careful, _ = resample_levels(levels, metadata, 2.0)
    fast, _ = resample_levels(levels, metadata, 2.0, fast=True)

    np.testing.assert_array_equal(fast[0].compute(), image[::2, ::2])
    assert not np.array_equal(careful[0].compute(), image[::2, ::2])


def test_labels_read_the_coarsest_level_without_fast():
    def must_not_read(block):
        raise AssertionError("the finest level should not be read")

    labels = np.arange(32 * 32, dtype=np.uint32).reshape(32, 32)
    level0 = da.from_array(labels, chunks=16).map_blocks(must_not_read, dtype=np.uint32, meta=np.empty((0, 0), np.uint32))
    level1 = da.from_array(labels[::2, ::2], chunks=16)

    data, _ = resample_levels([level0, level1], _metadata([(1.0, 1.0), (2.0, 2.0)], axes="yx", is_label=True), 2.0)

    np.testing.assert_array_equal(data[0].compute(), labels[::2, ::2])


def test_a_large_leftover_keeps_the_regular_chunks():
    arr = da.zeros((2048,), chunks=512, dtype=np.uint16)
    result = resample_array(arr, {0: 1.7})
    assert result.shape == (1205,)
    assert result.chunks == ((512, 512, 181),)  # it used to end with a chunk of 693


def test_a_small_leftover_is_merged_when_it_cannot_hold_the_overlap():
    arr = da.from_array(np.arange(40, dtype=np.float32), chunks=8)
    result = resample_array(arr, {0: 4.0}, chunks=(4,))
    assert result.shape == (10,)
    assert sum(result.chunks[0]) == 10
    assert result.compute().shape == (10,)


# --- regions and batches ------------------------------------------------------

@pytest.mark.parametrize("nearest", [False, True])
@pytest.mark.parametrize("factors", [{0: 0.7, 1: 1.3, 2: 3.7}, {1: 2.0, 2: 2.0}, {0: 0.25}])
def test_regions_reassemble_the_full_result(factors, nearest):
    image = _smooth_image()
    source = da.from_array(image, chunks=(5, 16, 16))
    expected = resample_array(source, factors, nearest=nearest).compute()

    z_chunk = 8
    pieces = []
    for start in range(0, expected.shape[0], z_chunk):
        region = (
            slice(start, min(start + z_chunk, expected.shape[0])),
            slice(0, expected.shape[1]),
            slice(0, expected.shape[2]),
        )
        pieces.append(resample_region(source, factors, region, (z_chunk, 16, 16), nearest=nearest).compute())
    assert np.abs(np.concatenate(pieces).astype(int) - expected.astype(int)).max() <= 1


@pytest.mark.parametrize("max_pixels", [1, 4 * 16 * 40 * 2, 25 * 40 * 40, 3 * 2 * 25 * 40 * 40])
def test_batches_cover_every_pixel_once_in_whole_blocks(max_pixels):
    shape, block = (3, 2, 25, 40, 40), (1, 1, 4, 16, 16)
    seen = np.zeros(shape, dtype=np.uint8)
    for region in _batches(shape, block, max_pixels):
        seen[region] += 1
        for piece, b, n in zip(region, block, shape):
            assert piece.start % b == 0
            assert piece.stop % b == 0 or piece.stop == n
    assert (seen == 1).all()


def test_batches_stay_as_large_as_memory_allows():
    # Plenty of memory: everything runs in parallel, in one batch.
    assert len(list(_batches((81, 3, 20, 100, 100), (1, 1, 12, 100, 100), 1e12))) == 1

    # Room for ten volumes: timepoints are grouped, and volumes are not split.
    volume = 20 * 100 * 100
    regions = list(_batches((81, 3, 20, 100, 100), (1, 1, 12, 100, 100), 10 * volume))
    assert regions[0][0] == slice(0, 3)
    assert regions[0][1] == slice(0, 3)
    assert regions[0][2] == slice(0, 20)
    assert len(regions) == 27


def test_batches_never_go_below_one_block():
    regions = list(_batches((8, 32, 32), (4, 16, 16), max_pixels=1))
    assert len(regions) == 2 * 2 * 2
    assert all(r[0].stop - r[0].start == 4 for r in regions)


# --- write_resampled: the store on disk ---------------------------------------

def _store(path, shape=(2, 2, 16, 64, 64), chunks=(1, 1, 8, 32, 32), num_levels=3, **write):
    image = _smooth_image(shape)
    metadata = {
        "size": [shape],
        "chunksize": [chunks],
        "scales": [(2.0, 0.5, 0.5)],
        "units": ("micrometer", "micrometer", "micrometer"),
        "time_increment": 1.0,
        "time_increment_unit": "second",
        "channel_names": [f"c{c}" for c in range(shape[1])],
        "channel_colors": ["FF0000", "00FF00"][: shape[1]],
        "dtype": "uint16",
        "axes": "tczyx",
        "data_type": "intensity",
        "name": "image",
    }
    manager = mm.ArrayManager([da.from_array(image, chunks=chunks)], metadata, chunks=chunks)
    manager.build_pyramid(num_levels=num_levels, downscale_factor=(1, 2, 2))
    manager.to_zarr(str(path), drop_singleton=False, **write)
    return image


def test_store_matches_the_lazy_result_and_keeps_the_codecs(tmp_path):
    image = _store(tmp_path / "in.zarr", shards=None)
    z = mm.ZarrManager(str(tmp_path / "in.zarr"))

    out = z.resample_to_zarr(tmp_path / "out.zarr", {"z": 1.0, "y": 0.7, "x": 0.7}, progress=False)

    expected = resample_array(da.from_array(image), {2: 0.5, 3: 0.7 / 0.5, 4: 0.7 / 0.5}).compute()
    computed = out.data[0].compute()
    assert computed.shape == expected.shape
    assert np.abs(computed.astype(int) - expected.astype(int)).max() <= 1

    for got, want in zip(out.metadata["scales"], [(1.0, 0.7, 0.7), (1.0, 1.4, 1.4), (1.0, 2.8, 2.8)]):
        assert tuple(got) == pytest.approx(want)
    assert out.metadata["channel_names"] == ["c0", "c1"]
    for src, dst in zip(z.zarr_data, out.zarr_data):
        assert src.compressors == dst.compressors
        assert src.dtype == dst.dtype


@pytest.mark.parametrize(
    "shape, chunks, shards, new_shape, new_chunks, new_shards",
    [
        # Levels of the real timelapse whose padded chunks nearly doubled the store.
        ((81, 3, 20, 7428, 6970), (1, 1, 3, 1858, 1743), (1, 1, 12, 7432, 6972),
         (81, 3, 20, 3078, 2888), (1, 1, 3, 1539, 1444), (1, 1, 12, 3078, 2888)),
        ((81, 3, 20, 1857, 1743), (1, 1, 3, 1857, 1743), (1, 1, 21, 1857, 1743),
         (81, 3, 20, 770, 722), (1, 1, 3, 770, 722), (1, 1, 21, 770, 722)),
        # Enlarging: more chunks, none larger than the input's.
        ((20, 100), (3, 40), (12, 80), (40, 250), (3, 36), (12, 72)),
        # No shards.
        ((64, 64), (16, 16), None, (40, 10), (14, 10), None),
    ],
)
def test_chunks_and_shards_fit_the_new_shape(shape, chunks, shards, new_shape, new_chunks, new_shards):
    src = zarr.create_array(store={}, shape=shape, chunks=chunks, shards=shards, dtype="uint16")
    kwargs = _fitted_array_kwargs(src, new_shape)
    assert kwargs["shape"] == new_shape
    assert kwargs["chunks"] == new_chunks
    assert kwargs.get("shards") == new_shards


def test_shrunk_uncompressed_store_holds_almost_no_padding(tmp_path):
    # Chunks larger than a level are stored in full: this store used to be about 6x its pixels.
    _store(tmp_path / "in.zarr", shape=(1, 1, 8, 64, 64), chunks=(1, 1, 8, 50, 50), num_levels=3,
           shards=None, compressor=None)
    z = mm.ZarrManager(str(tmp_path / "in.zarr"))
    assert not z.zarr_data[0].compressors

    out = z.resample_to_zarr(tmp_path / "out.zarr", (1.0, 1.5, 1.5), progress=False)

    pixels = sum(math.prod(level.shape) for level in out.data) * z.zarr_data[0].dtype.itemsize
    stored = sum(f.stat().st_size for f in (tmp_path / "out.zarr").rglob("*") if f.is_file() and f.name != "zarr.json")
    assert stored <= 1.1 * pixels
    for level in out.zarr_data:
        assert all(c <= n for c, n in zip(level.chunks, level.shape))


def test_coarse_levels_skip_pixels_of_the_level_below(tmp_path):
    _store(tmp_path / "in.zarr")
    z = mm.ZarrManager(str(tmp_path / "in.zarr"))
    out = z.resample_to_zarr(tmp_path / "out.zarr", (1.0, 0.7, 0.7), progress=False)

    level0, level1, level2 = (level.compute() for level in out.data)
    np.testing.assert_array_equal(level1, level0[:, :, :, ::2, ::2])
    np.testing.assert_array_equal(level2, level1[:, :, :, ::2, ::2])


def test_batches_do_not_change_the_store(tmp_path, monkeypatch):
    _store(tmp_path / "in.zarr", shards=None)
    z = mm.ZarrManager(str(tmp_path / "in.zarr"))
    whole = z.resample_to_zarr(tmp_path / "whole.zarr", (1.3, 0.7, 0.7), progress=False)

    monkeypatch.setattr(ngff, "_available_memory_mb", lambda: 0.1)  # a batch per block
    small = z.resample_to_zarr(tmp_path / "small.zarr", (1.3, 0.7, 0.7), progress=False)

    for a, b in zip(whole.data, small.data):
        assert np.abs(a.compute().astype(int) - b.compute().astype(int)).max() <= 1


def test_sharded_store_with_small_batches(tmp_path, monkeypatch):
    # With shards along z, a batch is whole shards: a race would lose pixels.
    shape, chunks = (1, 1, 32, 64, 64), (1, 1, 4, 16, 16)
    image = _store(tmp_path / "in.zarr", shape=shape, chunks=chunks, num_levels=2, shards=(1, 1, 8, 32, 32))
    z = mm.ZarrManager(str(tmp_path / "in.zarr"))
    assert z.zarr_data[0].shards == (1, 1, 8, 32, 32)

    monkeypatch.setattr(ngff, "_available_memory_mb", lambda: 0.1)
    out = z.resample_to_zarr(tmp_path / "out.zarr", (1.0, 0.25, 0.25), progress=False)

    assert out.zarr_data[0].shards == (1, 1, 8, 32, 32)
    expected = resample_array(da.from_array(image), {2: 0.5, 3: 0.5, 4: 0.5}).compute()
    assert np.abs(out.data[0].compute().astype(int) - expected.astype(int)).max() <= 1


def test_zarr_v2_store(tmp_path):
    _store(tmp_path / "in.zarr", ngff_version="0.4", zarr_format=2)
    z = mm.ZarrManager(str(tmp_path / "in.zarr"))
    out = z.resample_to_zarr(tmp_path / "out.zarr", (1.0, 1.0, 1.0), progress=False)
    assert out.root.metadata.zarr_format == 2
    assert out.data[0].shape == (2, 2, 32, 32, 32)
    assert tuple(out.metadata["scales"][0]) == (1.0, 1.0, 1.0)


def test_store_labels_and_groups(tmp_path, image_pyramid, metadata):
    source = tmp_path / "source.zarr"
    mm.ArrayManager(image_pyramid, metadata).to_zarr(str(source), drop_singleton=False)
    z = mm.ZarrManager(str(source), mode="a")
    z.create_empty_group("nuclei", metadata, is_label=True)
    z.root["labels"]["nuclei"]["0"][...] = 9
    z.create_empty_group("processed", metadata)

    z = mm.ZarrManager(str(source))
    out = z.resample_to_zarr(tmp_path / "out.zarr", {"z": 1.0, "y": 0.25, "x": 0.25}, progress=False)

    assert out.data[0].shape == (2, 2, 8, 32, 32)
    assert out.groups["processed"].data[0].shape == (2, 2, 8, 32, 32)
    assert out.labels["nuclei"].data[0].shape[-3:] == (8, 32, 32)
    assert (out.labels["nuclei"].data[0].compute() == 9).all()
    assert tuple(out.labels["nuclei"].metadata["scales"][0]) == (1.0, 0.25, 0.25)
    assert out.root["labels"].attrs.asdict() == z.root["labels"].attrs.asdict()


def test_fast_reads_the_pyramid_of_the_store(tmp_path):
    _store(tmp_path / "in.zarr", shape=(1, 1, 8, 64, 64), chunks=(1, 1, 8, 32, 32))
    z = mm.ZarrManager(str(tmp_path / "in.zarr"))
    out = z.resample_to_zarr(tmp_path / "out.zarr", (2.0, 1.0, 1.0), fast=True, progress=False)
    np.testing.assert_array_equal(out.data[0].compute(), z.data[1].compute())


def test_refusals_leave_the_output_alone(tmp_path):
    _store(tmp_path / "in.zarr")
    z = mm.ZarrManager(str(tmp_path / "in.zarr"))
    existing = z.resample_to_zarr(tmp_path / "out.zarr", 1.0, progress=False)
    shape = existing.data[0].shape

    with pytest.raises(FileExistsError):
        z.resample_to_zarr(tmp_path / "out.zarr", 2.0, progress=False)
    with pytest.raises(ValueError):
        z.resample_to_zarr(tmp_path / "out.zarr", -1.0, overwrite=True, progress=False)  # checked first
    assert mm.ZarrManager(str(tmp_path / "out.zarr")).data[0].shape == shape

    with pytest.raises(FileExistsError):
        z.resample_to_zarr(tmp_path / "in.zarr", 1.0, overwrite=True, progress=False)

    z.subset_dataset(Z=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="in-memory changes"):
        z.resample_to_zarr(tmp_path / "new.zarr", 1.0, progress=False)
