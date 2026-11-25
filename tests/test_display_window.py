"""Display windows (OMERO window start/end): written from the data, read back, updated, shown."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import zarr

import zarrsmith.microscope_manager as mm
from zarrsmith.utils.display_window import (
    _central_box,
    read_display_windows,
    sample_display_windows,
    window_from_values,
)

FORMATS = [("0.4", 2), ("0.5", 3)]


def _image(dtype=np.uint16):
    """Two channels: values 100..1099 and 2000..2999 in the middle, zeros (blank) around."""
    data = np.zeros((2, 4, 32, 32), dtype)
    rng = np.random.default_rng(0)
    data[0, :, 8:24, 8:24] = rng.integers(100, 1100, size=(4, 16, 16))
    data[1, :, 8:24, 8:24] = rng.integers(2000, 3000, size=(4, 16, 16))
    return mm.ArrayManager(data, {"axes": "czyx", "scales": [(1.0, 1.0, 1.0)], "channel_names": ["a", "b"]})


def _write(tmp_path, image=None, **kwargs):
    path = tmp_path / "image.ome.zarr"
    (image or _image()).to_zarr(str(path), progress=False, **kwargs)
    return path


def _windows(path):
    return read_display_windows(zarr.open_group(str(path), mode="r"))


def _open(path, mode="r"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mm.ZarrManager(str(path), mode=mode)


def test_window_from_values_ignores_blank_voxels():
    assert window_from_values(np.r_[np.zeros(1000), np.arange(1, 101)], (0, 100)) == (1.0, 100.0)
    assert window_from_values(np.zeros(10)) == (0.0, 1.0)
    assert window_from_values(np.array([np.nan, 5.0, 5.0])) == (5.0, 6.0)


def test_central_box_bounds_the_voxels_read():
    box = _central_box({"t": 10, "c": 2, "z": 100, "y": 1000, "x": 1000}, 1 << 20, 3)
    size = np.prod([s.stop - s.start for s in box.values()])
    assert 3 * size <= 1 << 20 and box["y"].start == (1000 - (box["y"].stop - box["y"].start)) // 2
    level = np.zeros((10, 2, 100, 1000, 1000), np.uint8)[:, :, :, :, :]  # a view: nothing allocated
    assert len(sample_display_windows(level, "tczyx", max_voxels=1 << 16)) == 2


@pytest.mark.parametrize("ngff_version,zarr_format", FORMATS)
def test_write_stores_the_percentiles_of_the_data(tmp_path, ngff_version, zarr_format):
    path = _write(tmp_path, ngff_version=ngff_version, zarr_format=zarr_format)
    (a0, a1), (b0, b1) = _windows(path)
    assert 100 <= a0 < 110 and 1090 < a1 <= 1099
    assert 2000 <= b0 < 2010 and 2990 < b1 <= 2999
    root = zarr.open_group(str(path), mode="r")
    omero = (root.attrs["ome"] if zarr_format == 3 else root.attrs.asdict())["omero"]
    assert omero["channels"][0]["window"]["max"] == 65535  # min/max: the dtype range, kept
    assert _open(path).metadata["channel_windows"] == [(a0, a1), (b0, b1)]


def test_display_window_off_keeps_the_dtype_range(tmp_path):
    path = _write(tmp_path, display_window=False)
    assert _windows(path) == [(0.0, 65535.0)] * 2
    assert _open(path).metadata["channel_windows"] is None  # the default: nothing chosen


def test_given_windows_are_written_and_kept_on_a_copy(tmp_path):
    image = _image()
    image.update_metadata({"channel_windows": [(10, 500), None]})
    path = _write(tmp_path, image)
    (a, b) = _windows(path)
    assert a == (10.0, 500.0) and 2000 <= b[0] < 2010  # None: from the data

    copy = tmp_path / "copy.ome.zarr"
    z = _open(path)
    z.subset_dataset(C=[1, 0])  # the windows follow their channel
    assert z.metadata["channel_windows"][1] == (10.0, 500.0)
    z.to_zarr(str(copy), progress=False)
    assert _windows(copy)[1] == (10.0, 500.0)


def test_reorder_keeps_each_window_with_its_channel(tmp_path):
    z = _open(_write(tmp_path))
    before = list(z.metadata["channel_windows"])
    z.reorder_channels([1, 0])
    assert z.metadata["channel_windows"] == before[::-1]


@pytest.mark.parametrize("ngff_version,zarr_format", FORMATS)
def test_update_metadata_in_place_sets_and_recomputes_windows(tmp_path, ngff_version, zarr_format):
    path = _write(tmp_path, ngff_version=ngff_version, zarr_format=zarr_format)
    computed = _windows(path)
    z = _open(path, "r+")
    z.create_empty_group("nuclei", dict(z.metadata, dtype="uint16", data_type="label"), is_label=True)
    z.update_metadata({"channel_windows": [(1, 2), (3, 400)]}, in_place=True)  # labels are skipped
    assert _windows(path) == [(1.0, 2.0), (3.0, 400.0)]
    assert _open(path).metadata["channel_windows"] == [(1.0, 2.0), (3.0, 400.0)]
    assert "omero" not in zarr.open_group(str(path), mode="r")["labels/nuclei"].attrs.get("ome", {})

    z.update_metadata({"channel_windows": [None, (5, 6)]}, in_place=True)  # None: from the data
    assert _windows(path) == [computed[0], (5.0, 6.0)]
    z.update_metadata({"channel_windows": None}, in_place=True)
    assert _windows(path) == computed
    assert z.metadata["channel_windows"] == computed


def test_invalid_windows_are_refused():
    image = _image()
    with pytest.raises(ValueError, match="end > start"):
        image.update_metadata({"channel_windows": [(5, 5), None]})
    with pytest.warns(UserWarning, match="one \\(start, end\\) or None per channel"):
        image.update_metadata({"channel_windows": [(1, 2)]})
    assert image.metadata.get("channel_windows") is None


def test_visualize_opens_at_the_window_with_the_whole_slider(tmp_path):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    viewer = ViewerModel()
    path = _write(tmp_path)
    z = _open(path)
    z.visualize(viewer=viewer)
    windows = z.metadata["channel_windows"]
    for layer, window in zip(viewer.layers, windows):
        assert tuple(layer.contrast_limits) == pytest.approx(window)
        assert tuple(layer.contrast_limits_range) == (0, 65535)
