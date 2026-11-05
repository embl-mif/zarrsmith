from __future__ import annotations

import json

import h5py
import numpy as np

from zarrsmith.mosaic_manager import H5Array, LuxendoMosaicManager


def test_h5_array_opens_files_without_locking(tmp_path, monkeypatch):
    path = tmp_path / "tile.h5"
    with h5py.File(path, "w") as f:
        f["Data"] = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    opened = []
    real = h5py.File

    def spy(*args, **kwargs):
        opened.append(kwargs.get("locking"))
        return real(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", spy)
    array = H5Array(path)  # the file is opened for its shape and dtype
    assert array.shape == (2, 3, 4) and array.dtype == np.uint16
    assert array[1, 2, 3] == 23  # and again for every read
    assert opened == [False, False]


def _write_tile(folder, name, desc, stack, tx, ty, tz, shape=(4, 8, 6), value=1, time_point=None):
    folder.mkdir(parents=True, exist_ok=True)
    with h5py.File(folder / f"{name}.lux.h5", "w") as f:
        f["Data"] = np.full(shape, value, np.uint16)
    info = {
        "channel": "1",
        "channel_description": desc,
        "stack": stack,
        "stack_description": "brain",
        **({} if time_point is None else {"time_point": str(time_point)}),
        "voxel_size_um": {"width": 2.0, "height": 2.0, "depth": 3.0},
        "image_size_vx": {"width": shape[2], "height": shape[1], "depth": shape[0]},
        "affine_to_sample": [
            {"matrix": np.eye(3).tolist(), "translation": [0, 0, 0]},
            {"matrix": [[2.0, 0, 0], [0, 2.0, 0], [0, 0, -3.0]], "translation": [tx, ty, tz]},
        ],
    }
    (folder / f"{name}.json").write_text(json.dumps({"processingInformation": info}))


def test_luxendo_scan(tmp_path):
    raw = tmp_path / "raw"
    _write_tile(raw / "stack_1-x00-y00_ch_GFP-L", "Cam_bottom_00000", "GFP-L", "1-x00-y00", 0.0, 0.0, 30.0)
    # a re-acquired stack: only the retake (_R_1) is used
    _write_tile(raw / "stack_2-x00-y00_ch_GFP-R", "Cam_bottom_00000", "GFP-R", "2-x00-y00", 100.0, 0.0, 30.0, value=5)
    _write_tile(raw / "stack_2-x00-y00_ch_GFP-R", "Cam_bottom_00000_R_1", "GFP-R", "2-x00-y00", 10.0, 0.0, 30.0, value=7)
    # files that are not tiles
    with h5py.File(raw / "main_raw.lux.h5", "w") as f:
        f["x"] = 1
    _write_tile(tmp_path / "ims" / "elsewhere", "Cam_bottom_00000", "GFP-L", "9-x00-y00", 0.0, 0.0, 0.0)

    m = LuxendoMosaicManager(tmp_path, progress=False)
    assert len(m.tiles) == 2  # ims/ is not scanned
    assert m.channels == ["GFP"] and m.groups == ["L", "R"]

    left, right = m.tiles
    assert right.attrs["path"].endswith("Cam_bottom_00000_R_1.lux.h5")
    assert right.data[0, 0, 0] == 7
    assert left.shape == (4, 8, 6) and left.dtype == np.uint16
    assert left.voxel_size == (3.0, 2.0, 2.0)
    # tile centred on the translation in x/y; z = translation / signed plane step * plane size
    assert left.origin == (-30.0, -7.0, -5.0)
    assert right.origin[2] == 5.0
    assert left.attrs["grid"] == (0, 0)

    zm = m.fuse(tmp_path / "f.zarr", blend="midline", chunks=(1, 4, 8, 8), progress=False, drop_singleton=False)
    assert zm.metadata["channel_names"] == ["GFP"]
    # as for zarrsmith convert, a single channel axis is dropped by default
    assert m.fuse(tmp_path / "g.zarr", progress=False).metadata["axes"] == "zyx"


def test_timepoints_of_the_layouts(tmp_path):
    # (a) numbered acquisitions of the same stack in one folder
    a = tmp_path / "a" / "raw" / "stack_1-x00-y00_GFP-L"
    _write_tile(a, "Cam_bottom_00000", "GFP-L", "1-x00-y00", 0.0, 0.0, 0.0, time_point=0)
    _write_tile(a, "Cam_bottom_00001", "GFP-L", "1-x00-y00", 0.0, 0.0, 0.0, time_point=1)
    assert LuxendoMosaicManager(tmp_path / "a", progress=False).timepoints == [0, 1]

    # (b) one folder per timepoint: the JSON timepoints differ
    for tp in (0, 1):
        _write_tile(tmp_path / "b" / "raw" / f"stack_1-x00-y00_GFP-L_tp{tp}", "Cam_bottom_00000", "GFP-L",
                    "1-x00-y00", 0.0, 0.0, 0.0, time_point=tp)
    assert LuxendoMosaicManager(tmp_path / "b", progress=False).timepoints == [0, 1]

    # (a) without time_point in the JSON: the acquisitions still get distinct timepoints
    a2 = tmp_path / "a2" / "raw" / "stack_1-x00-y00_GFP-L"
    _write_tile(a2, "Cam_bottom_00000", "GFP-L", "1-x00-y00", 0.0, 0.0, 0.0)
    _write_tile(a2, "Cam_bottom_00001", "GFP-L", "1-x00-y00", 0.0, 0.0, 0.0)
    assert LuxendoMosaicManager(tmp_path / "a2", progress=False).timepoints == [0, 1]

    # a retake is still one timepoint
    c = tmp_path / "c" / "raw" / "stack_1-x00-y00_GFP-L"
    _write_tile(c, "Cam_bottom_00000", "GFP-L", "1-x00-y00", 0.0, 0.0, 0.0, time_point=3)
    _write_tile(c, "Cam_bottom_00000_R_1", "GFP-L", "1-x00-y00", 0.0, 0.0, 0.0, time_point=3)
    m = LuxendoMosaicManager(tmp_path / "c", progress=False)
    assert len(m.tiles) == 1 and m.tiles[0].timepoint == 3 and m.timepoints == [3]


def test_opera_reader_refuses_luxendo_data(tmp_path):
    import pytest

    from zarrsmith.mosaic_manager import OperaMosaicManager, WrongReaderError, detect_microscope

    _write_tile(tmp_path / "raw" / "stack_1-x00-y00_ch_GFP-L", "Cam_bottom_00000", "GFP-L", "1-x00-y00", 0.0, 0.0, 30.0)
    assert detect_microscope(tmp_path) == "luxendo"
    assert detect_microscope(tmp_path / "raw" / "stack_1-x00-y00_ch_GFP-L") == "luxendo"
    with pytest.raises(WrongReaderError, match="LuxendoMosaicManager") as e:
        OperaMosaicManager(tmp_path)
    assert e.value.detected == "luxendo"
    assert LuxendoMosaicManager(tmp_path).tiles
