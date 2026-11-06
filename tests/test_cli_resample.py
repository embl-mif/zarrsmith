from __future__ import annotations

import sys

import dask.array as da
import numpy as np
import pandas as pd
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.cli import zarrsmith as cli


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["zarrsmith", *map(str, argv)])
    cli.main()


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "in.ome.zarr"
    metadata = {
        "scales": [(2.0, 0.5, 0.5)],
        "units": ("micrometer",) * 3,
        "time_increment": 1.0,
        "time_increment_unit": "second",
        "axes": "tczyx",
    }
    chunks = (1, 1, 4, 16, 16)
    image = np.arange(2 * 8 * 32 * 32, dtype=np.uint16).reshape(1, 2, 8, 32, 32)
    mm.ArrayManager(da.from_array(image, chunks=chunks), metadata, chunks=chunks).to_zarr(str(path), drop_singleton=False)
    return path


def test_zyx_pixel_size(store, tmp_path, monkeypatch):
    out = tmp_path / "out.ome.zarr"
    _run(monkeypatch, "resample", "-i", store, "-z", out, "-ps", 1, 1, 1)
    z = mm.ZarrManager(str(out))
    assert z.data[0].shape == (1, 2, 16, 16, 16)
    assert tuple(z.metadata["scales"][0]) == (1.0, 1.0, 1.0)


def test_one_pixel_size_for_all_axes(store, tmp_path, monkeypatch):
    out = tmp_path / "out.ome.zarr"
    _run(monkeypatch, "resample", "-i", store, "-z", out, "-ps", 2)
    z = mm.ZarrManager(str(out))
    assert z.data[0].shape == (1, 2, 8, 8, 8)
    assert tuple(z.metadata["scales"][0]) == (2.0, 2.0, 2.0)


def test_existing_output_needs_overwrite(store, tmp_path, monkeypatch):
    out = tmp_path / "out.ome.zarr"
    _run(monkeypatch, "resample", "-i", store, "-z", out, "-ps", 1)
    with pytest.raises(FileExistsError):
        _run(monkeypatch, "resample", "-i", store, "-z", out, "-ps", 2)
    _run(monkeypatch, "resample", "-i", store, "-z", out, "-ps", 2, "--overwrite")
    assert tuple(mm.ZarrManager(str(out)).metadata["scales"][0]) == (2.0, 2.0, 2.0)


def test_batch(store, tmp_path, monkeypatch, capsys):
    taken = tmp_path / "taken.ome.zarr"
    _run(monkeypatch, "resample", "-i", store, "-z", taken, "-ps", 1)

    csv = tmp_path / "images.csv"
    pd.DataFrame(
        [
            {"input": store, "output": tmp_path / "a.ome.zarr", "pixel_size": "1 1 1", "fast": ""},
            {"input": store, "output": tmp_path / "b.ome.zarr", "pixel_size": "2", "fast": "true"},
            {"input": store, "output": taken, "pixel_size": "2", "fast": ""},  # refused, the others still run
        ]
    ).to_csv(csv, index=False)

    _run(monkeypatch, "resample", "--batch", csv)

    assert tuple(mm.ZarrManager(str(tmp_path / "a.ome.zarr")).metadata["scales"][0]) == (1.0, 1.0, 1.0)
    assert tuple(mm.ZarrManager(str(tmp_path / "b.ome.zarr")).metadata["scales"][0]) == (2.0, 2.0, 2.0)
    assert tuple(mm.ZarrManager(str(taken)).metadata["scales"][0]) == (1.0, 1.0, 1.0)
    assert "1 row(s) skipped" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["-z", "out.zarr"],  # no pixel size
        ["-ps", "1"],  # no output
        ["-z", "out.zarr", "-ps", "1", "2"],  # two values
    ],
)
def test_argument_errors(store, monkeypatch, argv):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "resample", "-i", store, *argv)


def test_batch_refuses_other_options(store, tmp_path, monkeypatch):
    csv = tmp_path / "images.csv"
    csv.write_text("input,output,pixel_size\n")
    for extra in (["--fast"], ["--overwrite"], ["-ps", "1"], ["-i", store]):
        with pytest.raises(SystemExit):
            _run(monkeypatch, "resample", "--batch", csv, *extra)
