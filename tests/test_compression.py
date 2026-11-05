"""Default codec of the writers, its options and how an unusable codec is refused."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import zarr
from numcodecs import Blosc, GZip

import zarrsmith.microscope_manager as mm
from zarrsmith.cli import zarrsmith as cli
from zarrsmith.cli.zarrsmith import _compression_kwargs
from zarrsmith.utils.create_empty_dataset import create_empty_dataset
from zarrsmith.utils.ngff import ZarrWriteConfig, _resolve_format

_DATA = np.random.default_rng(0).integers(100, 140, (2, 8, 64, 64), dtype="uint16")
_METADATA = {
    "axes": "czyx",
    "scales": [(1.0, 1.0, 1.0)],
    "channel_names": ["A", "B"],
    "channel_colors": ["FF0000", "00FF00"],
}


def _manager():
    return mm.ArrayManager(_DATA, dict(_METADATA), chunks=(1, 4, 32, 32))


def _write(path, **options):
    _manager().to_zarr(str(path), progress=False, **options)
    return zarr.open_group(str(path), mode="r")["0"]


def _v3_blosc_config(array):
    assert len(array.compressors) == 1
    config = array.compressors[0].to_dict()
    assert config["name"] == "blosc"
    return config["configuration"]


def _stored_bytes(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@pytest.mark.parametrize("shards", [None, "auto"])
def test_default_is_blosc_zstd_byte_shuffle_level_1_on_v3(tmp_path, shards):
    out = tmp_path / "out.zarr"
    array = _write(out, shards=shards)
    config = _v3_blosc_config(array)
    assert (config["cname"], config["clevel"], config["shuffle"]) == ("zstd", 1, "shuffle")
    np.testing.assert_array_equal(array[:], _DATA)
    assert _stored_bytes(out / "0") < _DATA.nbytes / 2


def test_default_is_blosc_zstd_byte_shuffle_level_1_on_v2(tmp_path):
    array = _write(tmp_path / "out.zarr", zarr_format=2, ngff_version="0.4")
    codec = array.compressors[0]
    assert isinstance(codec, Blosc)
    assert (codec.cname, codec.clevel, codec.shuffle) == ("zstd", 1, Blosc.SHUFFLE)
    np.testing.assert_array_equal(array[:], _DATA)


@pytest.mark.parametrize("options", [{}, {"zarr_format": 2, "ngff_version": "0.4"}])
def test_none_stays_uncompressed(tmp_path, options):
    array = _write(tmp_path / "out.zarr", compressor=None, **options)
    assert len(array.compressors) == 0
    np.testing.assert_array_equal(array[:], _DATA)


def test_level_and_gzip_are_applied(tmp_path):
    assert _v3_blosc_config(_write(tmp_path / "a.zarr", compressor_level=5))["clevel"] == 5
    gzip = _write(tmp_path / "b.zarr", zarr_format=2, ngff_version="0.4", compressor="gzip", compressor_level=4)
    assert isinstance(gzip.compressors[0], GZip) and gzip.compressors[0].level == 4


@pytest.mark.parametrize("options, message", [
    ({"compressor": "gzip"}, "Unsupported compressor"),
    ({"compressor": "zstd"}, "Unsupported compressor"),
    ({"compressor_level": 10}, "compressor_level"),
    ({"compressor_level": -1}, "compressor_level"),
    ({"compressor_level": 1.5}, "compressor_level"),
])
def test_unusable_codec_leaves_an_existing_store_alone(tmp_path, options, message):
    out = tmp_path / "out.zarr"
    _write(out)
    before = _stored_bytes(out)
    with pytest.raises(ValueError, match=message):
        _manager().to_zarr(str(out), overwrite=True, progress=False, **options)
    assert _stored_bytes(out) == before
    np.testing.assert_array_equal(zarr.open_group(str(out), mode="r")["0"][:], _DATA)


def test_unusable_codec_creates_nothing(tmp_path):
    out = tmp_path / "out.zarr"
    with pytest.raises(ValueError):
        _manager().to_zarr(str(out), compressor="gzip", progress=False)
    assert not out.exists()


def test_level_is_not_checked_without_a_compressor():
    assert _resolve_format(ZarrWriteConfig(compressor=None, compressor_level=99)) == ("0.5", 3)


def test_zarr_manager_compresses_raw_data_and_labels(tmp_path):
    src = tmp_path / "src.zarr"
    _write(src, compressor=None)
    manager = mm.ZarrManager(str(src), mode="a")
    manager.create_empty_group("seg", {**manager.metadata, "data_type": "label"}, is_label=True)

    out = tmp_path / "out.zarr"
    manager.to_zarr(str(out), progress=False)
    root = zarr.open_group(str(out), mode="r")
    assert _v3_blosc_config(root["0"])["cname"] == "zstd"
    assert _v3_blosc_config(root["labels/seg/0"])["cname"] == "zstd"


def test_empty_arrays_keep_their_own_default(tmp_path):
    metadata = {
        "axes": "zyx", "size": [(8, 64, 64)], "chunksize": [(4, 32, 32)], "scales": [(1.0, 1.0, 1.0)],
        "units": ("micrometer",) * 3, "dtype": "uint16", "channel_names": [], "channel_colors": [],
        "time_increment": None, "time_increment_unit": None,
    }
    v3 = zarr.open_group(str(tmp_path / "v3.zarr"), mode="w", zarr_format=3)
    create_empty_dataset(v3, metadata)
    assert type(v3["0"].compressors[0]).__name__ == "ZstdCodec"

    v2 = zarr.open_group(str(tmp_path / "v2.zarr"), mode="w", zarr_format=2)
    create_empty_dataset(v2, metadata, ngff_version="0.4", zarr_format=2)
    assert len(v2["0"].compressors) == 0

    asked = zarr.open_group(str(tmp_path / "asked.zarr"), mode="w", zarr_format=3)
    create_empty_dataset(asked, metadata, compressor="blosc")
    assert _v3_blosc_config(asked["0"])["clevel"] == 1


# --- command line and batch CSV ---

def test_compression_kwargs():
    assert _compression_kwargs("blosc", None) == {"compressor": "blosc"}
    assert _compression_kwargs(" None ", None) == {"compressor": None}
    assert _compression_kwargs(None, None) == {"compressor": None}
    assert _compression_kwargs("gzip", 4.0) == {"compressor": "gzip", "compressor_level": 4}


def _convert(monkeypatch, src, out, *argv):
    monkeypatch.setattr(sys, "argv", ["zarrsmith", "convert", "-i", str(src), "-m", "zarr", "-z", str(out),
                                      "-nl", "1", *argv])
    cli.main()
    return zarr.open_group(str(out), mode="r")["0"]


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src.zarr"
    _write(src, compressor=None)
    return src


def test_convert_compresses_by_default(source, tmp_path, monkeypatch):
    config = _v3_blosc_config(_convert(monkeypatch, source, tmp_path / "out.zarr"))
    assert (config["cname"], config["clevel"], config["shuffle"]) == ("zstd", 1, "shuffle")


def test_convert_options(source, tmp_path, monkeypatch):
    assert len(_convert(monkeypatch, source, tmp_path / "none.zarr", "-cp", "none").compressors) == 0
    leveled = _convert(monkeypatch, source, tmp_path / "level.zarr", "--compressor", "BLOSC", "--compressor_level", "4")
    assert _v3_blosc_config(leveled)["clevel"] == 4
    gzip = _convert(monkeypatch, source, tmp_path / "gzip.zarr", "-zf", "2", "-cp", "gzip", "-cpl", "2")
    assert isinstance(gzip.compressors[0], GZip) and gzip.compressors[0].level == 2


def test_convert_refuses_an_unknown_codec_name(source, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["zarrsmith", "convert", "-i", str(source), "-z", str(tmp_path / "o.zarr"),
                                      "-cp", "lz4"])
    with pytest.raises(SystemExit):
        cli.main()


def test_convert_gzip_on_v3_keeps_the_existing_output(source, tmp_path, monkeypatch):
    out = tmp_path / "out.zarr"
    _convert(monkeypatch, source, out)
    before = _stored_bytes(out)
    with pytest.raises(ValueError, match="Unsupported compressor"):
        _convert(monkeypatch, source, out, "-cp", "gzip", "--overwrite")
    assert _stored_bytes(out) == before


def test_batch_columns(source, tmp_path, monkeypatch):
    outputs = {name: tmp_path / f"{name}.zarr" for name in ("empty", "none", "level")}
    csv = tmp_path / "batch.csv"
    pd.DataFrame([
        {"input": str(source), "microscope": "zarr", "output": str(outputs["empty"]), "num_levels": 1,
         "compressor": "", "compressor_level": ""},
        {"input": str(source), "microscope": "zarr", "output": str(outputs["none"]), "num_levels": 1,
         "compressor": "none", "compressor_level": ""},
        {"input": str(source), "microscope": "zarr", "output": str(outputs["level"]), "num_levels": 1,
         "compressor": "blosc", "compressor_level": 3},
    ]).to_csv(csv, index=False)
    monkeypatch.setattr(sys, "argv", ["zarrsmith", "convert", "--batch", str(csv)])
    cli.main()

    def array(name):
        return zarr.open_group(str(outputs[name]), mode="r")["0"]

    assert _v3_blosc_config(array("empty"))["clevel"] == 1
    assert len(array("none").compressors) == 0
    assert _v3_blosc_config(array("level"))["clevel"] == 3
