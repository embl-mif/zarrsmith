"""Writing must never delete data the user did not explicitly ask to replace."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.cli import zarrsmith as cli
from zarrsmith.mosaic_manager import Tile, fuse_tiles, fusion
from zarrsmith.utils.output_path import check_output_path, has_zarr_name, is_zarr_store, output_refusal


def _manager(value=1):
    return mm.ArrayManager(
        np.full((2, 4, 8, 8), value, dtype="uint16"),
        {"axes": "czyx", "channel_names": ["a", "b"], "scales": [(1.0, 1.0, 1.0)]},
    )


def _store(path, value=1):
    _manager(value).to_zarr(str(path), progress=False)
    return path


def _plain_folder(path):
    """A folder of the user's that is not a dataset."""
    path.mkdir()
    (path / "keep.txt").write_text("precious")
    return path


def _value(path):
    return int(mm.ZarrManager(str(path)).data[0].max().compute())


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["zarrsmith", *map(str, argv)])
    cli.main()


# --- the check itself ---

def test_names():
    assert has_zarr_name("a/b.zarr") and has_zarr_name("b.ome.zarr") and has_zarr_name("B.ZARR")
    assert not has_zarr_name("a.zarr/b") and not has_zarr_name("b.zarr.bak") and not has_zarr_name("folder")


def test_missing_path_and_empty_folder_can_be_written_to(tmp_path):
    assert output_refusal(tmp_path / "new.zarr") is None
    (tmp_path / "empty.zarr").mkdir()
    assert output_refusal(tmp_path / "empty.zarr") is None


def test_store_needs_overwrite_and_says_how(tmp_path):
    store = _store(tmp_path / "a.zarr")
    assert is_zarr_store(store)
    assert "--overwrite" in output_refusal(store, hint="pass --overwrite")
    assert output_refusal(store, overwrite=True) is None


@pytest.mark.parametrize("overwrite", [False, True])
def test_other_content_is_never_replaced(tmp_path, overwrite):
    assert output_refusal(_plain_folder(tmp_path / "folder"), overwrite=overwrite) is not None
    (tmp_path / "file.zarr").write_text("x")
    assert output_refusal(tmp_path / "file.zarr", overwrite=overwrite) is not None


def test_output_cannot_replace_its_input(tmp_path):
    store = _store(tmp_path / "parent.zarr" / "a.zarr")
    (tmp_path / "parent.zarr" / "zarr.json").write_text("{}")  # looks replaceable
    for out in (store, store.parent, store / "0"):
        assert output_refusal(out, overwrite=True, sources=[None, store]) is not None
    assert output_refusal(tmp_path / "other.zarr", overwrite=True, sources=[store]) is None


def test_output_can_go_inside_a_plain_input_folder(tmp_path):
    # the default of the mosaic widget: next to the raw data of the acquisition
    acquisition = _plain_folder(tmp_path / "acq")
    assert output_refusal(acquisition / "fused.ome.zarr", sources=[acquisition]) is None


def test_warnings(tmp_path):
    with pytest.warns(UserWarning, match="not named .zarr"):
        check_output_path(tmp_path / "out")
    store = _store(tmp_path / "a.zarr")
    with pytest.warns(UserWarning, match="Replacing the existing Zarr store"):
        check_output_path(store, overwrite=True)


# --- Python API ---

def test_to_zarr_defaults_to_not_overwriting(tmp_path):
    out = _store(tmp_path / "a.zarr", value=1)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        _manager(2).to_zarr(str(out))
    assert _value(out) == 1


def test_to_zarr_overwrite_replaces_a_store_with_a_warning(tmp_path):
    out = _store(tmp_path / "a.zarr", value=1)
    with pytest.warns(UserWarning, match="Replacing"):
        _manager(2).to_zarr(str(out), overwrite=True, progress=False)
    assert _value(out) == 2


@pytest.mark.parametrize("overwrite", [False, True])
def test_to_zarr_keeps_a_plain_folder(tmp_path, overwrite):
    folder = _plain_folder(tmp_path / "folder")
    with pytest.raises(FileExistsError):
        _manager().to_zarr(str(folder), overwrite=overwrite)
    assert (folder / "keep.txt").read_text() == "precious"


def test_to_zarr_into_an_empty_folder(tmp_path):
    (tmp_path / "a.zarr").mkdir()
    _manager(3).to_zarr(str(tmp_path / "a.zarr"), progress=False)
    assert _value(tmp_path / "a.zarr") == 3


def test_zarr_manager_to_zarr_defaults_to_not_overwriting(tmp_path):
    source, out = _store(tmp_path / "a.zarr", 1), _store(tmp_path / "b.zarr", 2)
    with pytest.raises(FileExistsError):
        mm.ZarrManager(str(source)).to_zarr(str(out))
    assert _value(out) == 2


@pytest.mark.parametrize("overwrite", [False, True])
def test_zarr_manager_cannot_overwrite_its_own_store(tmp_path, overwrite):
    # the data is read lazily from there: replacing the store used to leave only zeros
    source = _store(tmp_path / "a.zarr", 7)
    with pytest.raises(FileExistsError, match="input"):
        mm.ZarrManager(str(source)).to_zarr(str(source), overwrite=overwrite)
    assert _value(source) == 7


def test_create_empty_group_keeps_an_existing_group(tmp_path):
    z = mm.ZarrManager(str(_store(tmp_path / "a.zarr")), mode="r+")
    metadata = dict(z.metadata)
    z.create_empty_group("processed", metadata)
    z.root["processed/0"][0, 0, 0, 0] = 5
    with pytest.raises(FileExistsError, match="overwrite=True"):
        z.create_empty_group("processed", metadata)
    assert z.root["processed/0"][0, 0, 0, 0] == 5
    with pytest.warns(UserWarning, match="Replacing the existing group"):
        z.create_empty_group("processed", metadata, overwrite=True)
    assert z.root["processed/0"][0, 0, 0, 0] == 0


@pytest.mark.parametrize("overwrite", [False, True])
def test_create_empty_group_never_replaces_a_pyramid_level(tmp_path, overwrite):
    z = mm.ZarrManager(str(_store(tmp_path / "a.zarr", 9)), mode="r+")
    with pytest.raises(FileExistsError, match="array"):
        z.create_empty_group("0", dict(z.metadata), overwrite=overwrite)
    assert _value(tmp_path / "a.zarr") == 9


def test_chromatic_shift_defaults_to_not_overwriting(tmp_path):
    z = mm.ZarrManager(str(_store(tmp_path / "a.zarr")))
    out = _store(tmp_path / "b.zarr", 2)
    with pytest.raises(FileExistsError):
        z.apply_chromatic_shift(out, [0.0, 0.0], progress=False)
    with pytest.raises(FileExistsError):
        z.correct_chromatic_shift(out)
    assert _value(out) == 2
    folder = _plain_folder(tmp_path / "folder")
    with pytest.raises(FileExistsError):
        z.apply_chromatic_shift(folder, [0.0, 0.0], overwrite=True, progress=False)
    assert (folder / "keep.txt").exists()


def _tile():
    return Tile(np.full((4, 8, 8), 1, np.uint16), (0, 0, 0), (1.0, 1.0, 1.0), "A", None)


def test_failed_fusion_keeps_the_previous_output(tmp_path, monkeypatch):
    out = tmp_path / "fused.zarr"
    fuse_tiles([_tile()], out, progress=False)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(fusion, "_fuse_level0", fail)
    with pytest.raises(OSError, match="disk full"):
        fuse_tiles([_tile()], out, overwrite=True, progress=False)
    assert _value(out) == 1


def test_fusion_keeps_a_plain_folder(tmp_path):
    folder = _plain_folder(tmp_path / "folder")
    with pytest.raises(FileExistsError):
        fuse_tiles([_tile()], folder, overwrite=True, progress=False)
    assert (folder / "keep.txt").exists()


def test_fusion_temp_dir_cannot_be_inside_a_replaced_output(tmp_path):
    out = tmp_path / "fused.zarr"
    fuse_tiles([_tile()], out, progress=False)
    with pytest.raises(FileExistsError):
        fuse_tiles([_tile()], out, overwrite=True, temp_dir=out / "tmp", progress=False)
    assert _value(out) == 1


# --- command line ---

def test_convert_needs_overwrite(tmp_path, monkeypatch):
    source, out = _store(tmp_path / "in.zarr", 1), _store(tmp_path / "out.zarr", 2)
    with pytest.raises(FileExistsError, match="--overwrite"):
        _run(monkeypatch, "convert", "-i", source, "-z", out, "-m", "zarr")
    assert _value(out) == 2
    with pytest.warns(UserWarning, match="Replacing"):
        _run(monkeypatch, "convert", "-i", source, "-z", out, "-m", "zarr", "--overwrite")
    assert _value(out) == 1


@pytest.mark.parametrize("flags", [(), ("--overwrite",)])
def test_convert_keeps_a_plain_folder(tmp_path, monkeypatch, flags):
    folder = _plain_folder(tmp_path / "folder")
    with pytest.raises(FileExistsError):
        _run(monkeypatch, "convert", "-i", _store(tmp_path / "in.zarr"), "-z", folder, "-m", "zarr", *flags)
    assert (folder / "keep.txt").exists()


def test_convert_cannot_overwrite_its_input(tmp_path, monkeypatch):
    source = _store(tmp_path / "in.zarr", 4)
    with pytest.raises(FileExistsError, match="input"):
        _run(monkeypatch, "convert", "-i", source, "-z", source, "-m", "zarr", "--overwrite")
    assert _value(source) == 4


def test_convert_warns_about_the_output_name(tmp_path, monkeypatch):
    with pytest.warns(UserWarning, match="not named .zarr"):
        _run(monkeypatch, "convert", "-i", _store(tmp_path / "in.zarr"), "-z", tmp_path / "out", "-m", "zarr")
    assert is_zarr_store(tmp_path / "out")


def test_convert_batch_skips_the_row_in_the_way_only(tmp_path, monkeypatch, capsys):
    source = _store(tmp_path / "in.zarr", 1)
    kept, replaced, new = _store(tmp_path / "kept.zarr", 2), _store(tmp_path / "replaced.zarr", 2), tmp_path / "new.zarr"
    csv = tmp_path / "batch.csv"
    pd.DataFrame([{"input": source, "microscope": "zarr", "output": kept, "overwrite": ""},
                  {"input": source, "microscope": "zarr", "output": replaced, "overwrite": "true"},
                  {"input": source, "microscope": "zarr", "output": new, "overwrite": ""}]).to_csv(csv, index=False)
    _run(monkeypatch, "convert", "--batch", csv)
    assert (_value(kept), _value(replaced), _value(new)) == (2, 1, 1)
    err = capsys.readouterr().err
    assert "row skipped" in err and f"1 row(s) skipped, nothing was written for them: {kept}" in err


@pytest.mark.parametrize("command", ["convert", "fuse", "chromatic-shift"])
def test_overwrite_of_a_batch_is_in_its_csv(tmp_path, monkeypatch, capsys, command):
    csv = tmp_path / "batch.csv"
    csv.write_text("input,output\n")
    with pytest.raises(SystemExit):
        _run(monkeypatch, command, "--batch", csv, "--overwrite")
    assert "overwrite column" in capsys.readouterr().err


def test_chromatic_shift_keeps_an_existing_report(tmp_path, monkeypatch, capsys):
    source = _store(tmp_path / "in.zarr")
    report = tmp_path / "shift.json"
    report.write_text('{"edited": "by hand"}')
    with pytest.raises(FileExistsError, match="--overwrite"):
        _run(monkeypatch, "chromatic-shift", "-i", source, "-sf", 0, 0, "-r", report)
    assert json.loads(report.read_text()) == {"edited": "by hand"}
    _run(monkeypatch, "chromatic-shift", "-i", source, "-sf", 0, 0, "-r", report, "--overwrite")
    assert json.loads(report.read_text()) == [0.0, 0.0]
    assert "replacing the existing report" in capsys.readouterr().err


def test_chromatic_shift_batch_skips_the_row_in_the_way_only(tmp_path, monkeypatch, capsys):
    source = _store(tmp_path / "in.zarr", 1)
    kept, new = _store(tmp_path / "kept.zarr", 2), tmp_path / "new.zarr"
    csv = tmp_path / "batch.csv"
    pd.DataFrame([{"input": source, "output": kept, "shifts": "0 0"},
                  {"input": source, "output": new, "shifts": "0 0"}]).to_csv(csv, index=False)
    _run(monkeypatch, "chromatic-shift", "--batch", csv)
    assert (_value(kept), _value(new)) == (2, 1)
    assert "1 row(s) skipped" in capsys.readouterr().err
