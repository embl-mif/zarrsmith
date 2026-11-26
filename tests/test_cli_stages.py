"""The command line grouped by stage: one meaning per short flag, old spellings still working,
--help-all, and the commands plan-tiles and fuse-labels (with --pyramid_like)."""
from __future__ import annotations

import argparse
import csv
import json

import dask.array as da
import numpy as np
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.cli import zarrsmith as cli
from zarrsmith.cli.__arguments import STAGES, _build_parser, _parse_arguments
from zarrsmith.utils.dataset_ops import pyramid_like

COMMANDS = [name for _, commands in STAGES for name, *_ in commands]


def _visible_short_flags():
    """short flag -> {(long option, command)}, over the options shown by --help-all."""
    _, subparsers = _build_parser(show_all=True)
    flags = {}
    for name in COMMANDS:
        for action in subparsers.choices[name]._actions:
            if action.help is argparse.SUPPRESS:
                continue
            longs = [o for o in action.option_strings if o.startswith("--")]
            for opt in action.option_strings:
                if not opt.startswith("--") and opt != "-h":
                    flags.setdefault(opt, set()).add((longs[0] if longs else opt, name))
    return flags


def test_every_short_flag_means_one_option():
    clashes = {flag: uses for flag, uses in _visible_short_flags().items() if len({long for long, _ in uses}) > 1}
    assert clashes == {}


def test_overview_lists_every_command_by_stage(capsys):
    with pytest.raises(SystemExit):
        _parse_arguments(["-h"])
    text = capsys.readouterr().out
    for title, commands in STAGES:
        assert f"{title}:" in text
        for name, *_ in commands:
            assert f"  {name} " in text
    assert text.index("convert") < text.index("plan-tiles") < text.index("fuse-labels")


def test_help_hides_advanced_options_until_help_all(capsys):
    with pytest.raises(SystemExit):
        _parse_arguments(["fuse", "-h"])
    short = capsys.readouterr().out
    with pytest.raises(SystemExit):
        _parse_arguments(["fuse", "--help-all"])
    full = capsys.readouterr().out
    assert "--compressor_level" not in short and "--compressor_level" in full
    assert "--zarr_format" in short  # common storage options stay visible
    assert "more options" in short and "--help-all" in short
    for section in ("Input and output", "What to read", "Fusion", "Tile positions", "Pyramid", "Run"):
        assert section in short
    heads = [full.index(f"\n{title}:\n") for title in ("What to read", "Pyramid", "Storage layout", "Run")]
    assert heads == sorted(heads)


def test_hidden_options_still_parse(tmp_path):
    args = _parse_arguments(["convert", "-i", str(tmp_path), "-z", str(tmp_path / "o.ome.zarr"), "-cpl", "5"])
    assert args.compressor_level == 5 and not hasattr(args, "help_all")


@pytest.mark.parametrize("old, new, dest, value", [
    (["fuse", "-t", "1-x00-y00"], ["fuse", "-tl", "1-x00-y00"], "tiles", ["1-x00-y00"]),
    (["fuse", "-f", "field1"], ["fuse", "-fi", "field1"], "fields", ["field1"]),
    (["fuse", "-g", "L", "R"], ["fuse", "-gr", "L", "R"], "groups", ["L", "R"]),
    (["fuse", "-od", "uint16"], ["fuse", "-dt", "uint16"], "out_dtype", "uint16"),
    (["fuse", "-dt", "label"], ["fuse", "-dty", "label"], "data_type", "label"),
    (["chromatic-shift", "-rc", "2"], ["chromatic-shift", "-ref", "2"], "reference_channel", "2"),
    (["chromatic-shift", "--timepoint", "3"], ["chromatic-shift", "--timepoints", "3"], "timepoint", 3),
])
def test_old_spellings_still_work_with_a_note(tmp_path, capsys, old, new, dest, value):
    io = ["-i", str(tmp_path), "-z", str(tmp_path / "o.ome.zarr")]
    got_old = vars(_parse_arguments(old[:1] + io + old[1:]))
    note = capsys.readouterr().err
    got_new = vars(_parse_arguments(new[:1] + io + new[1:]))
    assert got_old[dest] == got_new[dest] == value
    assert got_old == got_new
    assert "Note:" in note and "still works" in note or "sets the data type" in note


def test_allocate_takes_z_and_old_o(tmp_path, capsys):
    a = _parse_arguments(["allocate", "-t", str(tmp_path), "-z", str(tmp_path / "a.ome.zarr")])
    b = _parse_arguments(["allocate", "-t", str(tmp_path), "-o", str(tmp_path / "a.ome.zarr")])
    assert a.out == b.out
    assert "-o is now -z/--zarr_path" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _parse_arguments(["allocate", "-t", str(tmp_path)])


# ---- pyramid_like -----------------------------------------------------------------------

def _image(path, shape=(8, 64, 96), levels=3, factor=(1, 2, 2), scale=(2.0, 0.5, 0.5)):
    data = np.random.default_rng(0).integers(0, 1000, shape).astype(np.uint16)
    md = {"axes": "zyx", "size": [shape], "chunksize": [(4, 32, 32)], "dtype": "uint16", "data_type": "intensity",
          "scales": [scale], "units": ("micrometer",) * 3}
    m = mm.ArrayManager([da.from_array(data)], md)
    m.build_pyramid(num_levels=levels, downscale_factor=factor)
    m.to_zarr(str(path), progress=False)
    return path


def test_pyramid_like_from_a_level(tmp_path):
    md = mm.ZarrManager(str(_image(tmp_path / "img.ome.zarr", levels=4)), mode="r").metadata
    assert pyramid_like(md, 0) == (4, (1, 2, 2))
    assert pyramid_like(md, 1) == (3, (1, 2, 2))
    assert pyramid_like(md, 3) == (1, (2, 2, 2))
    with pytest.raises(ValueError, match="out of range"):
        pyramid_like(md, 4)


# ---- plan-tiles -------------------------------------------------------------------------

def test_plan_tiles_writes_bounds_and_neighbours(tmp_path):
    img = _image(tmp_path / "img.ome.zarr")
    out = tmp_path / "tiles.csv"
    cli.main(["plan-tiles", "-i", str(img), "-oc", str(out), "-ts", "y=40;x=40", "-ov", "8", "-ha", "4"])
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2 * 3  # y: 0-40, 32-64; x: 0-40, 32-72, 64-96
    first = rows[0]
    assert first["dataset_id"] == "img__tile_000001" and first["source_dataset_id"] == "img"
    assert (first["x_min"], first["x_max"], first["y_min"], first["y_max"]) == ("0", "40", "0", "40")
    assert (first["z_min"], first["z_max"], first["tile_overlap_z"]) == ("0", "8", "0")  # z left whole
    assert first["right_tile_index"] == "2" and first["bottom_tile_index"] == "4" and first["behind_tile_index"] == ""
    last = rows[-1]
    assert last["right_dataset_id"] == last["bottom_dataset_id"] == ""


def test_plan_tiles_budget_and_level(tmp_path, capsys):
    img = _image(tmp_path / "img.ome.zarr")
    out = tmp_path / "tiles.csv"
    cli.main(["plan-tiles", "-i", str(img), "-oc", str(out), "-lv", "1", "-mtv", "4000", "-ov", "4", "-id", "a"])
    rows = list(csv.DictReader(open(out)))
    assert all(r["resolution_level"] == "1" and r["source_dataset_id"] == "a" for r in rows)
    assert max(int(r["x_max"]) for r in rows) == 48 and len(rows) > 1  # level 1 is 32 x 48
    cli.main(["plan-tiles", "-i", str(img), "-oc", str(tmp_path / "none.csv"), "-ts", "40", "-dr"])
    assert "Dry run" in capsys.readouterr().out and not (tmp_path / "none.csv").exists()


def test_plan_tiles_refuses_fractional_voxels(tmp_path):
    img = _image(tmp_path / "img.ome.zarr")
    with pytest.raises(ValueError, match="whole number"):
        cli.main(["plan-tiles", "-i", str(img), "-oc", str(tmp_path / "t.csv"), "-ts", "40.5"])


# ---- fuse-labels ------------------------------------------------------------------------

def _label_tiles(tmp_path):
    """Two label tiles overlapping by 8 along x, each numbered from 1; object 2 of tile 1 and
    object 1 of tile 2 are the same cell across the seam."""
    folder = tmp_path / "tiles"
    folder.mkdir()
    shape = (8, 64, 24)
    a, b = np.zeros(shape, np.uint16), np.zeros(shape, np.uint16)
    a[..., 0:10], a[..., 12:24] = 1, 2
    b[..., 0:8], b[..., 10:20] = 1, 2
    for name, data, x0 in (("t1", a, 0.0), ("t2", b, 16.0)):  # voxel size of level 1 of _image's
        md = {"axes": "zyx", "scales": [(2.0, 1.0, 1.0)], "data_type": "label",
              "translations": {"z": 0.0, "y": 0.0, "x": x0}}
        mm.ArrayManager(data, md).to_zarr(str(folder / f"{name}.ome.zarr"), progress=False)
    (tmp_path / "offsets.csv").write_text("store,offset\nt1,1000\nt2,2000\n")
    (tmp_path / "mapping.json").write_text(json.dumps({"2001": 1002}))
    return folder


def test_fuse_labels_renumbers_and_follows_the_source_pyramid(tmp_path):
    folder = _label_tiles(tmp_path)
    img = _image(tmp_path / "img.ome.zarr", shape=(8, 128, 80), levels=4)
    out = tmp_path / "merged.ome.zarr"
    cli.main(["fuse-labels", "-i", str(folder), "-z", str(out), "-lo", str(tmp_path / "offsets.csv"),
              "-lm", str(tmp_path / "mapping.json"), "--pyramid_like", str(img), "--pyramid_start_level", "1"])
    zm = mm.ZarrManager(str(out), mode="r")
    merged = np.asarray(zm.data[0])
    assert zm.metadata["data_type"] == "label" and merged.dtype == np.uint32
    assert set(np.unique(merged)) == {0, 1001, 1002, 2002}
    assert (merged[..., 12:24] == 1002).all()  # one cell across the seam
    assert len(zm.data) == 3  # levels 1, 2, 3 of the image
    src = mm.ZarrManager(str(img), mode="r").metadata["scales"]
    assert [tuple(s) for s in zm.metadata["scales"]] == [tuple(s) for s in src[1:]]


def test_fuse_labels_without_offsets_reads_the_stores_as_they_are(tmp_path):
    folder = _label_tiles(tmp_path)
    out = tmp_path / "merged.ome.zarr"
    cli.main(["fuse-labels", "-i", str(folder), "-z", str(out), "-nl", "1"])
    assert mm.ZarrManager(str(out), mode="r").metadata["data_type"] == "label"


def test_fuse_labels_needs_an_integer_dtype(tmp_path):
    folder = _label_tiles(tmp_path)
    with pytest.raises(SystemExit):
        _parse_arguments(["fuse-labels", "-i", str(folder), "-z", str(tmp_path / "o.ome.zarr"), "-dt", "float32"])


def test_fuse_still_takes_the_label_options_with_a_note(tmp_path, capsys):
    folder = _label_tiles(tmp_path)
    out = tmp_path / "merged.ome.zarr"
    cli.main(["fuse", "-i", str(folder), "-z", str(out), "-lo", str(tmp_path / "offsets.csv"),
              "-lm", str(tmp_path / "mapping.json"), "-nl", "1"])
    assert "fuse-labels --label_offsets" in capsys.readouterr().err
    assert set(np.unique(np.asarray(mm.ZarrManager(str(out), mode="r").data[0]))) == {0, 1001, 1002, 2002}
