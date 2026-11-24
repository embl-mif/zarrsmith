"""``--dry_run`` prints the recap the napari widgets show and writes nothing; ``--subset c=`` takes names."""
from __future__ import annotations

import numpy as np
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.cli import zarrsmith as cli
from zarrsmith.cli.__arguments import parse_subset_spec
from zarrsmith.utils.chunks import chunk_shard_layout, layout_summary, output_summary_line


def _image(tmp_path, channels=("DAPI", "GFP", "RFP")):
    data = np.arange(len(channels) * 40 * 300 * 500, dtype=np.uint16).reshape(len(channels), 40, 300, 500)
    md = {"axes": "czyx", "scales": [(2.0, 0.5, 0.5)], "channel_names": list(channels)}
    path = tmp_path / "in.ome.zarr"
    mm.ArrayManager(data, md).to_zarr(str(path), progress=False)
    return path


def test_convert_dry_run_prints_the_widget_recap_and_writes_nothing(tmp_path, capsys):
    src = _image(tmp_path)
    out = tmp_path / "out.ome.zarr"
    cli.main(["convert", "-i", str(src), "-z", str(out), "--dry_run", "-stm", "16"])
    printed = capsys.readouterr().out
    assert not out.exists()
    assert "Dry run: nothing written" in printed
    # the lines the napari converter shows for the same output (shared functions)
    dataset = mm.ZarrManager(str(src))
    axes, md = "czyx", dataset.metadata
    shape = md["size"][0]
    layout = chunk_shard_layout(axes, shape, md["dtype"], shard_mb=16, voxel_size=md["scales"][0])
    assert f"Zarr: {layout_summary(axes, md['dtype'], layout)}" in printed
    assert layout["shards"] is not None  # 16 MB shards of whole 4 MB chunks
    line = output_summary_line(axes, dict(zip(axes, shape)), md["dtype"], md["scales"][0], md["units"])
    assert f"Output: {line}" in printed


def test_dry_run_reports_an_output_in_the_way_instead_of_stopping(tmp_path, capsys):
    src = _image(tmp_path)
    out = tmp_path / "out.ome.zarr"
    out.mkdir()
    (out / "something.txt").write_text("x")
    cli.main(["convert", "-i", str(src), "-z", str(out), "--dry_run"])
    printed = capsys.readouterr().out
    assert "the real run would stop here" in printed and (out / "something.txt").exists()


def test_subset_selects_channels_by_name(tmp_path):
    assert parse_subset_spec("c=DAPI,RFP;z=0:10") == {"C": ["DAPI", "RFP"], "Z": slice(0, 10, None)}
    assert parse_subset_spec("c=GFP,0") == {"C": ["GFP", 0]}  # numbers stay indices
    assert parse_subset_spec("c=0,2") == {"C": [0, 2]}
    assert cli._channel_indices(["RFP", 0], ["DAPI", "GFP", "RFP"]) == [2, 0]
    with pytest.raises(ValueError, match="no channel named 'Cy5'"):
        cli._channel_indices(["Cy5"], ["DAPI", "GFP"])

    src = _image(tmp_path)
    out = tmp_path / "out.ome.zarr"
    cli.main(["convert", "-i", str(src), "-z", str(out), "-s", "c=RFP,DAPI", "-nl", "1"])
    zm = mm.ZarrManager(str(out))
    assert zm.metadata["channel_names"] == ["RFP", "DAPI"]
    assert zm.data[0].shape[0] == 2


def test_fuse_dry_run_prints_the_widget_recap_and_writes_nothing(tmp_path, capsys):
    from zarrsmith.mosaic_manager import ZarrMosaicManager, plan_fusion
    from zarrsmith.mosaic_manager.fusion import FUSION_SHARD_TARGET_MB, fusion_workers
    from zarrsmith.utils.chunks import level0_layout

    folder = tmp_path / "tiles"
    folder.mkdir()
    for name, x0 in (("t1", 0.0), ("t2", 200.0)):
        md = {"axes": "zyx", "scales": [(2.0, 0.5, 0.5)], "translations": {"z": 0.0, "y": 0.0, "x": x0}}
        mm.ArrayManager(np.full((20, 300, 500), 7, np.uint16), md).to_zarr(str(folder / f"{name}.ome.zarr"),
                                                                           progress=False)
    out = tmp_path / "fused.ome.zarr"
    cli.main(["fuse", "-i", str(folder), "-z", str(out), "--dry_run"])
    printed = capsys.readouterr().out
    assert not out.exists() and "Dry run: nothing written" in printed
    plan = plan_fusion(ZarrMosaicManager(str(folder)).fusion_tiles())
    assert f"Output: {plan.summary()}" in printed
    chunks = cli.default_chunk_size({"axes": plan.axes, "size": [plan.shape], "dtype": str(plan.dtype),
                                     "scales": [plan.scale]})[0]
    layout = level0_layout(plan.axes, plan.shape, plan.dtype, chunks, shard_mb=FUSION_SHARD_TARGET_MB)
    layout["workers"] = fusion_workers(plan, layout["shards"] or layout["chunks"], 0)
    layout["auto_workers"] = True
    assert f"Zarr: {layout_summary(plan.axes, plan.dtype, layout)}" in printed
