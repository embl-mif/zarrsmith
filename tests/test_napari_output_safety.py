"""The napari widgets check the output name, then what is at the output, before writing anything."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("napari")

from napari.components import ViewerModel
from qtpy.QtWidgets import QApplication

import zarrsmith.microscope_manager as mm
import zarrsmith.napari._convert_widget as convert_module
import zarrsmith.napari._export_widget as export_module
from zarrsmith.napari import _mosaic_widget as mw
from zarrsmith.napari import _output_check
from zarrsmith.napari._output_check import output_allowed, output_problem


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def warned(monkeypatch):
    """The messages shown in the viewer."""
    messages = []
    monkeypatch.setattr(_output_check, "show_warning", messages.append)
    return messages


def _store(path, value=1):
    mm.ArrayManager(
        np.full((2, 4, 8, 8), value, dtype="uint16"),
        {"axes": "czyx", "channel_names": ["a", "b"], "scales": [(1.0, 1.0, 1.0)]},
    ).to_zarr(str(path), progress=False)
    return path


def _plain_folder(path):
    path.mkdir()
    (path / "keep.txt").write_text("precious")
    return path


# --- the check ---

@pytest.mark.parametrize("name", ["folder", "out.tif", "out.zarr.bak", ""])
@pytest.mark.parametrize("overwrite", [False, True])
def test_a_name_without_zarr_is_refused(tmp_path, name, overwrite):
    assert "must end in .zarr" in output_problem(tmp_path / name, overwrite=overwrite)


@pytest.mark.parametrize("path", [None, ""])
def test_no_path_is_refused(path):
    assert "must end in .zarr" in output_problem(path)


def test_the_name_is_checked_first(tmp_path):
    assert "must end in .zarr" in output_problem(_plain_folder(tmp_path / "folder"))


@pytest.mark.parametrize("name", ["new.zarr", "new.ome.zarr"])
def test_a_new_output_is_allowed_silently(tmp_path, warned, name):
    assert output_allowed(tmp_path / name)
    assert warned == []


def test_an_empty_folder_is_allowed_silently(tmp_path, warned):
    (tmp_path / "empty.zarr").mkdir()
    assert output_allowed(tmp_path / "empty.zarr")
    assert warned == []


def test_content_is_refused_with_a_warning(tmp_path, warned):
    store = _store(tmp_path / "a.zarr")
    assert not output_allowed(store)
    assert len(warned) == 1 and "Overwrite output" in warned[0]


def test_overwrite_replaces_a_store_with_a_warning(tmp_path, warned):
    assert output_allowed(_store(tmp_path / "a.zarr"), overwrite=True)
    assert len(warned) == 1 and "will be deleted" in warned[0]


@pytest.mark.parametrize("overwrite", [False, True])
def test_a_folder_named_zarr_that_is_no_store_is_refused(tmp_path, warned, overwrite):
    assert not output_allowed(_plain_folder(tmp_path / "a.zarr"), overwrite=overwrite)
    assert len(warned) == 1


def test_the_warning_is_kept_in_the_log(tmp_path, warned, capsys):
    output_allowed(tmp_path / "folder")
    assert capsys.readouterr().out.startswith("WARNING: Not writing")


# --- converter ---

@pytest.fixture
def converter(app, monkeypatch, warned, tmp_path):
    """The converter with a dataset loaded; conversions are recorded, not run."""
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(convert_module, "current_viewer", lambda: ViewerModel())
    runs = []

    def worker(**kwargs):
        runs.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(convert_module, "convert_worker", worker)
    dock = convert_module.convert_widget()
    source = _store(tmp_path / "source.zarr", 5)
    dock.visualize_widget.file_format.value = "zarr"
    dock.visualize_widget.input_path.value = source
    yield SimpleNamespace(dock=dock, widget=dock.convert_widget, runs=runs, source=source,
                          batch=dock.batch_csv_widget)
    dock.close()


def test_loading_proposes_an_output_that_can_be_written(converter):
    assert output_problem(converter.widget.output_path.value) is None


def test_converter_refuses_a_folder_without_zarr_name(converter, tmp_path, warned):
    folder = _plain_folder(tmp_path / "folder")
    converter.widget.output_path.value = folder
    converter.widget()
    assert converter.runs == [] and "must end in .zarr" in warned[0]
    assert (folder / "keep.txt").exists()


def test_converter_refuses_content_until_overwrite_is_ticked(converter, tmp_path, warned):
    out = _store(tmp_path / "out.zarr", 2)
    converter.widget.output_path.value = out
    converter.widget()
    assert converter.runs == [] and "Overwrite output" in warned[0]
    converter.widget.overwrite.value = True
    converter.widget()
    assert len(converter.runs) == 1 and converter.runs[0]["overwrite"] is True
    assert "will be deleted" in warned[1]


def test_converter_writes_a_new_output_without_overwriting(converter, tmp_path):
    converter.widget.output_path.value = tmp_path / "new.ome.zarr"
    converter.widget()
    assert len(converter.runs) == 1 and converter.runs[0]["overwrite"] is False


@pytest.mark.parametrize("overwrite", [False, True])
def test_converter_cannot_replace_its_input(converter, warned, overwrite):
    converter.widget.output_path.value = converter.source
    converter.widget.overwrite.value = overwrite
    converter.widget()
    assert converter.runs == [] and "input" in warned[0]


def test_loading_another_dataset_unticks_overwrite(converter, tmp_path):
    converter.widget.overwrite.value = True
    converter.dock.visualize_widget.input_path.value = _store(tmp_path / "other.zarr")
    assert converter.widget.overwrite.value is False


def test_batch_row_is_not_appended_for_a_refused_output(converter, tmp_path, warned):
    csv = tmp_path / "batch.csv"
    converter.batch.batch_csv_path.value = csv
    converter.widget.output_path.value = tmp_path / "folder"
    converter.batch()
    assert not csv.exists() and "must end in .zarr" in warned[0]

    converter.widget.output_path.value = _store(tmp_path / "out.zarr")
    converter.batch()
    assert not csv.exists()

    converter.widget.overwrite.value = True
    converter.batch()
    assert pd.read_csv(csv, dtype=str).iloc[0]["overwrite"] == "true"


def test_conversion_passes_overwrite_on_to_the_writer(tmp_path):
    source, out = _store(tmp_path / "in.zarr", 1), _store(tmp_path / "out.zarr", 2)
    kwargs = dict(
        reader=mm.ZarrManager, path=source, scene_index=None, chunks=(1, 1, 2, 4, 4), t_range=(0, 0), z_range=(0, 3),
        y_range=(0, 7), x_range=(0, 7), channels=("a", "b"), n_levels=1, downscale_factor=(2, 2, 2), output_path=out,
        file_format="zarr", zarr_format=3,
    )
    with pytest.raises(FileExistsError):
        convert_module._run_conversion(**kwargs)
    assert int(mm.ZarrManager(str(out)).data[0].max().compute()) == 2
    with pytest.warns(UserWarning, match="Replacing"):
        convert_module._run_conversion(**kwargs, overwrite=True)
    assert int(mm.ZarrManager(str(out)).data[0].max().compute()) == 1


# --- mosaic fusion ---

@pytest.fixture
def mosaic_dock(app, monkeypatch, warned, tmp_path):
    from zarrsmith.mosaic_manager import LuxendoMosaicManager

    from .test_luxendo_mosaic import _write_tile

    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    runs = []

    def worker(tiles, out, kwargs):
        runs.append((out, kwargs))
        return MagicMock()

    monkeypatch.setattr(mw, "_fuse_worker", worker)
    acq = tmp_path / "acq"
    for stack, side, tx in (("1-x00-y00", "L", 0.0), ("2-x00-y00", "R", 20.0)):
        _write_tile(acq / "raw" / f"stack_{stack}_GFP-{side}", "Cam_bottom_00000", f"GFP-{side}", stack,
                    tx, 0.0, 30.0, shape=(8, 16, 16), value=5)
    dock = mw.mosaic_widget()
    mosaic = LuxendoMosaicManager(acq, progress=False)
    dock.state["mosaic"] = mosaic
    dock.populate(mosaic, acq)
    yield SimpleNamespace(dock=dock, widget=dock.fuse_widget, runs=runs, acq=acq, mosaic=mosaic)
    dock.close()


def test_fusion_refuses_a_folder_without_zarr_name(mosaic_dock, warned):
    mosaic_dock.widget.output_path.value = mosaic_dock.acq  # the acquisition itself
    mosaic_dock.widget()
    assert mosaic_dock.runs == [] and "must end in .zarr" in warned[0]


@pytest.mark.parametrize("overwrite", [False, True])
def test_fusion_refuses_a_folder_named_zarr_that_is_no_store(mosaic_dock, tmp_path, warned, overwrite):
    folder = _plain_folder(tmp_path / "folder.zarr")
    mosaic_dock.widget.output_path.value = folder
    mosaic_dock.widget.overwrite.value = overwrite
    mosaic_dock.widget()
    assert mosaic_dock.runs == [] and (folder / "keep.txt").exists()


def test_fusion_refuses_content_until_overwrite_is_ticked(mosaic_dock, tmp_path, warned):
    out = _store(tmp_path / "out.ome.zarr")
    mosaic_dock.widget.output_path.value = out
    mosaic_dock.widget()
    assert mosaic_dock.runs == [] and "Overwrite output" in warned[0]
    mosaic_dock.widget.overwrite.value = True
    mosaic_dock.widget()
    assert len(mosaic_dock.runs) == 1 and mosaic_dock.runs[0][1]["overwrite"] is True


def test_fusion_default_output_can_be_written(mosaic_dock, warned):
    mosaic_dock.widget()
    assert len(mosaic_dock.runs) == 1 and warned == []


def test_loading_another_mosaic_unticks_overwrite(mosaic_dock):
    mosaic_dock.widget.overwrite.value = True
    mosaic_dock.dock.populate(mosaic_dock.mosaic, mosaic_dock.acq)
    assert mosaic_dock.widget.overwrite.value is False


def test_fusion_batch_row_is_not_appended_for_a_refused_output(mosaic_dock, tmp_path, warned):
    csv = tmp_path / "batch.csv"
    mosaic_dock.dock.batch_widget.batch_csv_path.value = csv
    mosaic_dock.widget.output_path.value = tmp_path / "folder"
    mosaic_dock.dock.batch_widget()
    assert not csv.exists() and "must end in .zarr" in warned[0]


# --- export ---

class _Layer:
    def __init__(self, name, path):
        self.name, self._source = name, SimpleNamespace(path=str(path))
        self.source = self._source

    def __str__(self):
        return self.name


def test_export_names_keep_their_zarr_form(tmp_path):
    channels, merged, report, composite = export_module.export_files(
        [_Layer("a", tmp_path / "s.ome.zarr"), _Layer("b", tmp_path / "s.ome.zarr")], "V")
    assert channels == ["s.ome-Org-V_a.png", "s.ome-Org-V_b.png"]
    assert (merged, report, composite) == ("s.ome-Org-V_merged.png", "s.ome-Org-V.txt", "s.ome-Org-V_composite.tif")


def test_export_never_writes_over_a_source_without_zarr_in_its_name(tmp_path):
    # the '.zarr' of the name used to be what was replaced: all the files of a .tif were the .tif itself
    channels, *others = export_module.export_files([_Layer("a", tmp_path / "s.tif")], "V")
    assert "s.tif" not in channels + others
    assert channels == ["s-Org-V_a.png"] and others[-1] == "s-Org-V_composite.tif"


def _export_dock(app, monkeypatch, tmp_path):
    source = tmp_path / "s.zarr"
    screenshot = MagicMock(side_effect=AssertionError("nothing must be exported"))
    viewer = SimpleNamespace(layers=[_Layer("a", source)], screenshot=screenshot)
    monkeypatch.setattr(export_module, "current_viewer", lambda: viewer)
    monkeypatch.chdir(tmp_path)  # the widget changes the working directory
    dock = export_module.export_widget()
    return dock, screenshot


def test_export_refuses_existing_files(app, monkeypatch, tmp_path, warned):
    (tmp_path / "s-Org-View-1.txt").write_text("previous report")
    dock, screenshot = _export_dock(app, monkeypatch, tmp_path)
    dock.export_widget()
    screenshot.assert_not_called()
    assert "s-Org-View-1.txt" in warned[0] and "Overwrite files" in warned[0]
    assert (tmp_path / "s-Org-View-1.txt").read_text() == "previous report"


def test_export_with_overwrite_goes_on_with_a_warning(app, monkeypatch, tmp_path, warned):
    (tmp_path / "s-Org-View-1.txt").write_text("previous report")
    dock, screenshot = _export_dock(app, monkeypatch, tmp_path)
    dock.export_widget.overwrite.value = True
    monkeypatch.setattr(export_module, "get_channel_adjustment", MagicMock(side_effect=InterruptedError))
    with pytest.raises(InterruptedError):  # past the check; the export itself needs a real canvas
        dock.export_widget()
    assert "Replacing" in warned[0]
