"""The converter's ROI can be drawn, then dragged at its size; the batch CSV field starts empty."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("napari")

from napari.components import ViewerModel
from napari.layers.base._base_constants import ActionType
from qtpy.QtWidgets import QApplication

import zarrsmith.microscope_manager as mm
import zarrsmith.napari._convert_widget as convert_module
from zarrsmith.napari._convert_widget import _fit_span


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def loaded(app, monkeypatch, tmp_path):
    """The converter with a 64 x 80 (y, x) dataset loaded and shown."""
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    viewer = ViewerModel()
    monkeypatch.setattr(convert_module, "current_viewer", lambda: viewer)
    dataset = mm.ArrayManager(np.zeros((2, 6, 64, 80), np.uint16), {"axes": "czyx", "scales": [(1.0, 1.0, 1.0)]})
    monkeypatch.setitem(convert_module.READERS, "viventis", lambda path: dataset)
    dock = convert_module.convert_widget()
    dock.visualize_widget.file_format.value = "viventis"
    dock.visualize_widget.input_path.value = tmp_path
    dock.visualize_widget()  # "Visualize in napari"
    yield viewer, dock, app
    dock.close()


def _rect(y0, x0, y1, x1):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def _draw(roi, rect):
    """What napari does when a rectangle is drawn: ``adding`` on press, ``added`` on release."""
    roi.add(rect, shape_type="rectangle", gui=True)
    roi.events.data(value=roi.data, action=ActionType.ADDED, data_indices=(-1,), vertex_indices=((),))


def _ranges(dock):
    return dock.convert_widget.y_range.value, dock.convert_widget.x_range.value


def test_fit_span_clips_a_drawn_roi_and_shifts_a_moved_one():
    assert _fit_span(-5.2, 20.4, 63) == (0, 20)
    assert _fit_span(50, 70, 63) == (50, 63)
    assert _fit_span(50, 70, 63, size=20) == (43, 63)  # moved past the edge: shifted back
    assert _fit_span(-4, 16, 63, size=20) == (0, 20)
    assert _fit_span(-4, 100, 63, size=104) == (0, 63)  # larger than the image: clipped


def test_a_drawn_roi_replaces_the_old_one_and_can_then_be_dragged(loaded):
    viewer, dock, app = loaded
    roi = viewer.layers["ROI"]
    assert roi.mode == "add_rectangle"

    _draw(roi, _rect(10, 20, 30.4, 50.2))
    assert len(roi.data) == 1
    assert _ranges(dock) == ((10, 30), (20, 50))
    app.processEvents()
    assert roi.mode == "select" and roi.selected_data == {0}

    # Dragging moves the rectangle; it keeps its size, also when pushed past the edge.
    roi.data = [_rect(25.3, 40.3, 45.3, 70.3)]
    assert _ranges(dock) == ((25, 45), (40, 70))
    roi.data = [_rect(50, 70, 70, 100)]
    assert _ranges(dock) == ((43, 63), (49, 79))
    np.testing.assert_array_equal(roi.data[0], _rect(43, 49, 63, 79))


def test_a_drag_in_progress_is_left_alone(loaded):
    viewer, dock, _ = loaded
    roi = viewer.layers["ROI"]
    before = _ranges(dock)
    roi.add(_rect(1, 2, 3, 4), shape_type="rectangle", gui=True)  # press of a new drawing
    roi.events.data(value=roi.data, action=ActionType.CHANGING, data_indices=(0,), vertex_indices=((),))
    assert len(roi.data) == 2 and _ranges(dock) == before


def test_batch_csv_field_starts_empty_and_is_not_filled_on_load(loaded, tmp_path, capsys):
    viewer, dock, _ = loaded
    field = dock.batch_csv_widget.batch_csv_path
    assert field.value is None
    dock.visualize_widget.input_path.value = tmp_path / "other"  # loading another dataset
    assert field.value is None

    dock.convert_widget.output_path.value = tmp_path / "out.zarr"
    dock.batch_csv_widget()
    assert "choose a batch CSV file first" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.csv"))

    field.value = tmp_path / "runs" / "my_batch"  # typed in the save dialog without a suffix
    dock.batch_csv_widget()
    assert (tmp_path / "runs" / "my_batch.csv").exists()
