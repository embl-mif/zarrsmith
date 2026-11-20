"""The converter's ROI / Zrange / CropBox layers sit on the image when the dataset has a
stage position (NGFF translation), so a ROI drawn on the image crops those voxels."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("napari")

from napari.components import ViewerModel
from napari.layers import Image
from qtpy.QtWidgets import QApplication

import zarrsmith.microscope_manager as mm
import zarrsmith.napari._convert_widget as convert_module

POSITION = {"z": 9061.88, "y": -65812.96, "x": -100962.16}  # an Evident stage position (um)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def loaded(app, monkeypatch, tmp_path):
    """The converter with a dataset at a stage position loaded and shown."""
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    viewer = ViewerModel()
    monkeypatch.setattr(convert_module, "current_viewer", lambda: viewer)
    dataset = mm.ArrayManager(np.zeros((2, 6, 64, 80), np.uint16),
                              {"axes": "czyx", "scales": [(0.96, 0.26, 0.26)], "translations": POSITION})
    monkeypatch.setitem(convert_module.READERS, "viventis", lambda path: dataset)
    dock = convert_module.convert_widget()
    dock.visualize_widget.file_format.value = "viventis"
    dock.visualize_widget.input_path.value = tmp_path
    dock.visualize_widget()  # "Visualize in napari"
    yield viewer, dock
    dock.close()


def test_helper_layers_take_the_image_position(loaded):
    viewer, _ = loaded
    image = next(layer for layer in viewer.layers if isinstance(layer, Image))
    assert tuple(image.translate[-2:]) == pytest.approx((POSITION["y"], POSITION["x"]))
    assert tuple(viewer.layers["ROI"].translate) == pytest.approx(tuple(image.translate[-2:]))
    assert tuple(viewer.layers["Zrange"].translate) == pytest.approx(tuple(image.translate[-3:]))
    assert tuple(viewer.layers["CropBox"].translate) == pytest.approx(tuple(image.translate[-3:]))


def test_roi_drawn_on_the_image_crops_its_voxels(loaded):
    viewer, dock = loaded
    image = next(layer for layer in viewer.layers if isinstance(layer, Image))
    roi = viewer.layers["ROI"]
    # the user draws over voxels y 10..40, x 20..50 of the image, in world coordinates
    corners = [(3, 10, 20), (3, 10, 50), (3, 40, 50), (3, 40, 20)]
    world = [np.asarray(image.data_to_world(c))[-2:] for c in corners]
    roi.data = [np.array([roi.world_to_data(w) for w in world])]
    assert dock.convert_widget.y_range.value == (10, 40)
    assert dock.convert_widget.x_range.value == (20, 50)
