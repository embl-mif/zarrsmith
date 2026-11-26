import re
import sys
from pathlib import Path

import numpy as np
from magicgui import magicgui
from magicgui.widgets import FileEdit
from napari import current_viewer
from napari.layers.utils import layer_utils
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton

from zarrsmith.microscope_manager.readers import READERS
from zarrsmith.utils.axes import recorded_voxel_size, spatial_axes_in_order
from zarrsmith.utils.chunks import default_num_levels, output_summary_line
from ._batch_csv import append_batch_csv, same_path, split_zarr_name
from ._layout import compact_row, compose_dock, enable_drag_and_drop, make_button_holder, make_log_pane, make_title
from ._layout import make_section as _make_section
from ._dataset_helpers import axis_index as _axis_index
from ._dataset_helpers import axis_size as _axis_size
from ._dataset_helpers import dataset_axes as _dataset_axes
from ._dataset_helpers import scale_for_axes as _scale_for_axes
from ._dataset_helpers import points_projection_kwargs
from ._dataset_helpers import units_kwargs as _units_kwargs
from zarrsmith.utils.output_path import has_zarr_name
from ._output_check import confirm_batch_output, output_allowed
from ._zarr_options import (DEFAULT_CHUNK_MB, DEFAULT_SHARD_MB, NGFF_FORMATS, PYRAMID_METHODS, auto_channel_names,
                            channel_error, channel_option_problems, channel_warning_text, chunk_shard_layout,
                            follow_auto_value, layout_summary, parse_channel_colors, parse_channel_names)

def conservative_compute_multiscale_level(
    requested_shape,
    shape_threshold,
    downsample_factors,
):
    """Computed desired level of the multiscale given requested field of view.

    The level of the multiscale should be the lowest resolution such that
    the requested shape is above the shape threshold. By passing a shape
    threshold corresponding to the shape of the canvas on the screen this
    ensures that we have at least one data pixel per screen pixel, but no
    more than we need.

    Parameters
    ----------
    requested_shape : tuple
        Requested shape of field of view in data coordinates
    shape_threshold : tuple
        Maximum size of a displayed tile in pixels.
    downsample_factors : list of tuple
        Downsampling factors for each level of the multiscale. Must be increasing
        for each level of the multiscale.

    Returns
    -------
    level : int
        Level of the multiscale to be viewing.
    """
    # Scale shape by downsample factors
    threshold_factor = 0.1  # Adjust this factor to control the conservativeness
    scaled_shape = requested_shape / downsample_factors

    # Find the highest level (lowest resolution) allowed
    locations = np.argwhere(np.any(scaled_shape > (np.asarray(shape_threshold) * threshold_factor), axis=1))
    return locations[-1][0] if len(locations) > 0 else 0


layer_utils.compute_multiscale_level = conservative_compute_multiscale_level

# -------------------------
# MagicGUI controls
# -------------------------

from ._log import EmittingStream, make_log_appender  # noqa: F401  (re-exported)


def dataset_reader(microscope):
    """Return the ZarrSmith manager class matching the selected microscope key."""
    if microscope not in READERS:
        raise ValueError(f"Unsupported microscope: {microscope}")
    return READERS[microscope]


def _chunk_shape_for_dataset(dataset, chunk_z=16, chunk_y=512, chunk_x=512):
    """Map widget Z/Y/X chunk values onto whatever axes the dataset has."""
    chunk_map = {"t": 1, "c": 1, "z": chunk_z, "y": chunk_y, "x": chunk_x}
    axes = _dataset_axes(dataset)
    return tuple(int(max(1, min(_axis_size(dataset, ax), chunk_map[ax]))) for ax in axes)


def _present_range(dataset, axis, requested_range):
    """Return a clipped inclusive range tuple for a present axis, else None."""
    if _axis_index(dataset, axis) is None:
        return None
    max_idx = _axis_size(dataset, axis) - 1
    start, end = requested_range
    return int(max(0, min(start, max_idx))), int(max(0, min(end, max_idx)))


def _scene_tag(dataset, scene_index):
    """``_TR1`` for a scene of a file with several, so that their outputs differ; else nothing."""
    scenes = getattr(dataset, "scenes", ())
    if len(scenes) < 2:
        return ""
    return "_" + re.sub(r"[^A-Za-z0-9._-]+", "_", str(scenes[scene_index]))


def _span_tag(indices):
    """``0-4`` for consecutive indices, else ``0_2_5``."""
    indices = [int(i) for i in indices]
    if len(indices) > 1 and indices == list(range(indices[0], indices[-1] + 1)):
        return f"{indices[0]}-{indices[-1]}"
    return "_".join(str(i) for i in indices)


def _region_tag(dataset, *, t_range, z_range, y_range, x_range, channels) -> str:
    """``_z10-50_y100-611_x200-711`` for the axes the conversion crops, ``""`` for the full dataset.

    Only cropped axes are named (inclusive voxel indices, as in the widget), and the
    channels only when a subset is selected, in the dataset's axis order.
    """
    ranges = {"t": t_range, "z": z_range, "y": y_range, "x": x_range}
    parts = []
    for ax in _dataset_axes(dataset):
        if ax == "c":
            names = list(dataset.metadata.get("channel_names") or [])
            selected = [names.index(ch) for ch in channels if ch in names]
            if selected and sorted(selected) != list(range(len(names))):
                parts.append(f"c{_span_tag(selected)}")
            continue
        start, end = sorted(_present_range(dataset, ax, ranges[ax]))
        if (start, end) != (0, _axis_size(dataset, ax) - 1):
            parts.append(f"{ax}{start}-{end}")
    return "".join(f"_{p}" for p in parts)


def _set_range_widget(widget, size, enabled=True):
    max_idx = max(0, int(size) - 1)
    widget.max = max_idx
    widget.value = (0, max_idx)
    widget.enabled = bool(enabled and size > 1)

BATCH_CSV_COLUMNS = [
    "input",
    "microscope",
    "output",
    "chunk_size",
    "max_size(MB)",
    "scene_index",
    "zarr_format",
    "downscale_factor",
    "subset",
    "channel_colors",
    "channel_names",
    "num_levels",
    "method",
    "shards",
    "shard_target_mb",
    "shard_exclude_axes",
    "drop_singleton",
    "num_workers",
    "overwrite",
]


def _indices_to_expr(indices):
    indices = [int(i) for i in indices]
    if not indices:
        return ""
    if len(indices) == 1:
        return str(indices[0])
    step = indices[1] - indices[0]
    if step > 0 and all((b - a) == step for a, b in zip(indices[:-1], indices[1:])):
        start = indices[0]
        stop = indices[-1]+1
        return f"{start}:{stop}:{step}"
    return ",".join(str(i) for i in indices)


def _range_to_indices(range_value):
    start, end = (int(range_value[0]), int(range_value[1]))
    if end < start:
        start, end = end, start
    return list(range(start, end + 1))


def _subset_string_from_widget(dataset, t_range, z_range, y_range, x_range, channels):
    axes = _dataset_axes(dataset)
    parts = []
    if "t" in axes:
        parts.append(f"t={_indices_to_expr(_range_to_indices(t_range))}")
    if "c" in axes and channels:
        channel_names = list(dataset.metadata.get("channel_names", []))
        indices = [channel_names.index(name) for name in channels if name in channel_names]
        if indices:
            parts.append(f"c={_indices_to_expr(indices)}")
    if "z" in axes:
        parts.append(f"z={_indices_to_expr(_range_to_indices(z_range))}")
    if "y" in axes:
        parts.append(f"y={_indices_to_expr(_range_to_indices(y_range))}")
    if "x" in axes:
        parts.append(f"x={_indices_to_expr(_range_to_indices(x_range))}")
    return ";".join(parts)


def _channel_colors_from_metadata(dataset, selected_channel_names):
    channel_names = list(dataset.metadata.get("channel_names", []))
    colors = list(dataset.metadata.get("channel_colors", []))
    if not channel_names or not colors or len(channel_names) != len(colors):
        return []
    lookup = dict(zip(channel_names, colors))
    return [str(lookup[name]) for name in selected_channel_names if name in lookup]


def output_sizes(dataset, *, t_range, z_range, y_range, x_range, channels) -> dict:
    """``{axis: size}`` of the converted level 0 (every dataset axis, singletons included),
    following the selected ranges and channels."""
    ranges = {"t": t_range, "z": z_range, "y": y_range, "x": x_range}
    sizes = {}
    for ax in _dataset_axes(dataset):
        if ax == "c":
            names = dataset.metadata.get("channel_names", [])
            selected = [ch for ch in channels if ch in names]
            sizes[ax] = len(selected) if selected else _axis_size(dataset, "c")
        else:
            start, end = _present_range(dataset, ax, ranges[ax])
            sizes[ax] = abs(end - start) + 1
    return sizes


def output_channels(dataset, channels) -> list:
    """Names of the converted channels: the selected ones, all of them when none is selected,
    and none for a dataset without a channel axis."""
    if "c" not in _dataset_axes(dataset):
        return []
    names = list(dataset.metadata.get("channel_names", []))
    selected = [ch for ch in channels if ch in names]  # in the order _run_conversion subsets them
    return selected or names


def output_summary(dataset, *, t_range, z_range, y_range, x_range, channels, drop_singleton=True):
    """Level-0 recap of the converted dataset, in the format of the mosaic fusion plan.

    One line such as ``3 t x 2 ch x 120 x 1024 x 1024 px (tczyx), 1.51 GB,
    voxel 2 x 0.65 x 0.65 micrometer``, following the selected ranges and
    channels and, with ``drop_singleton``, the t/c/z axes dropped on write.
    """
    axes = _dataset_axes(dataset)
    sizes = output_sizes(dataset, t_range=t_range, z_range=z_range, y_range=y_range, x_range=x_range,
                         channels=channels)
    spatial = spatial_axes_in_order(axes)
    # the line zarrsmith convert --dry_run prints
    return output_summary_line(axes, sizes, dataset.data[0].dtype, _scale_for_axes(dataset, spatial),
                               dataset.metadata.get("units") or (), drop_singleton=drop_singleton)


def _fit_span(lo, hi, vmax, size=None):
    """Integer span ``(lo, hi)`` of a ROI side, inside ``[0, vmax]``.

    With ``size`` (the ROI was moved, so it keeps its size), a span reaching past an
    edge is shifted back inside rather than cut. Without it (drawn or resized), the
    span is clipped to the image.
    """
    lo, hi = sorted((int(np.floor(lo + 0.5)), int(np.floor(hi + 0.5))))
    if size is not None and 0 <= size <= vmax:
        lo = max(0, min(lo, vmax - size))
        return lo, lo + size
    return max(0, min(lo, vmax)), max(0, min(hi, vmax))


def _append_batch_csv(csv_path, row):
    return append_batch_csv(csv_path, row, BATCH_CSV_COLUMNS)

def _compact_advanced_rows(widget, insert_after):
    """Regroup the "Additional parameters" of the converter into compact rows.

    Related controls (chunks, downscale, ...) share one row instead of one row
    each. The magicgui widgets are moved, not copied, so ``widget.chunk_x`` and
    friends keep working (values, ``enabled``, signals). Their original
    one-per-row wrappers are hidden for good.

    Returns the new row widgets; show or hide them together as one group. Controls the
    panel does not have are left out, and so are rows left without any.
    """
    layout = widget.native.layout()
    present = {w.name for w in widget}

    def row(label, *items):
        items = [(prefix, name) for prefix, name in items if name in present]
        return compact_row(widget, label, *items) if items else None

    rows = [
        row("Chunks", (None, "chunk_auto"), ("MB", "chunk_mb")),
        row("Chunk shape", ("Z", "chunk_z"), ("Y", "chunk_y"), ("X", "chunk_x")),
        row("Downscale", ("Z", "downscale_z"), ("Y", "downscale_y"), ("X", "downscale_x")),
        row("Output", ("Levels", "n_levels"), ("Method", "method"), ("NGFF", "zarr_format")),
        row("Sharding", (None, "shards"), ("MB", "shard_target_mb")),
        row("Never merge", (None, "shard_exclude_axes")),
        row("Other", (None, "drop_singleton"), ("Threads", "num_workers")),
        row("Ch. names", (None, "channel_names")),
        row("Ch. colors", (None, "channel_colors")),
    ]
    rows = [r for r in rows if r is not None]
    widget.shard_exclude_axes.native.setMaximumHeight(60)

    index = layout.indexOf(insert_after) + 1
    for offset, row_widget in enumerate(rows):
        layout.insertWidget(index + offset, row_widget)
    return rows


def _run_conversion(
    reader,
    path,
    scene_index,
    chunks,
    t_range,
    z_range,
    y_range,
    x_range,
    channels,
    n_levels,
    downscale_factor,
    output_path,
    file_format,
    zarr_format,
    shards="auto",
    shard_target_mb=None,
    shard_exclude_axes=None,
    drop_singleton=True,
    num_workers=None,
    overwrite=False,
    method=None,
    channel_names=None,
    channel_colors=None,
    ):
    """Run the conversion pipeline used by the napari worker thread.

    ``method`` is the downsampling of the pyramid levels (``None``: mean for intensity
    data, nearest for labels). ``channel_names`` and ``channel_colors`` (one per converted
    channel, or ``None``/empty to keep the dataset's) are written to the output metadata."""
    print("Starting conversion in background thread...")

    if file_format in ("zeiss", "evident"):
        dataset = reader(path, scene_index=scene_index, chunks=chunks)
    else:
        dataset = reader(path, chunks=chunks)

    axes = _dataset_axes(dataset)
    print("Dataset axes:", axes)

    if len(channels) == 0 or "c" not in axes:
        channels_index = None
    else:
        channels_index = [
            dataset.metadata["channel_names"].index(ch)
            for ch in channels
            if ch in dataset.metadata.get("channel_names", [])
        ]

    subset_kwargs = {}
    tr = _present_range(dataset, "t", t_range)
    zr = _present_range(dataset, "z", z_range)
    yr = _present_range(dataset, "y", y_range)
    xr = _present_range(dataset, "x", x_range)

    if tr is not None and _axis_size(dataset, "t") > 1:
        subset_kwargs["T"] = list(range(tr[0], tr[1] + 1))
    if channels_index is not None and _axis_size(dataset, "c") > 1:
        subset_kwargs["C"] = channels_index
    if zr is not None and _axis_size(dataset, "z") > 1:
        subset_kwargs["Z"] = list(range(zr[0], zr[1] + 1))
    if yr is not None:
        subset_kwargs["Y"] = list(range(yr[0], yr[1] + 1))
    if xr is not None:
        subset_kwargs["X"] = list(range(xr[0], xr[1] + 1))

    if subset_kwargs:
        dataset.subset_dataset(**subset_kwargs, rebuild_pyramid=False)

    # after the subset: one name and colour per converted channel
    if channel_names:
        dataset.metadata["channel_names"] = list(channel_names)
        print("Channel names:", list(channel_names))
    if channel_colors:
        dataset.metadata["channel_colors"] = list(channel_colors)
        print("Channel colors:", list(channel_colors))

    chunks = _chunk_shape_for_dataset(
        dataset,
        chunk_z=chunks[2] if len(chunks) > 2 else 1,
        chunk_y=chunks[3] if len(chunks) > 3 else 512,
        chunk_x=chunks[4] if len(chunks) > 4 else 512,
    )

    print("Requested input chunks:", chunks)
    print("Chunks before pyramid:", [arr.chunksize for arr in dataset.data])

    dataset.build_pyramid(num_levels=n_levels, downscale_factor=downscale_factor, method=method)

    dataset.data = [
        arr.rechunk(chunks) if arr.ndim == len(chunks) else arr
        for arr in dataset.data
    ]
    dataset.metadata["chunksize"] = [tuple(arr.chunksize) for arr in dataset.data]

    print("Chunks after pyramid:", [arr.chunksize for arr in dataset.data])

    ngff_version = "0.4" if zarr_format == 2 else "0.5"
    to_zarr_kwargs = {
        "zarr_format": zarr_format,
        "ngff_version": ngff_version,
        "drop_singleton": bool(drop_singleton),
        "shards": shards,
        "overwrite": bool(overwrite),
        # given to the writer too: a ZarrManager writes its source levels, not the rechunked ones
        "chunks": chunks,
    }
    if num_workers is not None:
        to_zarr_kwargs["num_workers"] = int(num_workers)
    if shards is not None:
        if shard_target_mb is not None:
            to_zarr_kwargs["shard_target_mb"] = shard_target_mb
        if shard_exclude_axes is not None:
            to_zarr_kwargs["shard_exclude_axes"] = shard_exclude_axes

    dataset.to_zarr(output_path, **to_zarr_kwargs)

    return output_path

@thread_worker
def convert_worker(**kwargs):
    """Background worker wrapper used by the napari conversion widget."""
    return _run_conversion(**kwargs)


def convert_widget():
    """Create the main ZarrSmith napari conversion widget."""
    viewer = current_viewer()
    
    def lock_roi_in_3d(event=None):
        if "ROI" not in viewer.layers:
            return
        if viewer.dims.ndisplay == 3:
            viewer.layers["ROI"].editable = False
            viewer.layers["ROI"].selectable = False
            make_convert_widget.y_range.enabled = False
            make_convert_widget.x_range.enabled = False
        else:
            viewer.layers["ROI"].editable = True
            viewer.layers["ROI"].selectable = True
            dataset = _state.get("dataset")
            make_convert_widget.y_range.enabled = bool(dataset and _axis_index(dataset, "y") is not None)
            make_convert_widget.x_range.enabled = bool(dataset and _axis_index(dataset, "x") is not None)

    def image_offset(n):
        """``translate`` of the last ``n`` (spatial) axes of the image layers in the viewer.

        The ROI, Zrange and CropBox layers hold voxel indices of the dataset; they take
        the offset of the image layers so they sit on the image wherever it is placed
        (at its stage position by ``visualize()``, at 0 by napari-ome-zarr)."""
        from napari.layers import Image

        for layer in viewer.layers:
            if isinstance(layer, Image) and len(layer.translate) >= n:
                return tuple(float(v) for v in layer.translate[-n:])
        return (0.0,) * n

    def ensure_crop_layers(dataset):
        axes = _dataset_axes(dataset)
        if "ROI" not in viewer.layers and {"y", "x"}.issubset(axes):
            ymax = _axis_size(dataset, "y") - 1
            xmax = _axis_size(dataset, "x") - 1
            rect = np.array([
                [0, 0],
                [0, xmax],
                [ymax, xmax],
                [ymax, 0],
            ])
            layer = viewer.add_shapes(
                data=rect,
                shape_type="rectangle",
                name="ROI",
                edge_color="orange",
                face_color="transparent",
                edge_width=10,
                ndim=2,
                scale=_scale_for_axes(dataset, "yx"),
                translate=image_offset(2),
                **_units_kwargs(dataset, "yx"),
            )
            layer.mode = "add_rectangle"

        if "Zrange" not in viewer.layers and {"z", "y", "x"}.issubset(axes):
            points = np.array([
                [0, _axis_size(dataset, "y") // 2, _axis_size(dataset, "x") // 2],
                [_axis_size(dataset, "z") - 1, _axis_size(dataset, "y") // 2, _axis_size(dataset, "x") // 2],
            ])
            layer = viewer.add_points(
                data=points,
                name="Zrange",
                face_color="lime",
                size=20,
                ndim=3,
                **points_projection_kwargs(),
                scale=_scale_for_axes(dataset, "zyx"),
                translate=image_offset(3),
                **_units_kwargs(dataset, "zyx"),
            )
            layer.mode = "add"

        if "CropBox" not in viewer.layers and {"z", "y", "x"}.issubset(axes):
            layer = viewer.add_shapes(
                data=np.empty((0, 2, 3)),
                shape_type="polygon",
                name="CropBox",
                edge_color="cyan",
                face_color="transparent",
                ndim=3,
                scale=_scale_for_axes(dataset, "zyx"),
                translate=image_offset(3),
                **_units_kwargs(dataset, "zyx"),
            )
            layer.editable = False          # prevents adding/moving/editing shapes
            layer.selectable = False        # prevents selecting the shape
            layer.mode = "pan_zoom"         # ensures no shape tool is active
            layer.opacity = 0.7
            layer.edge_width = 2
            layer.blending = "additive"

    def update_3d_box():
        if any(name not in viewer.layers for name in ("ROI", "Zrange", "CropBox")):
            return
        roi_layer = viewer.layers["ROI"]
        zrange_layer = viewer.layers["Zrange"]
        if len(roi_layer.data) == 0 or len(zrange_layer.data) < 2:
            return
        zvals = [d[0] for d in zrange_layer.data]

        zrange = list(zvals[:2])

        roi = [list(r) for r in roi_layer.data[0]]

        polygon = [
            [[zrange[0]] + roi[0], [zrange[0]] + roi[1], [zrange[0]] + roi[2], [zrange[0]] + roi[3]],
            [[zrange[1]] + roi[0], [zrange[1]] + roi[1], [zrange[1]] + roi[2], [zrange[1]] + roi[3]],
            [[zrange[0]] + roi[0], [zrange[1]] + roi[0], [zrange[1]] + roi[1], [zrange[0]] + roi[1]],
            [[zrange[0]] + roi[0], [zrange[1]] + roi[0], [zrange[1]] + roi[3], [zrange[0]] + roi[3]],
            [[zrange[0]] + roi[1], [zrange[1]] + roi[1], [zrange[1]] + roi[2], [zrange[0]] + roi[2]],
            [[zrange[0]] + roi[3], [zrange[1]] + roi[3], [zrange[1]] + roi[2], [zrange[0]] + roi[2]],
        ]

        viewer.layers["CropBox"].data = polygon
        viewer.layers["CropBox"].refresh()

    def clamp(v, vmin, vmax):
        return int(max(vmin, min(v, vmax)))


    def roi_to_widget(event=None):
        """Sync the ROI layer to the Y/X ranges once a rectangle is drawn, moved or resized.

        Only one ROI is kept: a newly drawn rectangle replaces the previous one, and the
        layer then switches to select mode so the next drag moves that rectangle (its
        size is kept) or resizes it by its handles. Events sent while the mouse is still
        down (``adding``, ``changing``) are ignored, so a drag is never interrupted.
        """
        action = str(getattr(event, "action", "changed") or "changed")
        if _syncing["roi"] or action in ("adding", "changing", "removing"):
            return
        _syncing["roi"] = True

        try:
            if "ROI" not in viewer.layers:
                return
            layer = viewer.layers["ROI"]
            if layer is None or len(layer.data) == 0:
                return  # ROI deleted: the ranges stay as they are

            drawn = action == "added"
            vertices = np.asarray(layer.data[-1], dtype=float)
            dataset = _state["dataset"]
            y_max = _axis_size(dataset, "y") - 1
            x_max = _axis_size(dataset, "x") - 1

            # A move keeps the size of the ROI the widget holds; a drawn or resized one is clipped.
            spans = {}
            for i, (axis, vmax) in enumerate((("y", y_max), ("x", x_max))):
                lo, hi = float(vertices[:, i].min()), float(vertices[:, i].max())
                previous = getattr(make_convert_widget, f"{axis}_range").value
                old_size = int(previous[1]) - int(previous[0])
                new_size = int(np.floor(hi + 0.5)) - int(np.floor(lo + 0.5))
                moved = not drawn and new_size == old_size
                spans[axis] = _fit_span(lo, hi, vmax, size=old_size if moved else None)
            (y0, y1), (x0, x1) = spans["y"], spans["x"]

            rect = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
            if len(layer.data) > 1 or not np.array_equal(np.asarray(layer.data[0]), rect):
                layer.data = [rect]

            make_convert_widget.y_range.value = (y0, y1)
            make_convert_widget.x_range.value = (x0, x1)

            update_3d_box()
        finally:
            _syncing["roi"] = False

        if drawn:
            # After napari finishes the drawing: drag now moves the ROI just drawn.
            def _select_roi():
                if "ROI" in viewer.layers and viewer.layers["ROI"].editable:
                    roi = viewer.layers["ROI"]
                    roi.mode = "select"
                    roi.selected_data = {0} if len(roi.data) else set()

            QTimer.singleShot(0, _select_roi)

    def zpoints_to_widget(event=None):
        if _syncing["z"]:
            return
        _syncing["z"] = True
        
        try:
            if "Zrange" not in viewer.layers:
                return
            layer = viewer.layers["Zrange"]
            if layer is None or len(layer.data) < 2:
                return

            zs = sorted(int(p[0]) for p in layer.data[:2])

            make_convert_widget.z_range.value = (zs[0], zs[1])

            update_3d_box()
        finally:
            _syncing["z"] = False

    def widget_to_roi(event=None):
        if _syncing["roi"]:
            return
        _syncing["roi"] = True

        try:    
            if "ROI" not in viewer.layers:
                return
            layer = viewer.layers["ROI"]

            dataset = _state["dataset"]
            b = {
                "y_max": _axis_size(dataset, "y") - 1,
                "x_max": _axis_size(dataset, "x") - 1,
            }

            y0 = clamp(make_convert_widget.y_range.value[0], 0, b["y_max"])
            y1 = clamp(make_convert_widget.y_range.value[1], 0, b["y_max"])
            x0 = clamp(make_convert_widget.x_range.value[0], 0, b["x_max"])
            x1 = clamp(make_convert_widget.x_range.value[1], 0, b["x_max"])

            y0, y1 = sorted([y0, y1])
            x0, x1 = sorted([x0, x1])

            rect = np.array([
                [y0, x0],
                [y0, x1],
                [y1, x1],
                [y1, x0],
            ])

            layer.data = [rect]

            update_3d_box()
        finally:
            _syncing["roi"] = False
    
    def widget_to_zpoints(event=None):
        if _syncing["z"]:
            return
        _syncing["z"] = True
        
        try:
            if "Zrange" not in viewer.layers:
                return
            layer = viewer.layers["Zrange"]

            z0 = make_convert_widget.z_range.value[0]
            z1 = make_convert_widget.z_range.value[1]

            prev_points = layer.data

            points = np.array([
                [z0, prev_points[0,1], prev_points[0,2]],
                [z1, prev_points[1,1], prev_points[1,2]],
            ])

            layer.data = points

            update_3d_box()
        finally:
            _syncing["z"] = False
        
    def keep_two_zpoints(event=None):
        if _syncing["z"]:
            return
        _syncing["z"] = True
        
        try:
            if "Zrange" not in viewer.layers:
                return
            layer = viewer.layers["Zrange"]
            if len(layer.data) > 2:
                layer.data = layer.data[-2:]
        finally:
            _syncing["z"] = False

    _state = {"dataset": None, "auto_output": None}
    _syncing = {"roi": False, "z": False}


    @magicgui(
        call_button="Visualize in napari",
        input_path={"widget_type": "FileEdit", "mode": "d"},
        scene_index={"widget_type": "SpinBox", "min": 0, "max": 100, "step": 1},
        file_format={"choices": ["viventis", "opera", "luxendo", "zarr", "zeiss", "evident", "scape"]},
    )
    def make_visualize_widget(
        file_format="zarr",
        scene_index=0,
        input_path: FileEdit = None,
    ):
        dataset = _state["dataset"]
        if dataset is None:
            return

        viewer.layers.clear()

        if make_visualize_widget.file_format.value in ["zarr"]:
            viewer.open(make_visualize_widget.input_path.value, plugin='napari-ome-zarr')
            # napari-ome-zarr opens at the stored display window and limits the slider to it
            from zarrsmith.utils.display_window import widen_contrast_ranges

            widen_contrast_ranges(viewer.layers)
        else:
            dataset.visualize(viewer=viewer)


        # initialize ROI to full image when spatial axes are available
        ensure_crop_layers(dataset)

        widget_to_roi()
        widget_to_zpoints()
        update_3d_box()

        if "ROI" in viewer.layers:
            # Draw the ROI first; once drawn, the layer switches to select mode to drag it.
            viewer.layers["ROI"].mode = "add_rectangle"
            viewer.layers["ROI"].events.data.connect(roi_to_widget)
            make_convert_widget.y_range.changed.connect(widget_to_roi)
            make_convert_widget.x_range.changed.connect(widget_to_roi)

        if "Zrange" in viewer.layers:
            viewer.layers["Zrange"].events.data.connect(keep_two_zpoints)
            viewer.layers["Zrange"].events.data.connect(zpoints_to_widget)
            make_convert_widget.z_range.changed.connect(widget_to_zpoints)

    enable_drag_and_drop(make_visualize_widget.input_path)
    # ---

    @magicgui(
        call_button="Convert to zarr",

        chunk_auto={"label": "auto", "widget_type": "CheckBox", "value": True,
                    "tooltip": "Choose the chunk shape from the output: about the given size (uncompressed), "
                               "1 along t and c, z 8 times smaller than y and x. Untick to type it."},
        chunk_mb={"label": "Chunk MB", "widget_type": "FloatSpinBox", "min": 0.25, "max": 1024, "step": 0.25,
                  "value": DEFAULT_CHUNK_MB, "tooltip": "Target uncompressed size of a chunk (auto chunks)."},
        chunk_z={"label": "Chunk Z", "min": 1, "max": 2**16, "step": 1, "value": 32,
                 "tooltip": "Chunk shape; filled in when auto is ticked."},
        chunk_y={"label": "Chunk Y", "min": 1, "max": 2**16, "step": 1, "value": 256},
        chunk_x={"label": "Chunk X", "min": 1, "max": 2**16, "step": 1, "value": 256},

        n_levels={"label": "Resolution levels", "min": 1, "max": 10, "value": 5},
        downscale_z={"label": "Downscale Z", "min": 1, "max": 64, "step": 1, "value": 2},
        downscale_y={"label": "Downscale Y", "min": 1, "max": 64, "step": 1, "value": 2},
        downscale_x={"label": "Downscale X", "min": 1, "max": 64, "step": 1, "value": 2},
        zarr_format={"label": "NGFF", "choices": list(NGFF_FORMATS), "value": "0.5 (zarr v3)",
                     "tooltip": "OME-Zarr version: 0.5 (zarr v3, with shards) or 0.4 (zarr v2, no shards)."},
        method={"label": "Downsampling", "choices": list(PYRAMID_METHODS), "value": "auto",
                "tooltip": "auto: mean for intensity images, nearest for labels"},

        shards={"label": "Sharding (zarr v3 only)", "choices": ["none", "auto"], "value": "auto",
                "tooltip": "auto: group whole chunks into shard files of at most the given size (zarr v3 only; "
                           "every pyramid level gets its own). none: one file per chunk."},
        shard_target_mb={"label": "Shard target (MB)", "min": 1, "max": 100000, "step": 1, "value": DEFAULT_SHARD_MB,
                         "tooltip": "Target uncompressed size of a shard."},
        shard_exclude_axes={
            "label": "Never merge axes",
            "choices": ["t", "c", "z", "y", "x"],
            "widget_type": "Select",
            "allow_multiple": True,
            "value": ("t", "c"),
        },
        drop_singleton={"label": "Drop singleton axes", "widget_type": "CheckBox", "value": True},
        num_workers={"label": "Write threads (0 = auto)", "min": 0, "max": 256, "step": 1, "value": 0,
                     "tooltip": "Threads compressing and writing the output. 0: half of the cores (at most 8), "
                                "reduced so the shards being written fit in half of the free RAM."},
        channel_names={"label": "Channel names", "widget_type": "LineEdit", "value": "",
                       "tooltip": "Names of the output channels, separated by spaces, one per converted channel. "
                                  "Filled in with the selected channels (spaces replaced by underscores) "
                                  "until you type names of your own."},
        channel_colors={"label": "Channel colors", "widget_type": "LineEdit", "value": "",
                        "tooltip": "Colours of the output channels (hex codes such as FF00FF or names such as "
                                   "magenta), separated by spaces, one per converted channel. Empty: the "
                                   "colours of the dataset."},

        t_range={"label": "T range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},
        single_t = {"label": "Single T frame", "widget_type": "CheckBox", "value": False},
        z_range={"label": "Z range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},
        single_z = {"label": "Single Z plane", "widget_type": "CheckBox", "value": False},
        y_range={"label": "Y range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},
        x_range={"label": "X range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},

        channels={
            "label": "Channels",
            "choices": ["ch0", "ch1", "ch2"],
            "widget_type": "Select",
            "allow_multiple": True,
        },

        # Empty until a dataset is loaded, then named after it and the cropped region.
        output_path={"widget_type": "FileEdit", "mode": "d", "nullable": True},
        overwrite={"label": "Overwrite output", "widget_type": "CheckBox", "value": False},
    )
    def make_convert_widget(
        t_range=(0,1000),
        single_t=False,
        z_range=(0,1000),
        single_z=False,
        y_range=(0,1000),
        x_range=(0,1000),
        channels=(),
        chunk_auto=True,
        chunk_mb=DEFAULT_CHUNK_MB,
        chunk_x=256,
        chunk_y=256,
        chunk_z=32,
        n_levels=5,
        downscale_z=2,
        downscale_y=2,
        downscale_x=2,
        zarr_format="0.5 (zarr v3)",
        method="auto",
        shards="auto",
        shard_target_mb=DEFAULT_SHARD_MB,
        shard_exclude_axes=("t", "c"),
        drop_singleton=True,
        num_workers=0,
        channel_names="",
        channel_colors="",
        output_path: FileEdit = None,
        overwrite=False,
    ):

        path = make_visualize_widget.input_path.value
        if not output_allowed(output_path, overwrite=overwrite, sources=[path]):
            return
        try:
            names, colors = _channel_options()
        except ValueError as e:
            print(f"Cannot convert: {e}")
            return

        reader = dataset_reader(make_visualize_widget.file_format.value)
        scene_index = make_visualize_widget.scene_index.value
        file_format = make_visualize_widget.file_format.value

        chunk_z, chunk_y, chunk_x = _chunk_zyx()
        chunks = (1, 1, chunk_z, chunk_y, chunk_x)
        downscale_factor = (downscale_z, downscale_y, downscale_x)
        zarr_format = _zarr_format()

        shards_value = None if shards == "none" else "auto"

        worker = convert_worker(
                reader=reader,
                path=path,
                scene_index=scene_index,
                chunks=chunks,
                t_range=t_range,
                z_range=z_range,
                y_range=y_range,
                x_range=x_range,
                channels=channels,
                n_levels=n_levels,
                downscale_factor=downscale_factor,
                output_path=output_path,
                file_format=file_format,
                zarr_format=zarr_format,
                shards=shards_value,
                shard_target_mb=float(shard_target_mb) if shards_value is not None else None,
                shard_exclude_axes=tuple(shard_exclude_axes) if shards_value is not None else None,
                drop_singleton=bool(drop_singleton),
                num_workers=int(num_workers) if int(num_workers) > 0 else None,
                overwrite=bool(overwrite),
                method=None if method == "auto" else method,
                channel_names=names or None,
                channel_colors=colors or None,
            )
        
        _set_convert_enabled(False)
        make_batch_csv_widget.enabled = False
        make_visualize_widget.enabled = False

        @worker.returned.connect
        def _on_done(result_path):
            print(f"Conversion completed: {result_path}")
            _set_convert_enabled(True)
            make_visualize_widget.enabled = True

        @worker.errored.connect
        def _on_error(err):
            print("Conversion failed:", err)
            _set_convert_enabled(True)
            make_visualize_widget.enabled = True

        worker.start()

    ui = {}  # widgets created later, when the dock is composed

    output_label = QLabel("Load a dataset to see the output size.")
    output_label.setWordWrap(True)
    output_label.setStyleSheet("QLabel { color: #aaaaaa; }")

    channel_warning = QLabel()
    channel_warning.setWordWrap(True)
    channel_warning.setStyleSheet("QLabel { color: #e8a33d; }")
    channel_warning.setVisible(False)

    def _zarr_format() -> int:
        return NGFF_FORMATS[make_convert_widget.zarr_format.value]

    def _ranges() -> dict:
        w = make_convert_widget
        return {"t_range": w.t_range.value, "z_range": w.z_range.value, "y_range": w.y_range.value,
                "x_range": w.x_range.value, "channels": w.channels.value}

    def _zarr_layout(dataset) -> tuple:
        """``(axes, dtype, layout)``: level-0 chunks and shards of the output (see
        :func:`~zarrsmith.napari._zarr_options.chunk_shard_layout`)."""
        w = make_convert_widget
        axes = _dataset_axes(dataset)
        sizes = output_sizes(dataset, **_ranges())
        dtype = dataset.data[0].dtype
        layout = chunk_shard_layout(
            axes, [sizes[ax] for ax in axes], dtype, chunk_auto=w.chunk_auto.value, chunk_mb=w.chunk_mb.value,
            chunk_zyx=(w.chunk_z.value, w.chunk_y.value, w.chunk_x.value), zarr_format=_zarr_format(),
            shards=w.shards.value, shard_mb=w.shard_target_mb.value,
            shard_exclude_axes=tuple(w.shard_exclude_axes.value),
            voxel_size=recorded_voxel_size(dataset.metadata))
        return axes, dtype, layout

    def _chunk_zyx() -> tuple:
        """The ``(z, y, x)`` chunk to write: the automatic one, or the one typed."""
        w = make_convert_widget
        dataset = _state["dataset"]
        if dataset is None or not w.chunk_auto.value:
            return w.chunk_z.value, w.chunk_y.value, w.chunk_x.value
        axes, _, layout = _zarr_layout(dataset)
        sizes = dict(zip(axes, layout["chunks"]))
        return tuple(sizes.get(ax, 1) for ax in "zyx")

    def _show_chunks(axes, layout) -> None:
        """Fill in the chunk shape chosen automatically, and lock what does not apply."""
        w = make_convert_widget
        auto = bool(w.chunk_auto.value)
        if auto:
            _syncing["chunks"] = True
            try:
                for ax, size in zip(axes, layout["chunks"]):
                    if ax in "zyx":
                        getattr(w, f"chunk_{ax}").value = int(size)
            finally:
                _syncing["chunks"] = False
        w.chunk_mb.enabled = auto
        for ax in "zyx":
            getattr(w, f"chunk_{ax}").enabled = not auto and ax in axes

    def _channel_problems() -> dict:
        dataset = _state["dataset"]
        channels = output_channels(dataset, make_convert_widget.channels.value) if dataset is not None else []
        return channel_option_problems(make_convert_widget.channel_names.value,
                                       make_convert_widget.channel_colors.value, channels)

    def _channel_options() -> tuple:
        """The typed channel names and colours, checked against the converted channels (``ValueError``)."""
        error = channel_error(_channel_problems())
        if error:
            raise ValueError(error)
        return (parse_channel_names(make_convert_widget.channel_names.value),
                parse_channel_colors(make_convert_widget.channel_colors.value))

    def _check_channel_options(*_):
        """Warn, while typing, about channel names or colours that the conversion would refuse."""
        problems = _channel_problems()
        text = channel_warning_text(problems)
        channel_warning.setText(text)
        channel_warning.setVisible(bool(text))
        for name, messages in problems.items():
            getattr(make_convert_widget, name).native.setStyleSheet(
                "QLineEdit { border: 1px solid #e8a33d; }" if messages else "")

    def _follow_channel_names(*_, reset=False):
        """Fill in the names of the converted channels (spaces -> underscores) until the user types
        names of their own; ``reset`` (a new dataset) replaces whatever is there."""
        dataset = _state["dataset"]
        if dataset is None:
            return
        if reset:
            make_convert_widget.channel_names.value = ""
        follow_auto_value(make_convert_widget.channel_names,
                          auto_channel_names(output_channels(dataset, make_convert_widget.channels.value)),
                          _state, "auto_channel_names")

    def _show_levels(dataset, axes, dtype) -> None:
        """Fill in the number of pyramid levels zarrsmith convert would choose for the selected
        output (after the ranges and channels), unless the user typed one: the field then no
        longer holds the value filled in last."""
        w = make_convert_widget
        if _state.get("auto_levels") not in (None, w.n_levels.value):
            return
        sizes = output_sizes(dataset, **_ranges())
        downscale = (w.downscale_z.value, w.downscale_y.value, w.downscale_x.value)
        levels = min(w.n_levels.max, default_num_levels(
            {"axes": axes, "size": [[sizes[ax] for ax in axes]], "dtype": str(dtype)}, downscale))
        w.n_levels.value = levels
        _state["auto_levels"] = levels

    def _update_output_summary(*_):
        dataset = _state["dataset"]
        if dataset is None or _syncing.get("chunks"):
            return
        w = make_convert_widget
        try:
            text = output_summary(dataset, **_ranges(), drop_singleton=w.drop_singleton.value)
        except Exception as e:  # the recap must never break the widget
            text = f"cannot estimate ({e})"
        try:
            axes, dtype, layout = _zarr_layout(dataset)
            _show_chunks(axes, layout)
            _show_levels(dataset, axes, dtype)
            text += f"\nZarr: {layout_summary(axes, dtype, layout)}"
        except Exception as e:  # the recap must never break the widget
            text += f"\nZarr: cannot estimate ({e})"
        output_label.setText(f"Output: {text}")

    def _follow_output_name(*_, force=False):
        """Name the output after the dataset and the region it crops (nothing for the full
        dataset): ``scan_TR1.zarr``, then ``scan_TR1_y100-611_x200-711.zarr`` once a ROI is
        drawn. It follows the ROI and the ranges until the user types a path of their own;
        ``force`` names it again anyway (a new dataset was loaded)."""
        dataset, base = _state["dataset"], _state.get("output_base")
        if dataset is None or base is None:
            return
        current = make_convert_widget.output_path.value
        auto = _state["auto_output"]
        if not force and current is not None and auto is not None and not same_path(current, auto):
            return  # the user's own path
        parent, stem, suffix, is_zarr_input = base
        try:
            region = _region_tag(dataset, **_ranges())
        except Exception as e:  # the name must never break the widget
            print(f"Cannot name the output after the region: {e}")
            region = ""
        # The whole of a zarr input would otherwise be the input itself.
        new = parent / f"{stem}{region}{'_1' if is_zarr_input and not region else ''}{suffix}"
        _state["auto_output"] = new
        make_convert_widget.output_path.value = new

    for name in ("t_range", "z_range", "y_range", "x_range", "channels"):
        getattr(make_convert_widget, name).changed.connect(_follow_output_name)

    for name in ("t_range", "z_range", "y_range", "x_range", "channels", "drop_singleton", "chunk_auto",
                 "chunk_mb", "chunk_z", "chunk_y", "chunk_x", "zarr_format", "shards", "shard_target_mb",
                 "shard_exclude_axes", "downscale_z", "downscale_y", "downscale_x"):
        getattr(make_convert_widget, name).changed.connect(_update_output_summary)
    make_convert_widget.channels.changed.connect(_follow_channel_names)
    for name in ("channel_names", "channel_colors", "channels"):
        getattr(make_convert_widget, name).changed.connect(_check_channel_options)

    def _set_convert_enabled(enabled):
        """Enable or disable the conversion panel *and* its Convert button.

        The button sits below the (collapsible) panel, outside it, so that it stays
        visible when the panel is collapsed; disabling the panel alone would
        therefore not lock it, and a
        second click could start a second conversion. Its holder is disabled as
        a whole because magicgui re-enables its own call button after every
        call, which would undo disabling just the button.
        """
        make_convert_widget.enabled = enabled
        if "button_holder" in ui:
            ui["button_holder"].setEnabled(enabled)

    @make_convert_widget.zarr_format.changed.connect
    def _update_sharding_availability(zarr_format):
        # Sharding is a zarr v3 / NGFF 0.5 feature only.
        sharding_available = NGFF_FORMATS[zarr_format] == 3
        make_convert_widget.shards.enabled = sharding_available
        make_convert_widget.shard_target_mb.enabled = sharding_available
        make_convert_widget.shard_exclude_axes.enabled = sharding_available
        if not sharding_available:
            make_convert_widget.shards.value = "none"

    @magicgui(
        call_button="Append to batch CSV",
        # Save dialog: browse to a folder and type a new name (or pick an existing CSV).
        batch_csv_path={"widget_type": "FileEdit", "mode": "w", "filter": "CSV files (*.csv)", "nullable": True,
                        "tooltip": "CSV file the conversion is appended to (created if missing)."},
    )
    def make_batch_csv_widget(batch_csv_path: FileEdit = None):
        dataset = _state["dataset"]
        if dataset is None:
            return

        csv_path = Path(batch_csv_path) if batch_csv_path else None
        if csv_path is None or csv_path.is_dir():
            print("ERROR: choose a batch CSV file first (Choose file...). The row is not appended.")
            return

        overwrite = bool(make_convert_widget.overwrite.value)
        output = make_convert_widget.output_path.value
        sources = [make_visualize_widget.input_path.value]
        if not output or not has_zarr_name(output):
            output_allowed(output, overwrite=overwrite, sources=sources)  # warns why
            return

        try:
            typed_names, typed_colors = _channel_options()
        except ValueError as e:
            print(f"ERROR: {e} The row is not appended.")
            return

        selected_channel_names = list(make_convert_widget.channels.value) if "c" in _dataset_axes(dataset) else []
        # zarrsmith convert splits the names on spaces: the dataset's names get underscores
        normalized_channel_names = typed_names or auto_channel_names(
            output_channels(dataset, selected_channel_names)).split()
        subset = _subset_string_from_widget(
            dataset,
            make_convert_widget.t_range.value,
            make_convert_widget.z_range.value,
            make_convert_widget.y_range.value,
            make_convert_widget.x_range.value,
            selected_channel_names,
        )

        selected_colors = typed_colors or _channel_colors_from_metadata(
            dataset, output_channels(dataset, selected_channel_names))
        shards_selected = make_convert_widget.shards.value
        # one value per dataset axis, as zarrsmith convert expects
        chunk_size = _chunk_shape_for_dataset(dataset, *_chunk_zyx())
        row = {
            "input": str(Path(make_visualize_widget.input_path.value).resolve()),
            "microscope": str(make_visualize_widget.file_format.value),
            "output": str(Path(output)),
            "chunk_size": " ".join(str(v) for v in chunk_size),
            "max_size(MB)": "",
            "scene_index": str(make_visualize_widget.scene_index.value),
            "zarr_format": str(_zarr_format()),
            "downscale_factor": " ".join(str(v) for v in (make_convert_widget.downscale_z.value, make_convert_widget.downscale_y.value, make_convert_widget.downscale_x.value)),
            "subset": subset,
            "channel_colors": " ".join(selected_colors),
            "channel_names": " ".join(normalized_channel_names),
            "num_levels": str(make_convert_widget.n_levels.value),
            "method": "" if make_convert_widget.method.value == "auto" else str(make_convert_widget.method.value),
            # A blank cell means "auto" in the CLI, so disabling must be explicit.
            "shards": "none" if shards_selected == "none" else str(shards_selected),
            "shard_target_mb": "" if shards_selected == "none" else str(make_convert_widget.shard_target_mb.value),
            "shard_exclude_axes": (
                "" if shards_selected == "none"
                else (" ".join(make_convert_widget.shard_exclude_axes.value) or "none")
            ),
            "drop_singleton": str(bool(make_convert_widget.drop_singleton.value)).lower(),
            "num_workers": "" if int(make_convert_widget.num_workers.value) <= 0 else str(int(make_convert_widget.num_workers.value)),
            "overwrite": str(overwrite).lower(),
        }

        # An output the CSV already writes to: ask
        # whether to add the row anyway, under the next free name (_1, _2...).
        if not confirm_batch_output(csv_path, row):
            return
        output = row["output"]

        if not output_allowed(output, overwrite=overwrite, sources=sources):
            return
        csv_path = _append_batch_csv(csv_path, row)
        print(f"Appended batch row to {csv_path}. Run it with:\n  zarrsmith convert --batch \"{csv_path}\"")

    make_batch_csv_widget.enabled = False  # the CSV field starts empty and is never filled in for the user


    # -------------------------
    # Input path callback
    # -------------------------

    @make_visualize_widget.file_format.changed.connect
    def _update_mode(file_format):
        if file_format in ("zeiss", "evident"): 
            make_visualize_widget.scene_index.enabled = True 
        else: 
            make_visualize_widget.scene_index.enabled = False
        
        if file_format in ["zarr", "viventis", "luxendo"]:
            make_visualize_widget.input_path.mode = "d" 
        else: 
            make_visualize_widget.input_path.mode = "r"

    def _open_dataset(path):
        """Open ``path`` at the scene of the widget, limited to the scenes of this file."""
        reader = dataset_reader(make_visualize_widget.file_format.value)
        if make_visualize_widget.file_format.value not in ("zeiss", "evident"):
            return reader(path)

        scene_widget = make_visualize_widget.scene_index
        dataset = reader(path, scene_index=0)
        last = len(dataset.scenes) - 1
        scene = min(scene_widget.value, last)
        with scene_widget.changed.blocked():
            scene_widget.value = scene
            scene_widget.max = last
        print(f"Scene {scene} of {dataset.scenes}: {dataset.scenes[scene]}")
        return dataset if scene == 0 else reader(path, scene_index=scene)

    @make_visualize_widget.input_path.changed.connect
    def _on_input_path_change(path):
        if not path:
            return
        _load_dataset(path)

    @make_visualize_widget.scene_index.changed.connect
    def _on_scene_change(scene):
        path = make_visualize_widget.input_path.value
        if _state["dataset"] is not None and make_visualize_widget.scene_index.enabled:
            _load_dataset(Path(path), new_scene=True)

    def _load_dataset(path, new_scene=False):
        dataset = _open_dataset(path)

        for i in dataset.metadata:
            print(f"{i.upper()}: {dataset.metadata[i]}")

        _state["dataset"] = dataset

        is_zarr_input = path.name.lower().endswith(".zarr")
        if is_zarr_input:
            _, name, out_suffix = split_zarr_name(path)  # scan.ome.zarr -> scan, .ome.zarr
        else:
            name = path.stem if path.is_file() else path.name  # a directory name as-is
            out_suffix = ".zarr"

        tag = _scene_tag(dataset, make_visualize_widget.scene_index.value)
        # Follow the scene and the ROI until the user types a path of their own.
        current, auto = make_convert_widget.output_path.value, _state["auto_output"]
        follow_output = not new_scene or current is None or (auto is not None and same_path(current, auto))
        if follow_output:
            _state["auto_output"] = None  # the previous name is not the user's
        _state["output_base"] = (path.parent, f"{name}{tag}", out_suffix, is_zarr_input)
        if follow_output:
            _follow_output_name(force=True)
        make_convert_widget.overwrite.value = False  # asked for one output, not for the next one
        make_batch_csv_widget.enabled = True


        _set_convert_enabled(True)

        axes = _dataset_axes(dataset)
        downscale = (make_convert_widget.downscale_z.value, make_convert_widget.downscale_y.value,
                     make_convert_widget.downscale_x.value)
        num_levels = default_num_levels(dataset.metadata, downscale)

        # the chunk shape itself is filled in with the output summary (auto chunks)
        for ax in "zyx":
            chunk = getattr(make_convert_widget, f"chunk_{ax}")
            chunk.max = _axis_size(dataset, ax) + 1
            chunk.enabled = ax in axes and not make_convert_widget.chunk_auto.value

        make_convert_widget.n_levels.value = num_levels
        _state["auto_levels"] = num_levels  # follows the ranges from now on (see _show_levels)
        make_convert_widget.downscale_z.enabled = "z" in axes
        make_convert_widget.downscale_y.enabled = "y" in axes
        make_convert_widget.downscale_x.enabled = "x" in axes

        _set_range_widget(make_convert_widget.t_range, _axis_size(dataset, "t"), enabled="t" in axes)
        _set_range_widget(make_convert_widget.z_range, _axis_size(dataset, "z"), enabled="z" in axes)
        _set_range_widget(make_convert_widget.y_range, _axis_size(dataset, "y"), enabled="y" in axes)
        _set_range_widget(make_convert_widget.x_range, _axis_size(dataset, "x"), enabled="x" in axes)
        make_convert_widget.single_t.enabled = "t" in axes
        make_convert_widget.single_z.enabled = "z" in axes

        channel_names = dataset.metadata.get("channel_names", []) if "c" in axes else []
        n_channels = len(channel_names)
        row_height = 20  # approx height per item in pixels
        channels_widget = make_convert_widget.channels.native
        channels_widget.setMaximumHeight(row_height * max(1, n_channels))
        make_convert_widget.channels.choices = channel_names
        make_convert_widget.channels.value = tuple(channel_names)
        make_convert_widget.channels.enabled = n_channels > 0
        # no channel axis: nothing to name (zarrsmith convert refuses names then)
        make_convert_widget.channel_names.enabled = make_convert_widget.channel_colors.enabled = n_channels > 0
        if not n_channels:
            make_convert_widget.channel_colors.value = ""
        _follow_channel_names(reset=True)  # this dataset's channel names, not the previous one's
        _check_channel_options()

        if follow_output:
            _follow_output_name(force=True)  # the full dataset: ranges and channels were just reset
        _update_output_summary()

        viewer.layers.clear()

    make_visualize_widget.scene_index.enabled = False
    _set_convert_enabled(False)
    viewer.dims.events.ndisplay.connect(lock_roi_in_3d)
    # lock_roi_in_3d()

    def sync_single_z(checked):
        dataset = _state.get("dataset")
        has_z = bool(dataset and _axis_index(dataset, "z") is not None and _axis_size(dataset, "z") > 1)
        if checked and has_z:
            z_val = make_convert_widget.z_range.value
            # clamp to single plane
            make_convert_widget.z_range.value = (z_val[0], z_val[0])
            make_convert_widget.z_range.enabled = False
        else:
            make_convert_widget.z_range.enabled = has_z

    make_convert_widget.single_z.changed.connect(sync_single_z)

    def sync_single_t(checked):
        dataset = _state.get("dataset")
        has_t = bool(dataset and _axis_index(dataset, "t") is not None and _axis_size(dataset, "t") > 1)
        if checked and has_t:
            t_val = make_convert_widget.t_range.value
            # clamp to single frame
            make_convert_widget.t_range.value = (t_val[0], t_val[0])
            make_convert_widget.t_range.enabled = False
        else:
            make_convert_widget.t_range.enabled = has_t

    make_convert_widget.single_t.changed.connect(sync_single_t)
    # -------------------------
    # Compose single dock widget
    # -------------------------

    container = QWidget()
    layout = QVBoxLayout(container)
    
    # Collapsible section header, same pattern as "Additional parameters".
    title2_btn = QToolButton()
    title2_btn.setCheckable(True)
    title2_btn.setChecked(False)
    title2_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

    def _toggle_conversion(checked):
        # Only the parameters collapse: the title and the Convert button of the
        # box stay visible.
        make_convert_widget.native.setVisible(checked)
        title2_btn.setText("Dataset conversion: ▾" if checked else "Dataset conversion: ▸")
        title2_btn.setStyleSheet("""
            QToolButton {
                font-weight: bold;
                font-size: 13px;
                padding: 2px 0px;
                color: %s;
            }
            QToolButton:hover {
                color: white;
            }
        """ % ("#ffffff" if checked else "#aaaaaa"))

    title2_btn.toggled.connect(_toggle_conversion)

    section_loading = _make_section(make_title("Dataset loading:"), make_visualize_widget.native)
    section_conversion = _make_section(title2_btn)
    layout.setSpacing(6)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.addWidget(section_loading)
    layout.addWidget(section_conversion)

    advanced_btn = QToolButton()
    advanced_btn.setText("Additional parameters ▸")
    advanced_btn.setStyleSheet("""
        QToolButton {
            font-size: 13px;
            font-weight: 600;
            color: #cccccc;
            padding: 4px 0px;
        }
        QToolButton:hover {
            color: white;
        }
        """)
    advanced_btn.setCheckable(True)
    advanced_btn.setChecked(False)
    advanced_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

    def _toggle_advanced(checked):
        for w in advanced_widgets:
            w.setVisible(checked)
        advanced_btn.setText("Additional parameters ▾" if checked else "Additional parameters ▸")
        advanced_btn.setStyleSheet("""
            QToolButton {
                font-size: 13px;
                font-weight: 600;
                color: %s;
            }
        """ % ("#ffffff" if checked else "#aaaaaa"))
        
    advanced_btn.toggled.connect(_toggle_advanced)

    make_convert_widget.native.layout().insertWidget(7, advanced_btn)

    advanced_widgets = _compact_advanced_rows(make_convert_widget, advanced_btn)
    for w in advanced_widgets:
        w.setVisible(False)

    # Tighter vertical spacing and a shorter channel list keep the panel small.
    make_convert_widget.native.layout().setSpacing(2)
    make_convert_widget.native.layout().setContentsMargins(0, 0, 0, 0)
    make_convert_widget.channels.native.setMaximumHeight(64)
    make_convert_widget.channel_names.native.setPlaceholderText("dataset channel names, e.g. DAPI Alexa_488")
    make_convert_widget.channel_colors.native.setPlaceholderText("dataset colors, e.g. magenta 00FF00")

    section_conversion.layout().addWidget(make_convert_widget.native)
    _toggle_conversion(title2_btn.isChecked())

    layout.addWidget(_make_section(make_title("Batch CSV export:"), make_batch_csv_widget.native))

    log_section, log_widget = make_log_pane()

    stdout_stream = EmittingStream()
    stderr_stream = EmittingStream()

    previous_streams = (sys.stdout, sys.stderr)
    sys.stdout = stdout_stream
    sys.stderr = stderr_stream

    def _restore_console(*_):
        # Closing the widget deletes its log: output must not keep going to it.
        if sys.stdout is stdout_stream:
            sys.stdout = previous_streams[0]
        if sys.stderr is stderr_stream:
            sys.stderr = previous_streams[1]

    log_widget.destroyed.connect(_restore_console)

    append_with_time = make_log_appender(log_widget)

    stdout_stream.text_written.connect(append_with_time)
    stderr_stream.text_written.connect(append_with_time)

    # The Convert button stays inside the "Dataset conversion" box, below the
    # (collapsible) parameters, so it is visible even when they are collapsed.
    convert_button = make_convert_widget._call_button.native
    make_convert_widget.native.layout().removeWidget(convert_button)
    button_holder = make_button_holder(convert_button)
    ui["button_holder"] = button_holder  # locked as a whole while converting (see _set_convert_enabled)
    button_holder.setEnabled(make_convert_widget.native.isEnabled())
    section_conversion.layout().addWidget(output_label)  # visible with the parameters collapsed
    section_conversion.layout().addWidget(channel_warning)  # likewise
    section_conversion.layout().addWidget(button_holder)

    # All three boxes (input, conversion, batch) scroll together; the log is a
    # resizable pane below.
    layout.addStretch(1)
    dock = compose_dock(container, bottom=log_section)
    # exposed for tests
    dock.visualize_widget, dock.convert_widget = make_visualize_widget, make_convert_widget
    dock.batch_csv_widget = make_batch_csv_widget
    dock.output_label = output_label
    dock.channel_warning = channel_warning
    return dock
