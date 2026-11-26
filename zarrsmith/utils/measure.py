"""Measure the objects of a label image against an intensity image (``zarrsmith measure``):
one row per label (and timepoint) with ``skimage.measure.regionprops`` shape properties and,
per channel, intensity properties."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .axes import metadata_axes
from .labels import _plane as plane, fit_shape, resolve_channel

#: Shape properties measured for every label (2D labels: also perimeter, orientation,
#: eccentricity, axis lengths, and the form factor 4 pi area / perimeter^2).
BASE_PROPERTIES_2D = ("label", "area", "bbox", "centroid", "perimeter", "orientation", "eccentricity",
                      "solidity", "extent", "major_axis_length", "minor_axis_length")
BASE_PROPERTIES_ND = ("label", "area", "bbox", "centroid", "solidity", "extent")
#: Intensity properties measured per channel, as ``<property>_ch<i>[_<name>]`` columns.
BASE_INTENSITY_PROPERTIES = ("label", "mean_intensity", "min_intensity", "max_intensity")


def _suffix(index: int, names: Sequence[str]) -> str:
    if index < len(names) and names[index]:
        return f"ch{index}_" + (re.sub(r"[^0-9A-Za-z]+", "_", str(names[index])).strip("_") or "channel")
    return f"ch{index}"


def _shape_table(labels: np.ndarray, extra: Sequence[str]):
    import pandas as pd
    from skimage.measure import regionprops_table

    base = BASE_PROPERTIES_2D if labels.ndim == 2 else BASE_PROPERTIES_ND
    try:
        table = pd.DataFrame(regionprops_table(labels, properties=base))
    except NotImplementedError:
        table = pd.DataFrame(regionprops_table(labels, properties=BASE_PROPERTIES_ND))
    if "perimeter" in table.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            table["form_factor"] = np.where(table["perimeter"] > 0,
                                            4 * np.pi * table["area"] / np.square(table["perimeter"]), 0.0)
    if extra:  # asked for explicitly: an unsupported property is an error, not dropped
        table = table.merge(pd.DataFrame(regionprops_table(labels, properties=("label", *extra))), on="label")
    return table


def measure_labels(image, labels, *, channels: Sequence | None = None, level: int = 0, labels_level: int = 0,
                   properties: Sequence[str] = (), intensity_properties: Sequence[str] = (),
                   dataset_id: str | None = None):
    """Measure every label of ``labels`` (a ``ZarrManager`` at ``labels_level``) against the
    ``channels`` (indices or names; default: all) of ``image`` (at ``level``).

    Returns a DataFrame: one row per label (and per timepoint, with a ``timepoint`` column,
    when both have a ``t`` axis; a ``t`` of 1 on the image alone is dropped). The columns are
    the shape properties (:data:`BASE_PROPERTIES_2D` / :data:`BASE_PROPERTIES_ND`, plus
    ``properties``) and, per channel, :data:`BASE_INTENSITY_PROPERTIES` plus
    ``intensity_properties`` as ``<property>_ch<i>_<name>``. Labels whose level was rounded
    differently from the image's are cropped or zero-padded to it. ``dataset_id`` adds a first
    column with that value."""
    import pandas as pd
    from skimage.measure import regionprops_table

    axes = metadata_axes(image.metadata)
    label_axes = metadata_axes(labels.metadata)
    if "c" not in axes:
        raise ValueError(f"The image has no channel axis: axes {axes}.")
    for name, manager, lv in (("image", image, level), ("labels", labels, labels_level)):
        if not 0 <= lv < len(manager.data):
            raise ValueError(f"The {name} level {lv} is out of range ({len(manager.data)} level(s)).")
    data, lab_data = image.data[level], labels.data[labels_level]
    names = list(image.metadata.get("channel_names") or [])
    n_c = int(data.shape[axes.index("c")])
    chosen = [resolve_channel(c, names, n_c) for c in channels] if channels else list(range(n_c))

    has_t, lab_t = "t" in axes, "t" in label_axes
    drop_t = has_t and not lab_t and int(data.shape[axes.index("t")]) == 1
    if not drop_t and has_t != lab_t:
        raise ValueError(f"The image and the labels disagree on a time axis (image {axes}, labels {label_axes}).")
    if has_t and lab_t and data.shape[axes.index("t")] != lab_data.shape[label_axes.index("t")]:
        raise ValueError("The image and the labels have a different number of timepoints.")
    timepoints = list(range(int(lab_data.shape[label_axes.index("t")]))) if lab_t else [None]

    tables = []
    for t in timepoints:
        image_fixed = {"t": 0} if drop_t else ({"t": t} if t is not None else {})
        lab = plane(lab_data, label_axes, {"t": t} if t is not None else {}).astype(np.int64, copy=False)
        first = plane(data, axes, {"c": chosen[0], **image_fixed})
        if lab.shape != first.shape:
            lab = fit_shape(lab, first.shape)
        table = _shape_table(lab, properties)
        for c in chosen:
            channel = first if c == chosen[0] else plane(data, axes, {"c": c, **image_fixed})
            values = pd.DataFrame(regionprops_table(lab, intensity_image=channel,
                                                    properties=BASE_INTENSITY_PROPERTIES + tuple(intensity_properties)))
            suffix = _suffix(c, names)
            table = table.merge(values.rename(columns={k: f"{k}_{suffix}" for k in values.columns if k != "label"}),
                                on="label")
        if t is not None:
            table.insert(0, "timepoint", t)
        tables.append(table)
    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if dataset_id:
        result.insert(0, "dataset_id", dataset_id)
    return result
