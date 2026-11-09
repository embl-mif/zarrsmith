"""Every OME-Zarr layout zarrsmith writes must validate against the NGFF spec.

The validation uses ome-zarr-models, pydantic models derived from the spec, on
every group of the written hierarchy: images, ``labels`` containers and label
images, for both NGFF 0.4 (zarr v2) and 0.5 (zarr v3).
"""
from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import zarr

import zarrsmith.microscope_manager as mm

ome_zarr_models = pytest.importorskip("ome_zarr_models")
from ome_zarr_models import v04, v05  # noqa: E402

FORMATS = [("0.4", 2), ("0.5", 3)]
_MODELS = {"0.4": v04, "0.5": v05}


def _ngff_attrs(group: zarr.Group) -> dict:
    attrs = group.attrs.asdict()
    return attrs["ome"] if isinstance(attrs.get("ome"), dict) else attrs


def assert_valid_ome_zarr(path, ngff_version: str) -> list[str]:
    """Validate every NGFF group below ``path``; return the validated group paths."""
    models = _MODELS[ngff_version]
    root = zarr.open_group(str(path), mode="r")
    checked = []

    def visit(group: zarr.Group, where: str) -> None:
        attrs = _ngff_attrs(group)
        if "multiscales" in attrs:
            model = models.ImageLabel if "image-label" in attrs else models.Image
        elif "labels" in attrs:
            model = models.Labels
        else:
            model = None
        if model is not None:
            try:
                model.from_zarr(group)
            except Exception as err:  # pydantic errors carry the full explanation
                pytest.fail(f"{where or '/'} is not a valid NGFF {ngff_version} "
                            f"{model.__name__}:\n{err}")
            checked.append(where or "/")
        for name, sub in group.groups():
            visit(sub, f"{where}/{name}")

    visit(root, "")
    assert checked, f"no NGFF group found in {path}"
    return sorted(checked)  # group iteration order differs between zarr v2 and v3


@pytest.mark.parametrize("ngff_version,zarr_format", FORMATS)
def test_image_with_labels_and_subgroup_is_valid(tmp_path, image_pyramid, metadata,
                                                 ngff_version, zarr_format):
    path = tmp_path / "image.zarr"
    mm.ArrayManager(image_pyramid, metadata).to_zarr(
        path, ngff_version=ngff_version, zarr_format=zarr_format, progress=False)
    z = mm.ZarrManager(path, mode="a")
    z.create_empty_group("nuclei", metadata, is_label=True)
    z.create_empty_group("deconvolved", metadata)

    checked = assert_valid_ome_zarr(path, ngff_version)
    assert checked == ["/", "/deconvolved", "/labels", "/labels/nuclei"]

    # Rewriting the whole hierarchy goes through write_multiscale_to_group.
    copy = tmp_path / "copy.zarr"
    mm.ZarrManager(path, mode="r").to_zarr(
        copy, ngff_version=ngff_version, zarr_format=zarr_format, progress=False)
    assert assert_valid_ome_zarr(copy, ngff_version) == checked


@pytest.mark.parametrize("ngff_version,zarr_format", FORMATS)
def test_empty_dataset_is_valid(tmp_path, metadata, ngff_version, zarr_format):
    path = tmp_path / "empty.zarr"
    mm.ZarrManager(path, mode="a", metadata=metadata,
                   ngff_version=ngff_version, zarr_format=zarr_format)
    assert assert_valid_ome_zarr(path, ngff_version) == ["/"]


@pytest.mark.parametrize("ngff_version,zarr_format", FORMATS)
def test_standalone_label_store_is_valid(tmp_path, ngff_version, zarr_format):
    labels = np.zeros((8, 9), dtype=np.uint16)
    labels[2:6, 3:7] = 4
    metadata = {
        "size": [(8, 9), (4, 5)],
        "scales": [(0.25, 0.25), (0.5, 0.5)],
        "units": ("micrometer", "micrometer"),
        "dtype": "uint16",
        "axes": "yx",
        "data_type": "label",
    }
    path = tmp_path / "labels.zarr"
    levels = [da.from_array(labels, chunks=(4, 5)), da.from_array(labels[::2, ::2], chunks=(2, 3))]
    mm.ArrayManager(levels, metadata).to_zarr(
        path, ngff_version=ngff_version, zarr_format=zarr_format, progress=False)
    assert assert_valid_ome_zarr(path, ngff_version) == ["/"]


def test_v05_arrays_carry_dimension_names(tmp_path, image_pyramid, metadata):
    """NGFF 0.5 needs the axis names in each array's zarr.json, not in its attributes."""
    path = tmp_path / "dims.zarr"
    mm.ArrayManager(image_pyramid, metadata).to_zarr(path, progress=False)
    root = zarr.open_group(str(path), mode="r")
    for level in ("0", "1", "2"):
        assert root[level].metadata.dimension_names == tuple("tczyx")
        assert "dimension_names" not in root[level].attrs
