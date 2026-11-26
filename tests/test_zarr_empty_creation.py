# tests/test_zarr_empty_creation.py
from __future__ import annotations

import zarr
import pytest

import zarrsmith.microscope_manager as mm


def _assert_ngff_attrs(attrs, ngff_version):
    """The NGFF blocks sit at the root of the attributes in 0.4, under ``ome`` in 0.5."""
    if ngff_version == "0.4":
        assert "multiscales" in attrs and "omero" in attrs and "ome" not in attrs
    else:
        assert attrs["ome"]["version"] == ngff_version


FORMATS = pytest.mark.parametrize("ngff_version, zarr_format", [("0.4", 2), ("0.5", 3)])


@FORMATS
def test_create_empty_dataset_from_metadata(tmp_path, metadata, ngff_version, zarr_format):
    path = tmp_path / f"empty_v{ngff_version}.zarr"
    mm.ZarrManager(str(path), metadata={**metadata, "ngff_version": ngff_version, "zarr_format": zarr_format}, mode="a")

    root = zarr.open_group(str(path), mode="r")
    _assert_ngff_attrs(root.attrs.asdict(), ngff_version)
    assert "0" in root
    assert tuple(root["0"].shape) == metadata["size"][0]


@FORMATS
def test_create_empty_group_inherits_root_format(tmp_path, image_pyramid, metadata, ngff_version, zarr_format):
    root_path = tmp_path / f"root_v{ngff_version}.zarr"
    mm.ArrayManager(image_pyramid, metadata).to_zarr(
        str(root_path),
        ngff_version=ngff_version,
        zarr_format=zarr_format,
        overwrite=True,
    )

    d = mm.ZarrManager(str(root_path), mode="a")
    d.create_empty_group("processed", metadata)

    root = zarr.open_group(str(root_path), mode="r")
    _assert_ngff_attrs(root["processed"].attrs.asdict(), ngff_version)


def test_create_empty_label_group(tmp_path, image_pyramid, metadata):
    root_path = tmp_path / "with_label.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    
    writer.to_zarr(
        str(root_path),
        ngff_version="0.5",
        zarr_format=3,
        overwrite=True,
    )

    d = mm.ZarrManager(str(root_path), mode="a")
    d.create_empty_group("nuclei", metadata, is_label=True)

    root = zarr.open_group(str(root_path), mode="r")
    assert "labels" in root
    assert "labels" not in root.attrs.asdict()["ome"]
    assert root["labels"].attrs.asdict()["ome"]["labels"] == ["nuclei"]
    assert "nuclei" in root["labels"]
    label_grp = root["labels"]["nuclei"]
    assert "0" in label_grp
    # Nested under <image>/labels/<name>: the source image is two levels up.
    assert label_grp.attrs.asdict()["ome"]["image-label"] == {"version": "0.5", "source": {"image": "../../"}}


def test_create_empty_label_group_v04_registers_labels_group(tmp_path, image_pyramid, metadata):
    root_path = tmp_path / "with_label_v04.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)

    writer.to_zarr(
        str(root_path),
        ngff_version="0.4",
        zarr_format=2,
        overwrite=True,
    )

    d = mm.ZarrManager(str(root_path), mode="a")
    d.create_empty_group("nuclei", metadata, is_label=True)

    root = zarr.open_group(str(root_path), mode="r")
    assert "labels" not in root.attrs.asdict()
    assert root["labels"].attrs.asdict()["labels"] == ["nuclei"]
