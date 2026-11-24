"""Defaults and registries shared by the CLI and the napari widgets."""
from __future__ import annotations

import numpy as np
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.utils.axes import metadata_axes, metadata_axis_size
from zarrsmith.utils.chunks import default_chunk_size, default_num_levels


def _metadata(axes, size, dtype="uint16", scale=None):
    return {"axes": axes, "size": [size], "dtype": dtype, **({"scales": [scale]} if scale else {})}


def test_metadata_axes_and_sizes():
    md = _metadata("TZYX", (3, 10, 20, 30))
    assert metadata_axes(md) == "tzyx" and metadata_axes({}) == "tczyx"
    assert metadata_axis_size(md, "y") == 20 and metadata_axis_size(md, "c") == 1


def test_default_chunk_size_follows_the_dtype():
    chunks, size_mb, n_chunks = default_chunk_size(_metadata("tczyx", (1, 2, 100, 4096, 4096)))
    assert chunks == (1, 1, 32, 256, 256) and size_mb == 4  # about 4 MB, at most 32 planes
    assert n_chunks == {"z": 4, "y": 16, "x": 16}
    wide, wide_mb, _ = default_chunk_size(_metadata("tczyx", (1, 2, 100, 4096, 4096), "float32"))
    assert wide == (1, 1, 32, 128, 256) and wide_mb == 4  # 4 bytes a voxel: smaller chunks
    assert default_chunk_size(_metadata("yx", (50, 60)))[0] == (50, 60)  # small image: one chunk
    # max_size_mb still sets the size; old callers passing 100 get chunks of up to 100 MB
    big, big_mb, _ = default_chunk_size(_metadata("tczyx", (1, 2, 100, 4096, 4096)), max_size_mb=100)
    assert 50 < big_mb <= 100


def test_default_chunks_fit_the_image():
    assert default_chunk_size(_metadata("yx", (50000, 80000)))[0] == (1024, 2048)  # 2D: 4 MB
    assert default_chunk_size(_metadata("czyx", (2, 5, 2048, 2048)))[0] == (1, 5, 512, 512)  # few planes
    # y and x span the image: z takes the rest of the size
    assert default_chunk_size(_metadata("zyx", (500, 64, 64)))[0] == (500, 64, 64)
    assert default_chunk_size(_metadata("zyx", (2000, 64, 64)))[0] == (512, 64, 64)


def test_default_chunks_are_cubic_in_micrometers_up_to_32_planes():
    from zarrsmith.utils.chunks import auto_chunk_shape

    shape = (301, 1880, 2566)
    # isotropic voxels, or z steps up to 8 times the pixel size: cubic, then capped at 32 planes
    for scale in (None, (1.0, 1.0, 1.0), (2.0, 0.325, 0.325)):
        assert default_chunk_size(_metadata("zyx", shape, scale=scale))[0] == (32, 256, 256)
    # coarser z steps: fewer planes, so the chunk stays about cubic in micrometers
    assert default_chunk_size(_metadata("zyx", shape, scale=(10.4, 0.65, 0.65)))[0] == (16, 256, 512)
    # y and x too: pixels twice as wide in x make the chunk twice as long in y
    assert auto_chunk_shape("yx", (50000, 80000), "uint16", voxel_size=(0.5, 1.0)) == (2048, 1024)
    # unknown (None) or unusable sizes: chunks are cubic in voxels
    for vs in ({"z": None, "y": 0.65, "x": 0.65}, (0.0, 1.0, 1.0), ("a", 1, 1)):
        assert auto_chunk_shape("zyx", shape, "uint16", voxel_size=vs) == (32, 256, 256)


def test_unrecorded_voxel_sizes_are_none():
    from zarrsmith.utils.axes import UNRECORDED_SCALE_KEY, recorded_voxel_size

    md = {**_metadata("czyx", (2, 301, 1880, 2566), scale=(1.0, 0.65, 0.65)), UNRECORDED_SCALE_KEY: ("z",)}
    assert recorded_voxel_size(md) == {"z": None, "y": 0.65, "x": 0.65}
    assert recorded_voxel_size(_metadata("yx", (10, 10))) is None  # no scales at all
    # the unknown z counts in voxels: z steps of 1.0 are not taken as recorded
    md["scales"] = [(1.0, 0.1, 0.1)]
    assert default_chunk_size(md)[0] == (1, 32, 256, 256)


def test_default_num_levels():
    assert default_num_levels(_metadata("zyx", (10, 1000, 1000))) == 1
    assert default_num_levels(_metadata("zyx", (10, 8192, 8192))) == 3  # 8192 -> 4096 -> 2048
    # 2D image with a Z, Y, X factor: the Y and X factors apply (not Z's)
    assert default_num_levels(_metadata("yx", (8192, 8192)), (1, 4, 4)) == 2
    # a factor of 1 never makes an axis smaller: no endless loop
    assert default_num_levels(_metadata("zyx", (4096, 100, 100)), (1, 2, 2)) == 1


def test_napari_reader_uses_the_shared_registry():
    pytest.importorskip("magicgui")
    from zarrsmith.napari._convert_widget import dataset_reader

    assert dataset_reader("zeiss") is mm.READERS["zeiss"] is mm.ZeissManager
    with pytest.raises(ValueError, match="Unsupported"):
        dataset_reader("nope")


def test_cli_reader_detection(tmp_path):
    from zarrsmith.cli.zarrsmith import _resolve_zarr_manager

    store = tmp_path / "a.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    assert mm.detect_reader(store) == (mm.ZarrManager, "zarr")
    assert _resolve_zarr_manager(str(tmp_path / "x.czi")) == (mm.ZeissManager, "zeiss")
    assert mm.detect_reader(tmp_path / "unknown.xyz") is None
    with pytest.raises(TypeError, match="Could not auto-detect"):
        _resolve_zarr_manager(str(tmp_path / "unknown.xyz"))


def test_cli_color_errors_are_argparse_errors():
    import argparse

    from zarrsmith.cli.__arguments import parse_color

    assert parse_color("#ff00ff") == "FF00FF" and parse_color("cyan") == "00FFFF"
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid color"):
        parse_color("notacolor")


def test_add_layers_returns_the_layers():
    pytest.importorskip("napari")
    import dask.array as da
    from napari.components import ViewerModel

    from zarrsmith.utils.visualize import add_layers

    md = {"axes": "czyx", "scales": [(2.0, 0.5, 0.5)], "units": ("micrometer",) * 3,
          "channel_names": ["A", "B"], "channel_colors": ["FF0000", "00FF00"]}
    viewer = ViewerModel()
    layers = add_layers(viewer, [da.zeros((2, 4, 8, 8), np.uint16)], md, name="img", blending="additive")
    assert [layer.name for layer in layers] == ["img: A", "img: B"]
    assert all(layer.blending == "additive" for layer in layers)
    assert tuple(layers[0].scale) == (2.0, 0.5, 0.5)


def test_readers_record_which_voxel_sizes_are_missing(tmp_path):
    """A voxel size the file does not record is ``None`` for chunking (its 1.0 is only a placeholder)."""
    import warnings

    from zarrsmith.microscope_manager import ArrayManager
    from zarrsmith.utils.axes import recorded_voxel_size
    from zarrsmith.utils.ome import parse_ome_xml

    xml = ('<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image><Pixels SizeX="10" SizeY="10" '
           'SizeZ="5" PhysicalSizeX="0.5" PhysicalSizeY="0.5" Type="uint16"/></Image></OME>')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px = parse_ome_xml(xml)
    assert px.scale_zyx == (1.0, 0.5, 0.5) and px.unrecorded_axes == ("z",)

    arr = np.zeros((4, 8, 8), np.uint16)
    assert recorded_voxel_size(ArrayManager(arr, metadata={"axes": "zyx"}).metadata) == {"z": None, "y": None, "x": None}
    given = ArrayManager(arr, metadata={"axes": "zyx", "scales": [(2.0, 0.5, 0.5)]}).metadata
    assert recorded_voxel_size(given) == {"z": 2.0, "y": 0.5, "x": 0.5}

    # an OME-Zarr written with voxel sizes reads them back as recorded
    out = tmp_path / "out.ome.zarr"
    ArrayManager(arr, metadata={"axes": "zyx", "scales": [(2.0, 0.5, 0.5)]}).to_zarr(str(out), progress=False)
    assert recorded_voxel_size(mm.ZarrManager(str(out)).metadata) == {"z": 2.0, "y": 0.5, "x": 0.5}
