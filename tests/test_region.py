"""Physical <-> pixel bounds and region reads with a halo (zarrsmith.utils.region)."""
from __future__ import annotations

import numpy as np
import pytest

from zarrsmith.microscope_manager import ZarrManager
from zarrsmith.utils.region import (
    halo_window,
    physical_to_pixel_bounds,
    physical_to_pixels,
    pixel_to_physical_bounds,
    read_region,
)
from tests.test_allocate_pyramid import FORMAT_IDS, FORMATS, _make_template

# pixel i of y is centred on 10 + 0.5 i and covers [9.75 + 0.5 i, 10.25 + 0.5 i)
SCALE, TRANSLATION, SHAPE = {"y": 0.5}, {"y": 10.0}, {"y": 100}


def test_physical_bounds_take_every_pixel_they_overlap():
    # on pixel edges: the pixel that only touches a bound is left out
    assert physical_to_pixel_bounds({"y": (10.25, 11.25)}, SCALE, TRANSLATION, SHAPE) == {"y": (1, 3)}
    # inside pixels: both partly covered pixels are included
    assert physical_to_pixel_bounds({"y": (10.3, 11.3)}, SCALE, TRANSLATION, SHAPE) == {"y": (1, 4)}
    # open sides are the image edges
    assert physical_to_pixel_bounds({"y": (None, 11.0)}, SCALE, TRANSLATION, SHAPE) == {"y": (0, 3)}
    assert physical_to_pixel_bounds({"y": (59.0, None)}, SCALE, TRANSLATION, SHAPE) == {"y": (98, 100)}


def test_physical_bounds_outside_the_image():
    with pytest.raises(ValueError, match="outside the image"):
        physical_to_pixel_bounds({"y": (0.0, 11.0)}, SCALE, TRANSLATION, SHAPE)
    assert physical_to_pixel_bounds({"y": (0.0, 11.0)}, SCALE, TRANSLATION, SHAPE, clip=True) == {"y": (0, 3)}
    with pytest.raises(ValueError, match="cover no pixel"):
        physical_to_pixel_bounds({"y": (11.0, 11.0)}, SCALE, TRANSLATION, SHAPE)


def test_pixel_and_physical_bounds_round_trip():
    pixels = {"y": (4, 17)}
    physical = pixel_to_physical_bounds(pixels, SCALE, TRANSLATION)
    assert physical == {"y": (9.75 + 2.0, 9.75 + 8.5)}
    assert physical_to_pixel_bounds(physical, SCALE, TRANSLATION, SHAPE) == pixels


def test_physical_lengths_round_up():
    assert physical_to_pixels(1.0, 0.5) == 2
    assert physical_to_pixels(1.1, 0.5) == 3
    assert physical_to_pixels(0.0, 0.5) == 0


def test_halo_window_is_clipped_at_the_edges():
    window, crop = halo_window({"y": (2, 10), "x": (90, 100)}, {"y": 5, "x": 5}, {"y": 100, "x": 100})
    assert window == {"y": (0, 15), "x": (85, 100)}
    assert crop == {"y": slice(2, 10), "x": slice(5, 15)}


def test_read_region_from_a_plain_array():
    data = np.arange(20 * 30).reshape(20, 30)
    region = read_region(data, "yx", selection={"y": (5, 10), "x": slice(0, 4)}, halo=2)
    assert region.window == {"y": (3, 12), "x": (0, 6)}
    assert np.array_equal(region.core, data[5:10, 0:4])
    assert region.halo == {"y": (2, 2), "x": (0, 2)}


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_zarr_manager_reads_a_region_with_a_halo(tmp_path, zarr_format, ngff):
    path = tmp_path / "img.ome.zarr"
    data = _make_template(path, zarr_format, ngff)  # tczyx, scale (2, .5, .5), translation (10, 20, 30)
    z = ZarrManager(str(path))

    region = z.read_region(t=1, c=0, z=(4, 8), y=(10, 20), x=(0, 8), halo=3)
    # z has 8 planes and x starts at 0: the halo is clipped there; t and c get none
    assert region.window == {"t": (1, 2), "c": (0, 1), "z": (1, 8), "y": (7, 23), "x": (0, 11)}
    assert np.array_equal(region.core, data[1:2, 0:1, 4:8, 10:20, 0:8])
    assert np.array_equal(region.data, data[1:2, 0:1, 1:8, 7:23, 0:11])
    assert region.core_translation() == {"z": 18.0, "y": 25.0, "x": 30.0}

    # the same region in micrometers (edges of pixels 10 and 20 in y), with a 1 um halo
    region = z.read_region(t=1, c=0, z=(17.0, 25.0), y=(24.75, 29.75), x=(29.75, 33.75),
                           halo=1.0, physical=True)
    assert region.bounds["z"] == (4, 8) and region.bounds["y"] == (10, 20) and region.bounds["x"] == (0, 8)
    assert region.halo["y"] == (2, 2) and region.halo["z"] == (1, 0)
    assert np.array_equal(region.core, data[1:2, 0:1, 4:8, 10:20, 0:8])
    assert region.physical_bounds()["y"] == (24.75, 29.75)

    with pytest.raises(ValueError, match="outside the image"):
        z.read_region(y=(0, 100))
    with pytest.raises(KeyError, match="not found"):
        z.read_region(group="labels/missing")


@pytest.mark.parametrize("zarr_format,ngff", FORMATS, ids=FORMAT_IDS)
def test_read_region_uses_the_level_scale_and_translation(tmp_path, zarr_format, ngff):
    path = tmp_path / "img.ome.zarr"
    _make_template(path, zarr_format, ngff)
    z = ZarrManager(str(path), mode="r+")
    z.populate_pyramid(num_levels=2, downscale_factor=(1, 2, 2), progress=False)
    t1 = z.metadata["translations"][1]  # mean levels: 20 + (1.0 - 0.5) / 2 = 20.25 in y
    assert t1["y"] == pytest.approx(20.25)

    region = z.read_region(1, y=(t1["y"] + 4.5, t1["y"] + 9.5), physical=True)
    assert region.bounds["y"] == (5, 10)
    assert region.core_translation()["y"] == pytest.approx(t1["y"] + 5 * 1.0)
    assert np.array_equal(region.core, np.asarray(z.data[1])[:, :, :, 5:10, :])
