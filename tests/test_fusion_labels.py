"""blend="none": every voxel copied from one tile, for label images."""
from __future__ import annotations

import numpy as np
import pytest

from zarrsmith.mosaic_manager import Tile, fuse_tiles
from zarrsmith.mosaic_manager import fusion

# Above 2**24: a float32 round trip would change them.
A, B = 3_000_000_001, 3_000_000_002


def _fused(tmp_path, name, tiles, **kw):
    zm = fuse_tiles(tiles, tmp_path / f"{name}.zarr", progress=False, **kw)
    return zm, zm.data[0].compute()


def _two_tiles():
    """Two label tiles of 24 voxels along x, overlapping by 8 (x 16..24)."""
    a = np.full((4, 16, 24), A, np.uint32)
    b = np.full((4, 16, 24), B, np.uint32)
    return [Tile(a, (0, 0, 0), (1, 1, 1), "nuclei"), Tile(b, (0, 0, 16), (1, 1, 1), "nuclei")]


def test_none_cuts_the_overlap_in_the_middle_and_keeps_the_values(tmp_path):
    _, out = _fused(tmp_path, "cut", _two_tiles(), blend="none")
    assert out.dtype == np.uint32
    labels = out  # the single channel is dropped: zyx
    assert labels.shape == (4, 16, 40)
    assert (labels[..., :20] == A).all()  # overlap 16..24: middle at 20
    assert (labels[..., 20:] == B).all()


def test_none_is_independent_of_the_regions(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    tiles = [Tile(rng.integers(1, 2**31, size=(6, 40, 40), dtype=np.uint32), (0, 32 * iy, 32 * ix),
                  (1, 1, 1), "nuclei") for iy in range(2) for ix in range(2)]
    kw = dict(blend="none", chunks=(1, 6, 16, 16))
    _, whole = _fused(tmp_path, "whole", tiles, **kw)
    monkeypatch.setattr(fusion, "_REGION_3D", (16, 16))
    monkeypatch.setattr(fusion, "_REGION_VOXELS", 16 * 16 * 2)
    _, small = _fused(tmp_path, "small", tiles, **kw)
    np.testing.assert_array_equal(whole, small)
    # every voxel comes from one tile, unchanged
    assert np.isin(whole, np.concatenate([t.data.ravel() for t in tiles])).all()


def test_none_downsamples_by_sampling_not_averaging(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.integers(1, 1000, size=(4, 32, 32), dtype=np.uint32)
    _, out = _fused(tmp_path, "ds", [Tile(data, (0, 0, 0), (1, 1, 1), "nuclei")], blend="none", ds=(1, 2, 2))
    np.testing.assert_array_equal(out, data[:, ::2, ::2])


def test_fused_labels_are_written_as_a_label_image(tmp_path):
    zm, out = _fused(tmp_path, "labels", _two_tiles(), blend="none", data_type="label", levels=2)
    assert zm.metadata["data_type"] == "label"
    assert fusion.display_windows(tmp_path / "labels.zarr") == []
    coarse = np.unique(zm.data[1].compute())
    assert set(coarse) <= {A, B}


def test_labels_refuse_blending_and_float_dtypes(tmp_path):
    with pytest.raises(ValueError, match="blend='none'"):
        fuse_tiles(_two_tiles(), tmp_path / "a.zarr", data_type="label", progress=False)
    with pytest.raises(ValueError, match="integer dtype"):
        fuse_tiles(_two_tiles(), tmp_path / "b.zarr", blend="none", data_type="label",
                   out_dtype="float32", progress=False)
