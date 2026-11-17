# Python API

Typical Python usage of the managers and of the zarr writing helpers.

## Read, build a pyramid and export

```python
import zarrsmith.microscope_manager as mm

# Read a source dataset
source = mm.ViventisManager("path/to/Position_1")

# Build a pyramid in memory
source.build_pyramid(num_levels=3)

# Export to OME-Zarr (default: NGFF v0.5 / zarr v3)
# An existing output is refused; see "Existing outputs" in Advanced writing options
source.to_zarr("output.zarr")

# Re-open the written zarr dataset
z = mm.ZarrManager("output.zarr")
viewer = z.visualize(start_level=0, in_memory=False)
```

## Create an empty zarr dataset from metadata

```python
import zarrsmith.microscope_manager as mm

z = mm.ZarrManager(
    "empty.zarr",
    mode="a",
    metadata={
        "size": [(1, 2, 16, 256, 256)],
        "chunksize": [(1, 1, 16, 128, 128)],
        "scales": [(2.0, 0.5, 0.5)],
        "units": ("micrometer", "micrometer", "micrometer"),
        "axes": "tczyx",
        "channel_names": ["GFP", "RFP"],
        "channel_colors": ["00FF00", "FF0000"],
        "time_increment": 1.0,
        "time_increment_unit": "second",
        "dtype": "uint16",
    },
)
```

## Axis-aware ZarrManager metadata

Most microscope-specific managers still normalize data to legacy `tczyx`, but `ZarrManager` and `ArrayManager` can now write and read any unique subset of the image axes `t`, `c`, `z`, `y`, and `x`. 

> [!NOTE]
> The axes string must have one label per array dimension and may not contain non-image axes such as tiles, ROIs, scenes, or wells.

For example, a two-dimensional YX intensity image can be written as:

```python
import dask.array as da
import numpy as np
import zarrsmith.microscope_manager as mm

img = np.zeros((512, 512), dtype=np.uint16)
levels = [da.from_array(img, chunks=(256, 256))]
metadata = {
    "axes": "yx",
    "size": [(512, 512)],
    "chunksize": [(256, 256)],
    "scales": [(0.5, 0.5)],       # follows the spatial axes in axes order
    "units": ("micrometer", "micrometer"),
    "dtype": "uint16",
    "data_type": "intensity",    # or "label"
}

mm.ArrayManager(levels, metadata).to_zarr("yx_image.zarr")
```

Use `data_type="label"` for segmentation, mask, or annotation data. Label arrays must use an integer dtype. Label groups get the standard NGFF `image-label` block (pointing to the source image when the label is nested under `<image>/labels/<name>`) and are listed on the `labels` group; that is how readers tell labels from intensity images. The NGFF metadata is written with [ome-zarr-py](https://github.com/ome/ome-zarr-py): under `attrs["ome"]` for NGFF v0.5/Zarr v3, whose arrays also carry `dimension_names` in their `zarr.json`, and directly on the group attributes for NGFF v0.4/Zarr v2.

Legacy calls that create labels from an intensity image metadata dictionary still work:

```python
z = mm.ZarrManager("image.zarr", mode="a")
z.create_empty_group("nuclei", image_metadata, is_label=True)
```

When `image_metadata["axes"] == "tczyx"`, this legacy `is_label=True` form creates a `tzyx` label group by dropping the intensity channel axis. New callers that really want channelled label data can pass explicit label metadata instead:

```python
label_metadata = {**image_metadata, "data_type": "label"}
z.create_empty_group("classes", label_metadata, data_type="label")
```

## Update a region in an existing zarr image

```python
import numpy as np
import zarrsmith.microscope_manager as mm

z = mm.ZarrManager("output.zarr", mode="a")
patch = np.full((1, 1, 2, 64, 64), 999, dtype=np.uint16)

z.write_image_region(
    patch,
    t=slice(0, 1),
    c=slice(0, 1),
    z=slice(10, 12),
    y=slice(100, 164),
    x=slice(100, 164),
    level=0,
)
```

With `aligned=True`, `write_image_region` and `write_label_region` raise `ValueError` unless the region starts and ends on the shard grid (the chunk grid on zarr v2 or without shards) of every level they write. Only such regions are safe to write from several processes at once.

## Allocate an image and build its pyramid in place

`allocate_zarr` creates an empty image from a template (what [`zarrsmith allocate`](cli.md) runs), and `populate_pyramid` creates or replaces the levels on disk after `start_level` (default 0, an index or a dataset path), from that level, one task per output shard (chunk on zarr v2). The start level and the levels above it are kept; `num_levels` counts every level from level 0:

```python
from zarrsmith.utils.allocate import allocate_zarr

allocate_zarr("seg.ome.zarr", template="raw.ome.zarr", data_type="label", dtype="uint32",
              crop={"t": (0, 10)}, extra_attributes={"t_range": [0, 10]})
# translation= takes the stored format, {"z": .., "y": .., "x": ..}; axes left out keep the template's origin

z = mm.ZarrManager("seg.ome.zarr", mode="r+")
z.write_label_region(labels, aligned=True)
z.populate_pyramid(num_levels=4, method="mode")  # also group="labels/nuclei"
z.populate_pyramid(num_levels=6, start_level=2)   # keep levels 0-2, rebuild 3-5 from level 2
```

`method` is `"mean"`, `"nearest"` or `"mode"`; by default `"nearest"` for labels and `"mean"` otherwise. The lazy `build_pyramid` also takes `method` (default `"nearest"`). The method is recorded in the `multiscales` `type` and `metadata` fields, as the NGFF spec recommends.

A sharded level is built one inner chunk at a time, so each task holds one output shard plus one chunk's source window; the number of threads is limited to what fits in memory, sharded or not.

NGFF 0.6 stores can be read (axes from `coordinateSystems`, `scale`/`translation` from a `sequence`), but zarrsmith writes only 0.4 and 0.5: `populate_pyramid` and `create_empty_group` refuse a 0.6 store rather than mix versions, and `allocate_zarr` from a 0.6 template writes 0.5.

## Origins and custom attributes

`metadata["translations"]` holds one position per level as a dict keyed by spatial axis, e.g. `{"z": 0.0, "y": 1200.5, "x": -340.0}` (see [readers](readers.md#position-translation)), written as a `translation` after each level's `scale` and read back from any store that has one. It is `None` when the position is unknown, and then none is written; an all-zero position is not written either, which NGFF reads the same way. Subsetting moves it to the first kept pixel. Coarser levels follow the NGFF 0.6 convention (`zarrsmith.utils.translation.pyramid_translations`): `mean`/`mode` levels sit at `t0 + (s_k - s0) / 2`, `nearest` levels at `t0`.

`metadata["extra_attributes"]` holds the group attributes that are not NGFF metadata, such as a pipeline's `t_range`. They are read from a store and written back next to `multiscales` on zarr v2, and next to (not inside) `ome` on zarr v3. Both can be changed with `update_metadata`.

## Change the pixel size

`resample_dataset` gives a dataset a new pixel size and rescales the data accordingly. Nothing is computed until the dataset is written, and it works with every manager:

```python
z = mm.ZarrManager("input.zarr")
z.resample_dataset({"z": 1.0, "y": 0.5, "x": 0.5})  # one value per axis; omitted axes are kept
z.to_zarr("resampled.zarr")
```

Images use linear interpolation, with anti-aliasing on the axes that shrink; labels use nearest neighbour. The pyramid is rebuilt with the same number of levels. `fast=True` reads images from the coarsest pyramid level still fine enough instead of level 0: much less data, but the result keeps the aliasing of that level.

For large stores, `resample_to_zarr` computes one batch at a time, keeps the compression of the input, and fits the chunks and shards to each level. It is what [`zarrsmith resample`](cli.md) runs:

```python
z = mm.ZarrManager("input.zarr")
resampled = z.resample_to_zarr("resampled.zarr", {"z": 1.0, "y": 0.5, "x": 0.5})
```

## Correct the chromatic shift between channels

`ZarrManager` can measure the axial (Z) offset between the channels of a light-sheet dataset on the sample itself and write a corrected copy with the same structure as the input:

```python
z = mm.ZarrManager("input.zarr")
out = z.correct_chromatic_shift("corrected.zarr", reference_channel=0, timepoint=0)
```

See [Chromatic shift correction](chromatic-shift.md) for how the shift is measured, the options and how to check the result.
