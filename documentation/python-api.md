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

## Read a region with a halo

`read_region` reads one region of a level, plus a halo on the spatial axes (clipped at the image edge), e.g. to segment a tile with context and keep only its core:

```python
z = mm.ZarrManager("raw.ome.zarr")
r = z.read_region(1, t=0, c=2, z=(10, 20), y=(512, 1024), x=(0, 512), halo=16)  # level 1, pixels
r.data    # the block read, halo included
r.core    # the region alone (r.data[r.crop])
r.bounds, r.window, r.halo   # pixel bounds of the region and of the block; halo actually read

# bounds and halo in physical units, with the level's scale and translation
r = z.read_region(1, y=(250.0, 500.0), x=(0.0, 250.0), halo=8.0, physical=True)
r.core_translation()  # translation of an image holding only the core
```

Labels segmented from a region can be written as their own label image, placed where the region is and following the image it was read from:

```python
r = z.read_region(1, z=(0, 40), y=(512, 1024), x=(0, 512), halo=16)
labels = segment(r.data)                                  # the block, halo included (or only r.core)
r.to_label_image(labels, "tile_labels.ome.zarr")          # cropped to the core, uint32, marked as labels
r.to_label_image(labels, "tile_labels.ome.zarr", num_levels=1, overwrite=True)  # level 0 only
```

The image sits at `r.core_translation()` with the voxel size of the level read. Its units, time step, chunks (clipped to the labels) and format come from the source image (NGFF 0.4 / zarr v2 for a 0.4 source, else 0.5 / zarr v3; `ngff_version`/`zarr_format` override it). Its pyramid follows the source from the level read on, sampled with `method="nearest"` (or `"mode"`) so ids are never mixed: `zarrsmith.utils.dataset_ops.pyramid_like(metadata, start_level)` gives that number of levels and factor for any image. With a `t` axis in the region, the labels may keep it or hold one timepoint. Other keywords (`shards`, `compressor`, ...) go to the writer.

`group="seg"` or `group="labels/nuclei"` reads another image of the store. The conversions are also available on their own in `zarrsmith.utils.region`: `physical_to_pixel_bounds`, `pixel_to_physical_bounds`, `physical_to_pixels` and `halo_window`. Positions follow NGFF 0.6, where a translation is the centre of the first pixel: pixel `i` covers `[t + (i - 1/2) s, t + (i + 1/2) s)`, and physical bounds take every pixel they overlap.

## Plan tiles for parallel processing

`plan_tiles` cuts the spatial axes of one level into overlapping tiles, and gives the layout of an output that every tile can write to in parallel:

```python
from zarrsmith.utils.allocate import allocate_zarr

raw = mm.ZarrManager("raw.ome.zarr")
plan = raw.plan_tiles(1, tile_size={"y": 1024, "x": 1024}, overlap=64, halo=32)  # z is not split
plan.tiles            # Tile(index, grid, bounds) in level-1 pixels, numbered from 1 in z, y, x order
plan.rows()           # one dict per tile (bounds, tile size, overlap), e.g. for a CSV
plan.neighbors(tile)  # the next tile along z, y and x (None at the end of an axis)
# the tiles CSV of `zarrsmith plan-tiles`: bounds, planned size and overlap, forward neighbours
from zarrsmith.utils.tiling import write_tiles_csv
write_tiles_csv(plan, "tiles.csv", source_id="raw", input_uri="raw.ome.zarr")

allocate_zarr("seg.ome.zarr", template="raw.ome.zarr", data_type="label", dtype="uint32",
              **plan.allocate_kwargs("tzyx"))  # level 1, cropped to the ROI, shard = stride

seg = mm.ZarrManager("seg.ome.zarr", mode="r+")
for tile in plan.tiles:                        # in practice, one job per tile
    core = plan.core(tile)                     # the part this tile writes: up to the next tile's start
    r = raw.read_region(1, c=0, halo=plan.halo, **core)
    labels = segment(r.data)[r.crop]           # segment with context, keep the core
    seg.write_label_region(labels, aligned=True, **plan.write_region(tile))
```

Only z, y and x are tiled. By default (`split="t"`) each write block holds one timepoint, so jobs can be split per tile and per timepoint: each writes `t=slice(k, k + 1)` with `aligned=True` and holds one timepoint of its tile in memory. A job writing every timepoint of its tile writes several whole blocks, which is safe too. `split=""` puts every timepoint of a tile in one block (fewer, larger files, but one job per tile); `split="tc"` also splits channels. `crop={"t": (0, 5)}` keeps part of `t`.

An axis without a tile size is not split: one tile spans the whole region of interest (`roi=`, the whole level by default). Tiles start every `stride = tile_size - overlap` pixels; each tile writes its *core*, from its start to the next tile's start, so cores cover the ROI once. `allocate_kwargs` makes the output's write block (the shard on zarr v3, with inner chunks dividing it; the chunk on zarr v2) equal to the stride on tiled axes and to the whole axis on the others, so each core is whole blocks and `aligned=True` passes. With `physical=True`, `roi`, `tile_size`, `overlap` and `halo` are in physical units. A last tile shorter than the halo (and than 2 pixels) is folded into the one before it.

### Cut only when a tile would not fit

`max_tile_voxels` sets the most voxels one job may read for a tile, halo included, for example what fits in GPU memory. The image stays one tile if it fits; otherwise the axes without a `tile_size` are cut into as few tiles of about equal size as fit:

```python
plan = raw.plan_tiles(0, overlap=64, halo=32, max_tile_voxels=400_000_000)  # e.g. Cellpose 3D on a full A100
plan = raw.plan_tiles(0, overlap=64, halo=32, max_tile_voxels=400_000_000,
                      resize=30 / 15)                                     # Cellpose with diameter=15
plan.max_load_voxels   # largest tile read, halo included
```

- The budget is per timepoint and channel: t and c are never tiled.
- Of every way to cut the axes, the plan with the least area between tiles wins, then the one with the fewest tiles. A short axis such as z usually stays whole, and tiles are compact instead of thin slabs (less overlap, halo and merging).
- Strides are multiples of a power of two: up to the default inner chunk (16 along z, 256 along y and x), and at most an eighth of the stride so rounding wastes little of the budget. `allocate_kwargs` then picks inner chunks that divide the stride.
- A `tile_size` given for an axis is kept and counts towards the budget.
- `resize` is the factor by which the tool enlarges each tile before processing: the budget is checked against `voxels * resize ** ndim`. Cellpose rescales objects to about 30 pixels, so use `30 / diameter`.

## Allocate an image and build its pyramid in place

`allocate_zarr` creates an empty image from a template (what [`zarrsmith allocate`](cli.md) runs), with the template's pyramid levels by default. `populate_pyramid` fills or replaces the levels on disk after `start_level` (default 0, an index or a dataset path), from that level, one task per output shard (chunk on zarr v2). The start level and the levels above it are kept; `num_levels` counts every level from level 0. Without `num_levels` or `downscale_factor`, it keeps the levels the image already has:

```python
from zarrsmith.utils.allocate import allocate_zarr

allocate_zarr("seg.ome.zarr", template="raw.ome.zarr", data_type="label", dtype="uint32",
              crop={"t": (0, 10)}, extra_attributes={"t_range": [0, 10]})
# translation= takes the stored format, {"z": .., "y": .., "x": ..}; axes left out keep the template's origin
# start_level=1 makes level 1 of the template the output's level 0 (crop is then in level-1 pixels)

z = mm.ZarrManager("seg.ome.zarr", mode="r+")
z.write_label_region(labels, aligned=True)
z.populate_pyramid()                              # the levels allocate laid out, from the template
z.populate_pyramid(num_levels=4, method="mode")  # also group="labels/nuclei"
z.populate_pyramid(num_levels=6, start_level=2)   # keep levels 0-2, rebuild 3-5 from level 2
```

`method` is `"mean"`, `"nearest"` or `"mode"`; by default `"nearest"` for labels and `"mean"` otherwise. The lazy `build_pyramid` (what `zarrsmith convert` and the napari Convert widget use; `-me/--method` and *Downsampling* there) takes `method` too, with the same default, as does `fuse_tiles` (`pyramid_method`, which fills the levels in place like `populate_pyramid`). Every pyramid zarrsmith writes, and every image `allocate_zarr` creates, records its method in the `multiscales` `type` and `metadata` fields, as the NGFF spec recommends, and `write_image_region` / `write_label_region` refresh the coarser levels with that method, so data and level translations stay consistent. A store that records no method is refreshed with `"nearest"`, as before.

A sharded level is built one inner chunk at a time, so each task holds one output shard plus one chunk's source window; the number of threads is limited to what fits in memory, sharded or not.

NGFF 0.6 stores can be read (axes from `coordinateSystems`, `scale`/`translation` from a `sequence`), but zarrsmith writes only 0.4 and 0.5: `populate_pyramid` and `create_empty_group` refuse a 0.6 store rather than mix versions, and `allocate_zarr` from a 0.6 template writes 0.5.

## Origins and custom attributes

`metadata["translations"]` holds one position per level as a dict keyed by spatial axis, e.g. `{"z": 0.0, "y": 1200.5, "x": -340.0}` (see [readers](readers.md#position-translation)), written as a `translation` after each level's `scale` and read back from any store that has one. It is `None` when the position is unknown, and then none is written; an all-zero position is not written either, which NGFF reads the same way. Subsetting moves it to the first kept pixel; a crop of a dataset without a position records where it was cut from (its first voxel in the source), so a crop away from voxel 0 always has one. Coarser levels follow the NGFF 0.6 convention (`zarrsmith.utils.translation.pyramid_translations`): `mean`/`mode` levels sit at `t0 + (s_k - s0) / 2`, `nearest` levels at `t0`.

`metadata["extra_attributes"]` holds the group attributes that are not NGFF metadata, such as a pipeline's `t_range`. They are read from a store and written back next to `multiscales` on zarr v2, and next to (not inside) `ome` on zarr v3. Both can be changed with `update_metadata`.

## Display window

Viewers open each channel of an OME-Zarr with its OMERO display window (`window` `start`/`end`) as the contrast. Every write of intensity data (`to_zarr`, `zarrsmith convert`, the napari converter) sets it to the 0.5th-99.9th percentile of the channel's non-zero values, so images do not open nearly black over the whole range of their data type; `min`/`max` keep that range. The window is estimated after writing, from the coarsest level of the output: whole chunks from the centre of the image, about 4 million voxels per channel at up to 3 timepoints, so the cost does not grow with the image, even without a pyramid. A fusion sets the window from the fused values instead. `display_window=False` keeps the whole range.

`ZarrManager` reads the windows into `metadata["channel_windows"]` (one `(start, end)` per channel, `None` for a channel left at the whole range), keeps them through channel subsets and reordering, and writes them again on `to_zarr`. `visualize()` opens the channels at their window, with the contrast slider over the whole range of the data type so it can still be widened (napari-ome-zarr limits the slider to the window). Set them with `update_metadata({"channel_windows": [...]})`.

## Update the metadata of a store in place

`update_metadata` changes the in-memory metadata, which reaches disk with `to_zarr`. On a `ZarrManager`, `in_place=True` also writes the updates into the store's NGFF attributes, without copying any data:

```python
z = mm.ZarrManager("image.ome.zarr", mode="r+")
z.update_metadata({"channel_names": ["GFP", "RFP"], "scales": [(2.0, 0.5, 0.5), (4.0, 1.0, 1.0)]}, in_place=True)
```

It takes the same keys and applies them to the root image, its groups and its labels (`include_groups`, `include_labels`). `channel_windows` sets the display window (contrast) of each channel, e.g. `{"channel_windows": [(110, 1170), None]}`: `None` estimates that channel's window from the stored data (`None` alone: every channel), and label images are skipped. It needs `mode="r+"` (or `"a"`) and an NGFF 0.4 or 0.5 store. Only the entries tied to the updated keys are rewritten: the coordinate transformations (`scales`, `translations`, `time_increment`), the axes units (`units`, `time_increment_unit`), the label, color and display window of the OMERO channels, the `image-label` block (`data_type`) and the custom attributes (`extra_attributes`). Dataset paths, display windows and every other attribute are kept. Every dataset is checked before anything is written, so a refused update changes neither the store nor the manager. A dataset already changed in memory (`subset_dataset`, `build_pyramid`, `reorder_channels`, ...) no longer matches its arrays on disk and is refused: write it with `to_zarr`.

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
