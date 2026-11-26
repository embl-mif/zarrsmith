# Command line

The `zarrsmith` commands, grouped by the stage of a pipeline they belong to (`zarrsmith -h` lists them the same way):

| Stage | Command | Does |
|---|---|---|
| Make an OME-Zarr | `zarrsmith convert` | convert a dataset to OME-Zarr, or every dataset listed in a CSV file with `--batch` |
| | `zarrsmith fuse` | fuse the raw tiles of a microscope mosaic into one OME-Zarr image, or every mosaic listed in a CSV file with `--batch` |
| Correct an OME-Zarr | `zarrsmith resample` | change the pixel size of an OME-Zarr image, or of every image listed in a CSV file with `--batch` |
| | `zarrsmith chromatic-shift` | measure and correct the Z chromatic shift between the channels of an OME-Zarr image, or of every image listed in a CSV file with `--batch` |
| Process an OME-Zarr in tiles | `zarrsmith plan-tiles` | cut one level of an image into overlapping tiles: a CSV of tile bounds and neighbours, for parallel jobs |
| | `zarrsmith allocate` | create an empty OME-Zarr image from a template image, or from an explicit size, to fill tile by tile |
| | `zarrsmith fuse-labels` | stitch the label stores of a segmentation run tile by tile into one label image |
| | `zarrsmith pyramid` | build or rebuild the pyramid levels of an OME-Zarr image in place, from level 0 |
| Segment and measure | `zarrsmith project` | max (or sum, mean) projection along a spatial axis, on every pyramid level |
| | `zarrsmith threshold` | label the objects of one channel: Otsu threshold and connected components |
| | `zarrsmith mask` | zero one channel outside the objects of a label image, and add it as a new channel |
| | `zarrsmith measure` | regionprops of every label against the channels of an image, to a CSV |
| | `zarrsmith filter-labels` | keep the labels whose measurements are within limits, renumbered 1..N |

`zarrsmith <command> -h` shows the options of a command, with examples and the batch CSV columns, and `zarrsmith <command> --help-all` every option. `zarrsmith --version` prints the installed version.

Every command lists its options in the same sections, in the same order, so an option reads the same wherever it appears:

| Section | Holds |
|---|---|
| Input and output | `-i`, `-z`, `--batch` |
| What to read | the reader and the part of the input: `-m`, `--scene_index`, `--subset`; for `fuse` the channels, groups, cameras, fields, tiles and timepoints |
| the command's own sections | e.g. *Fusion* and *Tile positions* for `fuse`, *Estimate* and *Corrected image* for `chromatic-shift`, *Tiles* for `plan-tiles` |
| Pyramid | `-nl`, `-df`, `-me`, and for `fuse`/`fuse-labels` `--infer_downscale_factors` |
| Storage layout | `-zf`, `-cs`, `-ms`, shards, compression, `-ds` |
| Channel metadata | `-cn`, `-cc` |
| Performance | `-nw`, `-rw` |
| Run | `-dr`, `-ow` |

`-h` leaves out the options most runs do not need (most of *Storage layout*, *Channel metadata*, *Performance*, and a few fine-tuning options) and says how many there are; they work all the same, and `--help-all` shows them. A short flag means the same option in every command.

Each command that writes an image runs either one dataset (`-i INPUT -z OUTPUT`) or a CSV file (`--batch FILE.csv`), never both (`chromatic-shift` can also run with `-i` alone, to only measure). In a batch file, one row is one run and the columns take the place of the options.

## Convert a dataset

```console
zarrsmith convert -i INPUT_PATH -m MICROSCOPE -z OUTPUT_ZARR
zarrsmith convert --batch datasets.csv
```

`-m` is one of `luxendo`, `opera`, `viventis`, `zeiss`, `evident`, `zarr`, `zarrv04`, `scape`; leave it out to detect it from the input path. The options (`-h` shows the common ones, `--help-all` all of them):

| Flag | Meaning | Default |
|---|---|---|
| `-cs`, `--chunk_size` | chunk size, one value per axis of the dataset | chosen from `-ms` |
| `-ms`, `--max_size` | max chunk size in MB, uncompressed (ignored with `-cs`). A chunk is what a reader fetches: t and c are 1, and the chunk is about cubic in micrometers, from the voxel size, with at most 32 z planes so that viewing one plane reads at most 32 (32 × 256 × 256 for uint16; 16 × 256 × 512 when the z step is 16 times the pixel size). A voxel size the file does not record counts as unknown: the chunk is then cubic in voxels, up to 32 planes; the shards group them into files of at most 512 MB | about 4 |
| `-si`, `--scene_index` | scene of a `.czi` or `.vsi` file | the first scene |
| `-cn`, `--channel_names` / `-cc`, `--channel_colors` | names and colours (hex or matplotlib names) of the channels | those of the input |
| `-zf`, `--zarr_format` | 2 (NGFF 0.4) or 3 (NGFF 0.5) | 3 |
| `-df`, `--downscale_factor` | pyramid factor: one value, or `Z Y X`. Each level averages blocks of the level above (intensity data) or keeps one pixel per block (labels), as `zarrsmith pyramid` does | 2 on every spatial axis |
| `-nl`, `--num_levels` | number of pyramid levels | until no spatial axis of the coarsest level is over 2048 pixels |
| `-me`, `--method` | downsampling of the pyramid levels: `mean`, `nearest` or `mode` (`mean` is refused for labels); batch column `method` | `mean` for intensity data, `nearest` for labels |
| `-s`, `--subset` | crop or select before writing, e.g. `"y=10:100:2;x=20:80;c=0,2"`; channels also by name, `"c=DAPI,GFP"` (a number is always an index) | everything |
| `-sh`, `--shards` / `-stm`, `--shard_target_mb` / `-sea`, `--shard_exclude_axes` | sharding (see [Advanced writing options](advanced.md)) | `auto`, 512 MB (256 MB for `fuse`), `t c` |
| `-cp`, `--compressor` | `blosc`, `gzip` (zarr v2 only) or `none`; see [Compression](advanced.md#compression) | `blosc` |
| `-cpl`, `--compressor_level` | compression level, 0 to 9 | 1 |
| `-ds`, `--drop_singleton` | drop size-1 `t`/`c`/`z` axes | true |
| `-nw`, `--num_workers` | write threads | half the cores, at most 8 |
| `-ow`, `--overwrite` | replace an existing output (see [Existing outputs](advanced.md#existing-outputs)) | off |
| `-dr`, `--dry_run` | print the output (axes, size, voxel), its chunks and shards (size and count) and the pyramid levels, the recap the napari converter shows, and stop: nothing is written. With `--batch`, for every row. Also for `fuse`, with the fusion workers (camera registration is skipped) | off |

The batch CSV has the columns `input`, `microscope`, `output`, plus any of `chunk_size`, `max_size(MB)`, `scene_index`, `zarr_format`, `downscale_factor`, `subset`, `channel_colors`, `channel_names`, `num_levels`, `shards`, `shard_target_mb`, `shard_exclude_axes`, `compressor`, `compressor_level`, `drop_singleton`, `num_workers`, `overwrite`. The napari converter writes such rows with *Append to batch CSV*.

An existing non-empty output is refused unless `--overwrite` is given, and only a Zarr store is ever replaced. With `--batch`, overwriting is set per row in the `overwrite` column (`--overwrite` is not accepted); a row whose output is in the way is skipped with a warning, the other rows still run, and the skipped rows are listed at the end. This also applies to `fuse`, `chromatic-shift` and `resample`.

## Fuse a mosaic

`fuse` fuses the raw tiles of a mosaic (see [Mosaics](mosaic.md)) into one OME-Zarr image. Label tiles of a segmentation run tile by tile are stitched with [`fuse-labels`](#stitch-label-tiles) instead:

```console
# 3D, downsampled 4 (z) x 8 (y) x 8 (x), smooth (sigmoid) transitions between tiles and illumination sides
zarrsmith fuse -i /path/to/2026-10-14_103957 -z acq_3D.ome.zarr -rds 4 8 8 -nl 3

# 2D maximum projection, downsampled 8 x 8 in y/x, sharp cut between illumination sides
zarrsmith fuse -i /path/to/2026-10-14_103957 -z acq_MIP.ome.zarr -rds 1 8 8 -p max -b midline

# where are the tiles? (overlaps are brighter)
zarrsmith fuse -i /path/to/2026-10-14_103957 -z acq_tiles.ome.zarr -rds 1 8 8 -p max -b sum

# many mosaics from a CSV file
zarrsmith fuse --batch mosaics.csv
```

The fusion options:

| Flag | Meaning | Default |
|---|---|---|
| `-m`, `--microscope` | `luxendo`, `opera`, `zeiss`, `zarr` (a folder of OME-Zarr stores placed at their NGFF translation) | auto-detected for raw Luxendo acquisitions, Opera Harmony exports (`Images/Index.xml`), Zeiss `.czi` files and folders of OME-Zarr stores |
| `-rds`, `--downsample` | downsampling while reading (block mean): one value or `Z Y X` | 1 (full resolution) |
| `-p`, `--projection` | `none` (3D), `max` or `sum` | `none` |
| `-b`, `--blend` | `sigmoid`, `max`, `sum`, `midline` or `none` (no blending, for labels; see [Fusion options](mosaic.md#fusion-options)) | `sigmoid` |
| `-ss`, `--sigmoid_steepness` | steepness of the `sigmoid` transition, higher is sharper | 4 |
| `-ch`, `--channels` / `-gr`, `--groups` | tile channels and groups (e.g. `L R`) to fuse. Quote a name with spaces: `-ch "Alexa 488"` on the command line, `"Alexa 488" Brightfield` in a `--batch` CSV cell (double quotes: Excel drops a leading `'` when it saves) | all |
| `-cam`, `--cameras` | cameras to fuse (MuVi SPIM: `left right`); several are blended into one image, see [MuVi SPIM](mosaic.md#muvi-spim-two-opposing-cameras) | all |
| `-fi`, `--fields` | the field to fuse (groups of touching tiles, `field0`, `field1`, ... listed when the mosaic is loaded). Tiles from several fields are not fused: an error is printed (see [Fields](mosaic.md#fields)) | the first field, unless `-tl` is given |
| `-tl`, `--tiles` | only fuse these tiles, and fit the output to them: stacks such as `1-x00-y00` (every channel), or `channel/stack` ids | all |
| `-st`, `--skip_tiles` | tiles to leave blank (the output keeps its size): `channel/stack` ids such as `cFos-647/1-x00-y00`, or a stack such as `1-x00-y00` for every channel | none |
| `-tp`, `--timepoints` | timepoints to fuse; more than one gives an output with a `t` axis, see [Timepoints](mosaic.md#timepoints) | all |
| `-rtp`, `--reference_timepoint` | timepoint whose tiles set the geometry of every timepoint, or `all` to place every timepoint at [its own positions](mosaic.md#own-positions-per-timepoint) | the first timepoint |
| `-nff`, `--no_flatfield` | Opera: do not apply Harmony's flat-field correction (see [Opera](mosaic.md#opera-harmony-exports)) | off (the correction is applied) |
| `-ma`, `--midline_axis` | axis along which `sigmoid`/`midline` separate the tile groups (or cameras): `x`, `y` or `z` | `z` for several cameras in 3D, `x` otherwise |
| `-tf`, `--transforms` | a [transforms file](mosaic.md#transforms-file) giving the tile positions to use instead of the stage positions | the stage positions |
| `-tfo`, `--transforms_only` | only write `<output>.transforms.json` (every fusion writes it first), fuse nothing | off |
| `-rc`, `--register_cameras` | MuVi SPIM: register the right camera view onto the left one on the images, at the camera handover (z centre of the stack), instead of the stage position alone; see [mosaic.md](mosaic.md#registering-the-camera-views) | off |
| `-rm`, `--register_model` | `rigid` (translation and rotation; the right view is resampled) or `translation` (only moved) | `rigid` |
| `-rch`, `--register_channel` | reference channel of `--register_cameras` (e.g. a nuclear stain): the transform is measured on it and applied to every channel; `all` registers every channel on its own images. Timepoints follow `-rtp`: measured at the reference timepoint and applied to all, or at every timepoint with `-rtp all` | the first channel |
| `-dt`, `--dtype` | output dtype | the input dtype (`float32` with `-b sum`) |
| `-dty`, `--data_type` | `intensity` or `label` (a label image; needs `-b none`, the default for labels, and an integer dtype). For a folder of OME-Zarr stores it also picks which stores are read | `label` when every tile is label data (a folder of label stores), else `intensity` |
| `-idf`, `--infer_downscale_factors` / `-sl`, `--start_level` | make the pyramid follow another OME-Zarr image from a level on: its number of levels below that level and its factor (see [Stitch label tiles](#stitch-label-tiles)); `-nl`/`-df` override it | the convert defaults |
| `-rw`, `--read_workers` | regions fused in parallel processes; 0 chooses from the cores (half of them, at most 16), the number of regions and the free memory, as the napari widget does. Not to be confused with `-nw` (write threads) | 0 |
| `-td`, `--temp_dir` | deprecated and ignored: level 0 is fused directly into the output | |
| `-ow`, `--overwrite` | replace an existing output, once the fusion has succeeded | off |

Every output option of `convert` is available too, applied to the fused output, with the same defaults. Two read differently for a mosaic: `-cs` is given in `C Z Y X` order (`C Y X` for projections, after `T` for several timepoints), and `-s/--subset` crops the fused output (`z`/`y`/`x` in output voxels, after `-rds`) and selects channels (`c`), e.g. `"z=0:100;y=200:900;c=1"`; timepoints are chosen with `-tp`. `-si` loads one scene of a Zeiss mosaic (default: every scene, one field each), ignored for the others.

The batch CSV has the columns `input`, `microscope`, `output`, plus any of the output columns of `convert --batch` and of the fusion columns `downsample`, `projection`, `blend`, `sigmoid_steepness`, `channels`, `groups`, `cameras`, `fields`, `tiles`, `skip_tiles`, `timepoints`, `reference_timepoint`, `midline_axis`, `out_dtype`, `read_workers`, `temp_dir`, `transforms`, `transforms_only`, `register_cameras`, `register_model`, `register_channel`, `overwrite`. Leave a cell empty for the default; lists are space separated. (`label_offsets` and `label_mapping` columns still work; `fuse-labels` is the command for label tiles.) A row asking for tiles from several fields is not fused; the other rows still run, and the rows that were not fused are listed at the end. The napari *Mosaic Fusion* widget writes such rows with *Append to batch CSV*.

```text
input,microscope,output,downsample,projection,blend,num_levels,shards
/path/to/acquisition_1,luxendo,/path/to/acq1_3D.ome.zarr,4 8 8,none,sigmoid,3,auto
/path/to/acquisition_1,luxendo,/path/to/acq1_MIP.ome.zarr,1 8 8,max,midline,,none
```

## Change the pixel size

`resample` writes a copy of an OME-Zarr image with a new pixel size, and rescales the data accordingly. Other formats go through `convert` first:

```bash
zarrsmith resample -i INPUT_ZARR -z OUTPUT_ZARR -ps 1              # the same size on every spatial axis
zarrsmith resample -i INPUT_ZARR -z OUTPUT_ZARR -ps 1 0.5 0.5      # Z Y X
zarrsmith resample --batch images.csv
```

The output has the compression and number of pyramid levels of the input. Its chunks and shards start from those of the input and are adjusted to divide each new level evenly, so they are never larger than the input's and store almost no padding. Image groups and labels are resampled too, unless `-ig false` or `-il false` is given. Images use linear interpolation, with anti-aliasing on the axes that shrink; labels use nearest neighbour.

| Option | Does | Default |
|---|---|---|
| `-ps`, `--pixel_size` | the new pixel size in the units of the dataset: one value, or three in `Z Y X` order | none, required |
| `-f`, `--fast` | read images from the coarsest pyramid level still fine enough, instead of level 0 | off (level 0) |
| `-ig`, `--include_groups` / `-il`, `--include_labels` | also resample image groups and labels (`true` or `false`) | true |
| `-nw`, `--num_workers` | number of write threads | half the cores, at most 8, fewer if memory requires it |
| `-ow`, `--overwrite` | replace an existing output | off |

`--fast` reads much less data when shrinking, but pyramid levels are usually built by skipping pixels, so the result keeps their aliasing. Leave it out for data you will measure. Labels always read the coarsest suitable level, because they have no intensities to average.

Large datasets are written one batch at a time, each as large as the free memory allows, in whole chunks or shards. Memory use therefore depends on the batch size rather than on the number of timepoints or the size of the volume. Only the blocks of one batch run in parallel, so a block as large as the batch (very large shards with little memory) runs on one thread.

The batch CSV has the columns `input`, `output`, `pixel_size`, plus any of `fast`, `include_groups`, `include_labels`, `overwrite`, `num_workers`.

## Correct the chromatic shift

`chromatic-shift` measures the Z offset between the channels of an OME-Zarr image on the sample itself and writes a corrected copy with the same chunks, shards, compression and metadata as the input (see [Chromatic shift correction](chromatic-shift.md)):

```console
# measure, save the report and write the corrected image
zarrsmith chromatic-shift -i brain.ome.zarr -z brain_csc.ome.zarr -r shift.json

# only measure
zarrsmith chromatic-shift -i brain.ome.zarr -r shift.json

# apply known shifts (one per channel), or the shifts of an earlier report
zarrsmith chromatic-shift -i brain.ome.zarr -z brain_csc.ome.zarr -sf 0 104.4
zarrsmith chromatic-shift -i other.ome.zarr -z other_csc.ome.zarr -sf shift.json

# many images from a CSV file
zarrsmith chromatic-shift --batch images.csv
```

| Flag | Meaning | Default |
|---|---|---|
| `-i`, `--input_path` | input OME-Zarr image | none, required (or `--batch`) |
| `-z`, `--zarr_path` | corrected OME-Zarr image | none: only measure |
| `-r`, `--report` | save the estimate report (JSON) | no report |
| `-sf`, `--shifts` | skip the estimate: one shift per channel in the Z unit, or a JSON report | none: the shift is measured |
| `-ref`, `--reference_channel` | reference channel, as an index or a name | 0 |
| `-tp`, `--timepoints` | timepoint the shift is measured on, an index along the `t` axis; every timepoint gets the same shift | 0 |
| `-mxs`, `--max_shift` | largest shift searched, in the Z unit | 10% of the stack depth |
| `-cl`, `--coarse_level` | pyramid level of the coarse estimate | the coarsest level with at least 16 Z planes and 32 pixels in Y and X |
| `-rl`, `--refine_level` | pyramid level of the refinement crops; `none` skips refinement | 0 |
| `-nr`, `--n_regions` | number of refinement crops | 32 |
| `-cy`, `--crop_yx` | crop width in Y and X, in pixels | 64 |
| `-mzn`, `--min_z_normal` | how flat a surface must be to count as Z-facing (1 = flat) | 0.7 |
| `-mc`, `--min_correlation` | crops with a lower correlation peak are left out | 0.2 |
| `-th`, `--threshold` | sample mask threshold on the 0–1 rescaled coarse image | Otsu |
| `-fw`, `--fail_on_warning` | exit with an error, writing only the report, when the estimate has warnings | off |
| `-e`, `--extent` | `union`, `same` or `intersection` | `union` |
| `-o`, `--order` | `1` (linear) or `0` (whole planes, values untouched) | 1 |
| `-ig`, `-il` | also write image subgroups / labels (`true` or `false`) | true |
| `-ow`, `--overwrite` | replace an existing output and report | off |
| `-nw`, `--num_workers` | write threads | half the cores, at most 8 |

In a pipeline (e.g. a Nextflow process), use `-r` to keep the report next to the output and `--fail_on_warning` so that a doubtful estimate (search limit reached, no usable crop) fails the step instead of producing a wrongly corrected image.

The batch CSV has the columns `input`, `output` (empty: only measure), plus any of `report`, `shifts`, `reference_channel`, `timepoint`, `max_shift`, `coarse_level`, `refine_level`, `n_regions`, `crop_yx`, `min_z_normal`, `min_correlation`, `threshold`, `extent`, `order`, `include_groups`, `include_labels`, `overwrite`, `num_workers`, `fail_on_warning`. Leave a cell empty for the default; shifts are space separated.

```text
input,output,report,reference_channel,max_shift
/path/to/a.ome.zarr,/path/to/a_csc.ome.zarr,/path/to/a_csc.json,cFos-647,150
/path/to/b.ome.zarr,,/path/to/b_shift.json,0,
```

## Plan tiles

`plan-tiles` cuts one level of an OME-Zarr image into overlapping tiles for parallel jobs (e.g. a segmentation on a GPU) and writes them to a CSV, one row per tile:

```console
# 512-voxel tiles along every axis, overlapping by 32, each job reading 16 more voxels around its tile
zarrsmith plan-tiles -i image.ome.zarr -oc tiles.csv -ts 512 -ov 32 -ha 16

# tiles of level 1, only cut in y and x
zarrsmith plan-tiles -i image.ome.zarr -oc tiles.csv -ts "y=1024;x=1024" -ov 32 -lv 1

# as large as fit a memory budget: at most 300 M voxels read per tile, halo included
zarrsmith plan-tiles -i image.ome.zarr -oc tiles.csv -mtv 300000000 -ov 16 -ha 20
```

| Option | Does | Default |
|---|---|---|
| `-i`, `--input_path` / `-oc`, `--output_csv` | the image, and the tiles CSV | none, required |
| `-id`, `--source_id` | `source_dataset_id` of the rows; `dataset_id` is `<id>__<tile_id>` | the store name without `.ome.zarr` |
| `-lv`, `--level` | pyramid level to tile; the bounds are in its voxels | 0 |
| `-ts`, `--tile_size` | one value for every spatial axis, or per axis, e.g. `"y=1024;x=1024"` | an axis is left whole |
| `-ov`, `--overlap` / `-ha`, `--halo` | overlap between neighbouring tiles, and the context a job reads around its tile (one value, or per axis) | 0 |
| `-mtv`, `--max_tile_voxels` | most voxels one tile may read, halo included: axes without a tile size are cut to fit it, choosing the tile counts with the least cut area | no limit |
| `-rs`, `--resize` | linear factor by which the job enlarges each tile (e.g. `30 / diameter` for Cellpose): the budget is checked on `voxels * resize ** ndim` | 1 |
| `-roi`, `--roi` | part of the level to tile, e.g. `"z=0:40;y=100:900"` | the whole level |
| `-pu`, `--physical` | sizes, overlap, halo and ROI in physical units (the NGFF translation is the centre of the first voxel) | voxels of `--level` |
| `-dr`, `--dry_run` | print the plan, write nothing | off |

A last tile smaller than the halo (at least 2 voxels) is merged into the one before it. Tiles are numbered from 1, `x` fastest, then `y`, then `z`. The CSV has the columns `dataset_id`, `source_dataset_id`, `tile_id`, `tile_index`, `input_uri`, `x_min` ... `z_max` (half-open), `resolution_level`, `tile_overlap_<ax>` and `tile_size_<ax>` (the planned values: an axis left whole reports its ROI length and overlap 0), and the forward neighbours of each tile, `right_*` (next along `x`), `bottom_*` (along `y`) and `behind_*` (along `z`), as `dataset_id` and `tile_index`, blank at the end of an axis, so each pair of touching tiles is listed once.

## Create an empty image and build its pyramid

`allocate` and `pyramid` split writing an image into three steps, for pipelines where several jobs write tiles of the same output (e.g. a pixel classifier writing probabilities):

```bash
# 1. an empty image with the geometry and pyramid levels of raw.ome.zarr, float32, 3 channels
zarrsmith allocate -t raw.ome.zarr -z prob.ome.zarr --dtype float32 --num_channels 3

# 2. jobs write tiles of level 0, e.g. with ZarrManager(..., mode="r+").write_image_region(tile, ..., aligned=True)

# 3. fill the other levels from level 0 on disk (same levels and factors as raw.ome.zarr)
zarrsmith pyramid -z prob.ome.zarr
```

From a template, the output keeps its axes, level-0 size, dtype, pixel size, units, origin (`translation`), channels, chunks, shards, pyramid levels and zarr format, unless an option changes them. `--start_level 1` makes level 1 of the template the output's level 0 (e.g. to segment at a coarser resolution); `--crop` is then in level-1 pixels. Without a template, `--axes`, `--size` and optionally `--scale`, `--units` and `--time_increment` describe it:

```bash
zarrsmith allocate -z new.ome.zarr --axes tczyx --size 5 2 100 2048 2048 --scale 2 0.5 0.5
```

| Option | Does | Default |
|---|---|---|
| `-t`, `--template` | OME-Zarr image whose root image gives the geometry | none |
| `-z`, `--zarr_path` | the new store | none, required |
| `-dt`, `--dtype` / `-dty`, `--data_type` | dtype, and `intensity` or `label`; a label output drops the template's channel axis unless channels are given | the template's |
| `-cn`, `--channel_names` / `-nc`, `--num_channels` / `-cc`, `--channel_colors` | the channels of the output (adds a `c` axis if needed) | the template's |
| `-cr`, `--crop` | the part of the template to keep, e.g. `"t=0:10;y=512:1536"`; it moves the origin, so the output gets a translation | the whole template |
| `-cs`, `--chunk_size` / `-sh`, `--shards` | level-0 chunks and shards, one value per output axis (`--shards auto` or `none` also work) | the template's; `auto` shards without a template |
| `-sl`, `--start_level` | level of the template (index or path) that becomes the output's level 0; `--crop` is in its pixels | 0 |
| `-nl`, `--num_levels` / `-df`, `--downscale_factor` | levels to create, and the factor between them | the template's levels from `--start_level` on; 1 level without a template |
| `-tr`, `--translation` | without a value: keep the template's origin (moved by `--crop`); with one value per spatial axis: the level-0 origin. An all-zero origin is not written, which NGFF reads the same way | written when the template has one, the crop moves the origin, or the coarser levels are averaged (intensity data with more than one level), from an origin of 0 when the template has none |
| `-zf`, `--zarr_format` | 2 (NGFF 0.4) or 3 (NGFF 0.5) | the template's, else 3 |
| `-at`, `--attrs` | custom group attributes, as JSON or a `.json` file, e.g. `'{"t_range": [0, 10]}'` | none |
| `-cp`, `--compressor` / `-cpl`, `--compressor_level` | codec | blosc, level 1 |
| `-ow`, `--overwrite` | replace an existing output | off |

Zarr v2 has no shards: an explicit `--shards` shape is refused, `--shards auto` is dropped with a warning, and a v3 template's shards are not carried over. Parallel jobs must write tiles that start and end on the shard grid (the chunk grid on zarr v2 or without shards), so no two jobs write the same file; `aligned=True` raises an error for any other tile.

`pyramid` creates or replaces the levels after `--start_level` (default 0) of the root image (or of `--group labels/nuclei`, or of every image with `--group all`). The start level and the levels above it are kept as they are. Each new level is computed from the level above it, one task per output shard (chunk on zarr v2), with the zarr format, codecs and chunk layout of level 0, and the `multiscales` metadata is updated. Running it again replaces the levels cleanly. `zarrsmith pyramid -z img.ome.zarr --start_level 2 --num_levels 6` keeps levels 0-2 and rebuilds 3-5 from level 2.

| Option | Does | Default |
|---|---|---|
| `-z`, `--zarr_path` | the store to update | none, required |
| `-g`, `--group` | image(s) to process: group paths, or `all` | the root image |
| `-nl`, `--num_levels` | total number of levels, level 0 included | the levels the image already has (e.g. allocated from a template); with one level, until no spatial axis is above 2048 pixels |
| `-df`, `--downscale_factor` | one value, or one per spatial axis (`1 2 2` keeps Z) | the factor between the existing levels, else 2 |
| `-me`, `--method` | `mean` (average), `nearest` (one pixel per block) or `mode` (most frequent value, smoother label pyramids) | `mean` for intensity data, `nearest` for labels |
| `-sl`, `--start_level` | level to build from, by index or path; it and the levels above it are kept | 0 |
| `-nw`, `--num_workers` | number of threads | half the cores, at most 8 |

Translations follow the NGFF 0.6 convention, where a pixel's position is its centre: with `mean` and `mode` a coarse pixel sits at the centre of the block it summarizes, so level `k` is at `t_s + (s_k - s_s) / 2` (`s_k` its pixel size, `s` the start level); with `nearest` it keeps the origin of the start level.

## Stitch label tiles

`fuse-labels` stitches the label stores of a segmentation run tile by tile (one OME-Zarr store per tile, each at its own NGFF translation) into one label image:

```console
zarrsmith fuse-labels -i tiles/ -z merged.ome.zarr -lo offsets.csv -lm mapping.json

# the pyramid of the labels follows the segmented image from level 1 on (it was segmented at level 1)
zarrsmith fuse-labels -i tiles/ -z merged.ome.zarr -lo offsets.csv -lm mapping.json \
    --infer_downscale_factors image.ome.zarr --start_level 1
```

| Option | Does | Default |
|---|---|---|
| `-i`, `--input_path` / `-z`, `--zarr_path` | the folder of tile stores, and the merged image | none, required |
| `-lo`, `--label_offsets` | CSV with the columns `store` (store name, with or without `.ome.zarr`) and `offset`, added to every non-zero id of that store; every store needs one | none |
| `-lm`, `--label_mapping` | JSON `{"id": new_id}` applied after the offsets, e.g. the objects a tile edge cut in two, merged by a label solver; ids not listed are kept | none |
| `-dt`, `--dtype` | output dtype, an integer type | `uint32` |
| `--tiles_csv` / `--label_offset_step` | instead of `-lo`: a CSV with the columns `tile_index` and `segmented_tile_path` (or `store`) of every tile, e.g. the tiles CSV of `plan-tiles` with the segmented store of each tile; each store gets the offset `tile_index * step` (written next to the output as `<name>.label_offsets.csv`) | |
| `-idf`, `--infer_downscale_factors` / `-sl`, `--start_level` | make the pyramid follow this image from that level on: the same number of levels below it and the same factor, so level `k` of the labels matches level `start + k` of the image; `-nl`/`-df` override it | the convert defaults |

The storage, performance and run options of `fuse` apply too (`--help-all`). Labels are never blended: every voxel is copied from one tile, overlaps are cut in their middle, and the pyramid levels sample voxels instead of averaging them. Each tile is given only the mapping entries between its offset and the next larger one, so the ids of a tile must stay below the next tile's offset (see [Renumbering label tiles](mosaic.md#renumbering-label-tiles)). Without `--label_offsets` and `--label_mapping` the ids must already agree across tiles, and only stores marked as labels are read.

## Segment and measure

These commands derive images and tables from an OME-Zarr image, keeping its pyramid level for level, so their outputs line up with it and with each other. An output is written in the format of its input (`-zf` changes it) and is not replaced without `--overwrite`.

```console
zarrsmith project -i stack.ome.zarr -z mip.ome.zarr                              # max over z, every level
zarrsmith threshold -i mip.ome.zarr -z nuclei.ome.zarr -c DAPI -lv 1 -fa 1.1     # Otsu x 1.1 at level 1
zarrsmith measure -i mip.ome.zarr -l nuclei.ome.zarr -oc features.csv
zarrsmith filter-labels -l nuclei.ome.zarr -tb features.csv -z kept.ome.zarr -oc kept.csv --min_area 50 \
    --filters "solidity:0.8:"
zarrsmith mask -i mip.ome.zarr -l kept.ome.zarr -c DAPI -z mip_clean.ome.zarr    # adds DAPI_clean
```

| Command | Options | Default |
|---|---|---|
| `project` | `-p`, `--projection` `max`, `sum` or `mean`; `--axis` the spatial axis to remove (also from the scales, units and translations) | `max`, `z` |
| `threshold` | `-c`, `--channel` (index or name); `-lv`, `--level` where the threshold and the labelling are made, the other levels being that labelling resized with nearest neighbour, so an object keeps its id on every level; `-fa`, `--factor` on the Otsu threshold; `-dt`, `--dtype`; `--levels all` (every level of the image, the finer ones upsampled) or `from_level` | level 0, factor 1, `uint32`, `all` |
| `mask` | `-l`, `--labels` (same pyramid as the image; a level rounded differently is cropped or padded); `-c`, `--channel`; `--suffix` of the new channel's name; `--masked_only` writes only the masked channel | `_clean`, every channel plus the masked one |
| `measure` | `-l`, `--labels`; `-oc`, `--output_csv`; `-ch`, `--channels` (indices or names); `-lv`/`-llv` the levels of the image and of the labels; `--properties` / `--intensity_properties` extra regionprops properties; `--dataset_id` a first column | every channel, level 0 |
| `filter-labels` | `-l`, `--labels`; `-tb`, `--table` (a CSV with a `label` column, and `timepoint` for labels with a `t` axis); `-z` and `-oc` the filtered labels and the kept rows (with `original_label`); `--min_area`/`--max_area`; `--filters "property:min:max;..."` on any column; `--keep_ids` keeps the ids instead of renumbering | renumbered 1..N per timepoint |

`measure` writes one row per label (and timepoint): `label`, `area`, `bbox-*`, `centroid-*`, `solidity`, `extent` (2D also `perimeter`, `orientation`, `eccentricity`, axis lengths and `form_factor`), and per channel `mean_intensity`, `min_intensity`, `max_intensity` as `<property>_ch<i>_<name>`. A `t` of 1 on the image alone is dropped.

## Old command names and flags

The commands used to have other names. They still work, with their original options, and print a one-line notice with the new name:

| Old | New |
|---|---|
| `zarrsmith 2zarr` | `zarrsmith convert` |
| `zarrsmith batch2zarr -i FILE.csv` | `zarrsmith convert --batch FILE.csv` |
| `zarrsmith mosaic2zarr` (`-ow true`) | `zarrsmith fuse` (`--overwrite`) |
| `zarrsmith batch-mosaic2zarr -i FILE.csv` | `zarrsmith fuse --batch FILE.csv` |

Batch CSV files written before the rename (with a `read_downsample` column) still run.

Some short flags changed so that each means the same option in every command. The old spellings still work for now and print a note with the new one:

| Command | Old | New |
|---|---|---|
| `fuse` | `-t` (tiles), `-f` (fields), `-g` (groups) | `-tl`, `-fi`, `-gr` |
| `fuse` | `-od`, `--out_dtype` | `-dt`, `--dtype` (`-dt label` / `-dt intensity` still set the data type) |
| `fuse` | `-dt`, `--data_type` | `-dty`, `--data_type` |
| `fuse` | `-lo`, `-lm` | `zarrsmith fuse-labels -lo ... -lm ...` |
| `chromatic-shift` | `-rc`, `--timepoint` | `-ref`, `--timepoints` |
| `allocate` | `-o`, `--out` | `-z`, `--zarr_path` |

Long option names did not change, so batch CSV columns are the same.
