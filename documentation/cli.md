# Command line

The `zarrsmith` command has four subcommands:

| Command | Does |
|---|---|
| `zarrsmith convert` | convert a dataset to OME-Zarr, or every dataset listed in a CSV file with `--batch` |
| `zarrsmith fuse` | fuse the raw tiles of a mosaic into one OME-Zarr image, or every mosaic listed in a CSV file with `--batch` |
| `zarrsmith chromatic-shift` | measure and correct the Z chromatic shift between the channels of an OME-Zarr image, or of every image listed in a CSV file with `--batch` |
| `zarrsmith resample` | change the pixel size of an OME-Zarr image, or of every image listed in a CSV file with `--batch` |

`zarrsmith -h` lists them, and `zarrsmith convert -h`, `zarrsmith fuse -h`, `zarrsmith chromatic-shift -h` and `zarrsmith resample -h` show every option, examples and the batch CSV columns. `zarrsmith --version` prints the installed version.

Each command runs either one dataset (`-i INPUT -z OUTPUT`) or a CSV file (`--batch FILE.csv`), never both (`chromatic-shift` can also run with `-i` alone, to only measure). In a batch file, one row is one run and the columns take the place of the options.

## Convert a dataset

```console
zarrsmith convert -i INPUT_PATH -m MICROSCOPE -z OUTPUT_ZARR
zarrsmith convert --batch datasets.csv
```

`-m` is one of `luxendo`, `opera`, `viventis`, `zeiss`, `evident`, `zarr`, `zarrv04`, `scape`; leave it out to detect it from the input path. The output options:

| Flag | Meaning | Default |
|---|---|---|
| `-cs`, `--chunk_size` | chunk size, one value per axis of the dataset | chosen from `-ms` |
| `-ms`, `--max_size` | max chunk size in MB (ignored with `-cs`) | 100 |
| `-si`, `--scene_index` | scene of a `.czi` or `.vsi` file | the first scene |
| `-cn`, `--channel_names` / `-cc`, `--channel_colors` | names and colours (hex or matplotlib names) of the channels | those of the input |
| `-zf`, `--zarr_format` | 2 (NGFF 0.4) or 3 (NGFF 0.5) | 3 |
| `-df`, `--downscale_factor` | pyramid factor: one value, or `Z Y X` | 2 on every spatial axis |
| `-nl`, `--num_levels` | number of pyramid levels | until no spatial axis of the coarsest level is over 2048 pixels |
| `-s`, `--subset` | crop or select before writing, e.g. `"y=10:100:2;x=20:80;c=0,2"` | everything |
| `-sh`, `--shards` / `-stm`, `--shard_target_mb` / `-sea`, `--shard_exclude_axes` | sharding (see [Advanced writing options](advanced.md)) | `auto`, 1024 MB, `t c` |
| `-cp`, `--compressor` | `blosc`, `gzip` (zarr v2 only) or `none`; see [Compression](advanced.md#compression) | `blosc` |
| `-cpl`, `--compressor_level` | compression level, 0 to 9 | 1 |
| `-ds`, `--drop_singleton` | drop size-1 `t`/`c`/`z` axes | true |
| `-nw`, `--num_workers` | write threads | half the cores, at most 8 |
| `-ow`, `--overwrite` | replace an existing output (see [Existing outputs](advanced.md#existing-outputs)) | off |

The batch CSV has the columns `input`, `microscope`, `output`, plus any of `chunk_size`, `max_size(MB)`, `scene_index`, `zarr_format`, `downscale_factor`, `subset`, `channel_colors`, `channel_names`, `num_levels`, `shards`, `shard_target_mb`, `shard_exclude_axes`, `compressor`, `compressor_level`, `drop_singleton`, `num_workers`, `overwrite`. The napari converter writes such rows with *Append to batch CSV*.

An existing non-empty output is refused unless `--overwrite` is given, and only a Zarr store is ever replaced. With `--batch`, overwriting is set per row in the `overwrite` column (`--overwrite` is not accepted); a row whose output is in the way is skipped with a warning, the other rows still run, and the skipped rows are listed at the end. This also applies to `fuse`, `chromatic-shift` and `resample`.

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

## Fuse a mosaic

`fuse` fuses the raw tiles of a mosaic (see [Mosaics](mosaic.md)) into one OME-Zarr image:

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
| `-m`, `--microscope` | `luxendo`, `opera`, `zeiss` | auto-detected for raw Luxendo acquisitions, Opera Harmony exports (`Images/Index.xml`) and Zeiss `.czi` files |
| `-rds`, `--downsample` | downsampling while reading (block mean): one value or `Z Y X` | 1 (full resolution) |
| `-p`, `--projection` | `none` (3D), `max` or `sum` | `none` |
| `-b`, `--blend` | `sigmoid`, `max`, `sum` or `midline` | `sigmoid` |
| `-ss`, `--sigmoid_steepness` | steepness of the `sigmoid` transition, higher is sharper | 4 |
| `-ch`, `--channels` / `-g`, `--groups` | tile channels and groups (e.g. `L R`) to fuse | all |
| `-f`, `--fields` | the field to fuse (groups of touching tiles, `field0`, `field1`, ... listed when the mosaic is loaded). Tiles from several fields are not fused: an error is printed (see [Fields](mosaic.md#fields)) | the first field, unless `-t` is given |
| `-t`, `--tiles` | only fuse these tiles, and fit the output to them: stacks such as `1-x00-y00` (every channel), or `channel/stack` ids | all |
| `-st`, `--skip_tiles` | tiles to leave blank (the output keeps its size): `channel/stack` ids such as `cFos-647/1-x00-y00`, or a stack such as `1-x00-y00` for every channel | none |
| `-tp`, `--timepoints` | timepoints to fuse; more than one gives an output with a `t` axis, see [Timepoints](mosaic.md#timepoints) | all |
| `-rtp`, `--reference_timepoint` | timepoint whose tiles set the geometry of every timepoint, or `all` to place every timepoint at [its own positions](mosaic.md#own-positions-per-timepoint) | the first timepoint |
| `-nff`, `--no_flatfield` | Opera: do not apply Harmony's flat-field correction (see [Opera](mosaic.md#opera-harmony-exports)) | off (the correction is applied) |
| `-ma`, `--midline_axis` | axis along which `sigmoid`/`midline` separate the tile groups: `x`, `y` or `z` | `x` |
| `-od`, `--out_dtype` | output dtype | the input dtype (`float32` with `-b sum`) |
| `-rw`, `--read_workers` | tiles read in parallel processes; not to be confused with `-nw` (write threads) | 4 |
| `-td`, `--temp_dir` | folder for the temporary fused level 0 (a local disk is faster) | next to the output |
| `-ow`, `--overwrite` | replace an existing output, once the fusion has succeeded | off |

Every output option of `convert` is available too, applied to the fused output, with the same defaults. Two read differently for a mosaic: `-cs` is given in `C Z Y X` order (`C Y X` for projections, after `T` for several timepoints), and `-s/--subset` crops the fused output (`z`/`y`/`x` in output voxels, after `-rds`) and selects channels (`c`), e.g. `"z=0:100;y=200:900;c=1"`; timepoints are chosen with `-tp`. `-si` loads one scene of a Zeiss mosaic (default: every scene, one field each), ignored for the others.

The batch CSV has the columns `input`, `microscope`, `output`, plus any of the output columns of `convert --batch` and of the fusion columns `downsample`, `projection`, `blend`, `sigmoid_steepness`, `channels`, `groups`, `fields`, `tiles`, `skip_tiles`, `timepoints`, `reference_timepoint`, `midline_axis`, `out_dtype`, `read_workers`, `temp_dir`, `overwrite`. Leave a cell empty for the default; lists are space separated. A row asking for tiles from several fields is not fused; the other rows still run, and the rows that were not fused are listed at the end. The napari *Mosaic Fusion* widget writes such rows with *Append to batch CSV*.

```text
input,microscope,output,downsample,projection,blend,num_levels,shards
/path/to/acquisition_1,luxendo,/path/to/acq1_3D.ome.zarr,4 8 8,none,sigmoid,3,auto
/path/to/acquisition_1,luxendo,/path/to/acq1_MIP.ome.zarr,1 8 8,max,midline,,none
```

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
| `-rc`, `--reference_channel` | reference channel, as an index or a name | 0 |
| `-tp`, `--timepoint` | timepoint the shift is measured on, an index along the `t` axis; every timepoint gets the same shift | 0 |
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

## Old command names

The commands used to have other names. They still work, with their original options, and print a one-line notice with the new name:

| Old | New |
|---|---|
| `zarrsmith 2zarr` | `zarrsmith convert` |
| `zarrsmith batch2zarr -i FILE.csv` | `zarrsmith convert --batch FILE.csv` |
| `zarrsmith mosaic2zarr` (`-ow true`) | `zarrsmith fuse` (`--overwrite`) |
| `zarrsmith batch-mosaic2zarr -i FILE.csv` | `zarrsmith fuse --batch FILE.csv` |

Batch CSV files written before the rename (with a `read_downsample` column) still run.
