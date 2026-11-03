# Command line

The `zarrsmith` command has four subcommands:

| Command | Does |
|---|---|
| `zarrsmith convert` | convert a dataset to OME-Zarr, or every dataset listed in a CSV file with `--batch` |
| `zarrsmith fuse` | fuse the raw tiles of a mosaic into one OME-Zarr image, or every mosaic listed in a CSV file with `--batch` |
| `zarrsmith register` | measure the tile positions of a mosaic from the image data, for `fuse -reg`, or of every mosaic listed in a CSV file with `--batch` |
| `zarrsmith chromatic-shift` | measure and correct the Z chromatic shift between the channels of an OME-Zarr image, or of every image listed in a CSV file with `--batch` |

`zarrsmith -h` lists them, and `zarrsmith convert -h`, `zarrsmith fuse -h`, `zarrsmith register -h` and `zarrsmith chromatic-shift -h` show every option, examples and the batch CSV columns. `zarrsmith --version` prints the installed version.

Each command runs either one dataset (`-i INPUT -z OUTPUT`; for `register`, `-i INPUT -r FILE`) or a CSV file (`--batch FILE.csv`), never both (`chromatic-shift` can also run with `-i` alone, to only measure). In a batch file, one row is one run and the columns take the place of the options.

## Convert a dataset

```console
zarrsmith convert -i INPUT_PATH -m MICROSCOPE -z OUTPUT_ZARR
zarrsmith convert --batch datasets.csv
```

`-m` is one of `luxendo`, `opera`, `viventis`, `zeiss`, `evident`, `zarr`, `zarrv04`, `scape`; leave it out to detect it from the input path. The output options:

| Flag | Meaning |
|---|---|
| `-cs`, `--chunk_size` | chunk size, one value per axis of the dataset |
| `-ms`, `--max_size` | max chunk size in MB (ignored with `-cs`) |
| `-si`, `--scene_index` | scene of a `.czi` or `.vsi` file (default: the first scene) |
| `-cn`, `--channel_names` / `-cc`, `--channel_colors` | names and colours (hex or matplotlib names) of the channels |
| `-zf`, `--zarr_format` | 2 (NGFF 0.4) or 3 (NGFF 0.5, default) |
| `-df`, `--downscale_factor` / `-nl`, `--num_levels` | pyramid factor and number of levels (estimated if omitted) |
| `-s`, `--subset` | crop or select before writing, e.g. `"y=10:100:2;x=20:80;c=0,2"` |
| `-sh`, `-stm`, `-sea` | sharding (see [Advanced writing options](advanced.md)) |
| `-ds`, `--drop_singleton` | drop size-1 `t`/`c`/`z` axes (default true) |
| `-nw`, `--num_workers` | write threads |
| `-ow`, `--overwrite` | replace an existing output (see [Existing outputs](advanced.md#existing-outputs)) |

The batch CSV has the columns `input`, `microscope`, `output`, plus any of `chunk_size`, `max_size(MB)`, `scene_index`, `zarr_format`, `downscale_factor`, `subset`, `channel_colors`, `channel_names`, `num_levels`, `shards`, `shard_target_mb`, `shard_exclude_axes`, `drop_singleton`, `num_workers`, `overwrite`. The napari converter writes such rows with *Append to batch CSV*.

An existing non-empty output is refused unless `--overwrite` is given, and only a Zarr store is ever replaced. With `--batch`, overwriting is set per row in the `overwrite` column (`--overwrite` is not accepted); a row whose output is in the way is skipped with a warning, the other rows still run, and the skipped rows are listed at the end. This also applies to `fuse` and `chromatic-shift`.

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

| Flag | Meaning |
|---|---|
| `-m`, `--microscope` | `luxendo`, `opera`, `zeiss`; auto-detected for raw Luxendo acquisitions, Opera Harmony exports (`Images/Index.xml`) and Zeiss `.czi` files |
| `-rds`, `--downsample` | downsampling while reading (block mean): one value or `Z Y X` (default 1) |
| `-p`, `--projection` | `none` (3D, default), `max` or `sum` |
| `-b`, `--blend` | `sigmoid` (default), `max`, `sum` or `midline` |
| `-ss`, `--sigmoid_steepness` | steepness of the `sigmoid` transition, higher is sharper (default 4) |
| `-ch`, `--channels` / `-g`, `--groups` | tile channels and groups (e.g. `L R`) to fuse |
| `-f`, `--fields` | the field to fuse (groups of touching tiles, `field0`, `field1`, ... listed when the mosaic is loaded); default: the first field. Tiles from several fields are not fused: an error is printed (see [Fields](mosaic.md#fields)) |
| `-t`, `--tiles` | only fuse these tiles, and fit the output to them: stacks such as `1-x00-y00` (every channel), or `channel/stack` ids |
| `-st`, `--skip_tiles` | tiles to leave blank (the output keeps its size): `channel/stack` ids such as `cFos-647/1-x00-y00`, or a stack such as `1-x00-y00` for every channel |
| `-tp`, `--timepoints` | timepoints to fuse (default: all); more than one gives an output with a `t` axis, see [Timepoints](mosaic.md#timepoints) |
| `-rtp`, `--reference_timepoint` | timepoint whose tiles set the geometry of every timepoint (default: the first) |
| `-reg`, `--registration` | file made by [`zarrsmith register`](#register-a-mosaic): place the tiles at the positions it found (default: the stage positions). If it cannot be used, nothing is fused |
| `-nff`, `--no_flatfield` | Opera: do not apply Harmony's flat-field correction (applied by default, see [Opera](mosaic.md#opera-harmony-exports)) |
| `-ma`, `--midline_axis` | axis along which `sigmoid`/`midline` separate the tile groups: `x` (default), `y` or `z` |
| `-od`, `--out_dtype` | output dtype |
| `-rw`, `--read_workers` | tiles read in parallel processes (default 4); not to be confused with `-nw` (write threads) |
| `-td`, `--temp_dir` | folder for the temporary fused level 0 (a local disk is faster) |
| `-ow`, `--overwrite` | replace an existing output, once the fusion has succeeded |

Every output option of `convert` is available too, applied to the fused output. Two read differently for a mosaic: `-cs` is given in `C Z Y X` order (`C Y X` for projections, after `T` for several timepoints), and `-s/--subset` crops the fused output (`z`/`y`/`x` in output voxels, after `-rds`) and selects channels (`c`), e.g. `"z=0:100;y=200:900;c=1"`; timepoints are chosen with `-tp`. `-si` loads one scene of a Zeiss mosaic, ignored for the others.

The batch CSV has the columns `input`, `microscope`, `output`, plus any of the output columns of `convert --batch` and of the fusion columns `downsample`, `projection`, `blend`, `sigmoid_steepness`, `channels`, `groups`, `fields`, `tiles`, `skip_tiles`, `timepoints`, `reference_timepoint`, `registration`, `midline_axis`, `out_dtype`, `read_workers`, `temp_dir`, `overwrite`. Leave a cell empty for the default; lists are space separated. A row asking for tiles from several fields, or with a registration that cannot be used, is not fused; the other rows still run, and the rows that were not fused are listed at the end. The napari *Mosaic Fusion* widget writes such rows with *Append to batch CSV*.

```text
input,microscope,output,downsample,projection,blend,num_levels,shards
/path/to/acquisition_1,luxendo,/path/to/acq1_3D.ome.zarr,4 8 8,none,sigmoid,3,auto
/path/to/acquisition_1,luxendo,/path/to/acq1_MIP.ome.zarr,1 8 8,max,midline,,none
```

## Register a mosaic

`register` measures the errors of the stage positions of a mosaic from the image data and saves them to a file that `fuse -reg` uses (see [Content-based registration](mosaic.md#content-based-registration)):

```console
# measure on one channel: the result holds for every channel and timepoint (only the z offset
# between tile groups, e.g. left and right illumination, is measured per channel)
zarrsmith register -i /path/to/2026-10-14_103957 -r registration.json -rc autofluo-488

# fuse with these positions, as many times as needed
zarrsmith fuse -i /path/to/2026-10-14_103957 -z acq_3D.ome.zarr -rds 4 8 8 -reg registration.json

# measure the pairs that were rejected again, reading their whole overlaps
zarrsmith register -i /path/to/2026-10-14_103957 -r registration.json --retry_rejected -bm none

# many mosaics from a CSV file
zarrsmith register --batch mosaics.csv
```

The file is written after every pair: if a run stops, run the same command again and it continues. A file with other positions, another reference or another search range is refused.

| Flag | Meaning |
|---|---|
| `-i`, `--input_path` / `-r`, `--registration` | the acquisition, and the registration file (`.json`) |
| `-m`, `--microscope` | `luxendo`, `opera`, `zeiss`; auto-detected as for `fuse` |
| `-rc`, `--reference_channel` | channel to match, ideally with structure in the overlaps (default: the first) |
| `-rtp`, `--reference_timepoint` | timepoint to match (default: the first) |
| `-f`, `--fields` | fields to register (default: all) |
| `-t`, `--tiles` / `-st`, `--skip_tiles` | only register these tiles / do not read these tiles, as for `fuse` |
| `-ms`, `--max_shift` | largest error searched between two tiles, as a fraction of the tile size: one value or `Z Y X` (default `0.02 0.06 0.06`). Reads grow with it |
| `-bm`, `--budget_mb` | most raw data read per pair of tiles, in MB (default 1024); `none` reads whole overlaps |
| `--min_ncc` | lowest match score that is trusted (default 0.1) |
| `--max_residual` | links further than this from the solution, in working voxels, are dropped (default 2) |
| `--retry_rejected` | with an existing file: measure again the pairs that were rejected |
| `-rw`, `--read_workers` | pairs measured in parallel (default 4) |
| `-si`, `--scene_index` / `-nff`, `--no_flatfield` | as for `fuse` |

The batch CSV has the columns `input`, `microscope`, `registration`, plus any of `scene_index`, `reference_channel`, `reference_timepoint`, `fields`, `tiles`, `skip_tiles`, `max_shift`, `budget_mb`, `min_ncc`, `max_residual`, `retry_rejected`, `read_workers`, `flatfield`. Leave a cell empty for the default; lists are space separated. A row that fails is reported and the other rows still run.

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

| Flag | Meaning |
|---|---|
| `-i`, `--input_path` | input OME-Zarr image |
| `-z`, `--zarr_path` | corrected OME-Zarr image; leave out to only measure |
| `-r`, `--report` | save the estimate report (JSON) |
| `-sf`, `--shifts` | skip the estimate: one shift per channel in the Z unit, or a JSON report |
| `-rc`, `--reference_channel` | reference channel, as an index or a name (default 0) |
| `-tp`, `--timepoint` | timepoint the shift is measured on, an index along the `t` axis (default 0); every timepoint gets the same shift |
| `-mxs`, `--max_shift` | largest shift searched, in the Z unit (default: 10% of the stack depth) |
| `-cl`, `-rl`, `-nr`, `-cy`, `-mzn`, `-mc`, `-th` | coarse level, refine level (`none` skips refinement), number of crops, crop width, `min_z_normal`, `min_correlation`, mask threshold |
| `-fw`, `--fail_on_warning` | exit with an error, writing only the report, when the estimate has warnings |
| `-e`, `--extent` | `union` (default), `same` or `intersection` |
| `-o`, `--order` | `1` (linear, default) or `0` (whole planes, values untouched) |
| `-ig`, `-il` | also write image subgroups / labels (`true` or `false`, default true) |
| `-ow`, `--overwrite` | replace an existing output and report |
| `-nw`, `--num_workers` | write threads |

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
