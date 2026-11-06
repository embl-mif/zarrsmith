# Advanced writing options

These options control how the OME-Zarr output is laid out and how fast it is written. They are available in the Python API (`to_zarr(...)` keyword arguments), the CLI (flags and batch CSV columns) and the napari converter (the *Additional parameters* panel).

## Existing outputs

ZarrSmith does not replace existing data unless you ask for it. The same rules apply to `to_zarr`, `fuse`, the chromatic shift correction, the CLI and the napari widgets:

| What is at the output path | Default | With `overwrite=True` |
|---|---|---|
| nothing, or an empty folder | written | written |
| a Zarr store | refused (`FileExistsError`) | deleted and replaced, with a warning |
| a file, or a non-empty folder that is not a Zarr store | refused | refused |
| the input of this write, or a folder containing it | refused | refused |

```python
dataset.to_zarr("out.zarr")                   # FileExistsError if out.zarr already holds data
dataset.to_zarr("out.zarr", overwrite=True)   # replaces the store, with a warning
```

Replacing a store deletes everything in that folder. The input can never be the output because reading is lazy: `ZarrManager("a.zarr").to_zarr("a.zarr")` would delete the data it is about to read. To change a store in place, use `write_image_region`, `write_label_region` or `create_empty_group` instead.

A warning is also given when the output name does not end in `.zarr` or `.ome.zarr`. The napari widgets refuse such a name.

| Python / CSV column | CLI flag | napari control | Default |
|---|---|---|---|
| `shards` | `-sh`, `--shards` | Sharding (zarr v3 only) | `auto` |
| `shard_target_mb` | `-stm`, `--shard_target_mb` | Shard target (MB) | 1024 |
| `shard_exclude_axes` | `-sea`, `--shard_exclude_axes` | Never merge axes | `t c` |
| `drop_singleton` | `-ds`, `--drop_singleton` | Drop singleton axes | `true` |
| `num_workers` | `-nw`, `--num_workers` | Write threads (0 = auto) | auto |

In the batch CSV, leave a cell empty to use the default (so an empty `shards` cell means `auto`; write `none` to disable sharding).

## Sharding (zarr v3 only)

Sharding is on by default (`shards="auto"`). Without it every chunk is its own file; with sharding, many chunks are packed into one larger file (a *shard*), which greatly reduces the number of files on disk while each chunk stays individually readable. Sharding needs zarr v3 / NGFF 0.5: with zarr v2 ZarrSmith issues a warning and sets sharding to none.

```python
# Let ZarrSmith choose shard sizes (about shard_target_mb of uncompressed data per shard)
dataset.to_zarr("out.zarr", shards="auto", shard_target_mb=256)

# Disable sharding: one file per chunk
dataset.to_zarr("out.zarr", shards=None)

# Or give an explicit shard shape (one value per axis); it is snapped to a
# multiple of the chunk shape and clipped to each level's extent
dataset.to_zarr("out.zarr", shards=(1, 1, 8, 2160, 4096))
```

```console
zarrsmith convert -i INPUT -m opera -z OUT.zarr -sh auto -stm 256 -sea t c   # or: -sh none to disable
```

- `shards="auto"` picks a shard shape per pyramid level and leaves small levels unsharded.
- `shard_exclude_axes` lists axes that `"auto"` never merges chunks along. It defaults to `t c`, so each timepoint and channel stays in its own shard. Use `none` (CLI/CSV) or `()` (Python) to allow every axis to merge. An explicit shard shape is always used exactly as given.
- `shard_target_mb` is the target uncompressed size of a shard. The default is 1024 MB (1 GB). Larger shards mean fewer files but more memory per write thread (see below).

## Compression

Chunks are compressed by default with Blosc, using zstd and byte shuffle at level 1. This was the best compromise between size and speed in a benchmark on `uint16` light-sheet, mosaic and organoid data: about 2.5 to 2.9 times smaller on most of them, and about 1.5 times smaller on a noisy mosaic.

| Python / CSV column | CLI flag | Values | Default |
|---|---|---|---|
| `compressor` | `-cp`, `--compressor` | `blosc`, `gzip` (zarr v2 only), or no compression | `blosc` |
| `compressor_level` | `-cpl`, `--compressor_level` | 0 to 9 | 1 |

```python
dataset.to_zarr("out.zarr")                                   # Blosc-zstd, level 1
dataset.to_zarr("out.zarr", compressor="blosc", compressor_level=5)
dataset.to_zarr("out.zarr", compressor=None)                  # uncompressed
```

On the command line and in the batch CSV, write `none` in lowercase for uncompressed chunks. An empty CSV cell means the default.

- Higher levels write much more slowly and rarely help. Level 5 only made the noisy mosaic smaller (by about 7%), and it wrote about 6 times more slowly than level 2 there. Level 3 was usually slightly larger than level 1.
- An unusable combination, such as `gzip` with zarr v3 or a level above 9, is refused before anything is written or replaced.
- Existing stores are not changed. `chromatic-shift` keeps the codec of its input, and `create_empty_dataset` and `create_empty_group` keep their own default: with `compressor=None` they use zarr's default codec on zarr v3 and no compression on zarr v2.

## Singleton axes

With `drop_singleton=True` (the default) any `t`, `c` or `z` axis of size 1 is removed from the written dataset, so a single-timepoint, single-channel stack is stored as a plain `zyx` array. The axes list, scales, units and any `chunks`/`shards` you gave for the full axis set are adjusted accordingly. `y` and `x` are never dropped. Pass `drop_singleton=False` (`-ds false` in the CLI, `false` in the CSV, or untick the napari checkbox) to keep every axis.

```python
dataset.to_zarr("out.zarr", drop_singleton=False)   # keep size-1 t/c/z axes
```

Note that, because this is on by default, existing code that writes datasets with size-1 `t`, `c` or `z` axes now produces arrays with fewer dimensions.

## Parallel writing

Blocks of the dataset are compressed and written by several threads at once. `num_workers` sets the thread count; by default ZarrSmith uses **half of the available cores, capped at 8**, because writing stops getting faster beyond roughly 8 threads while memory use keeps growing.

```python
dataset.to_zarr("out.zarr", num_workers=4)
```

```console
zarrsmith convert -i INPUT -m opera -z OUT.zarr -nw 4
```

How it works:

- **Sharded levels** are written one whole shard per task, so threads never touch the same file and no locking is needed. Each thread holds one shard in memory, so the thread count is automatically reduced to fit in about half of the free RAM, and a warning is shown if that limits parallelism. If it does, use smaller shards (a lower `shard_target_mb`).
- **All pyramid levels are written in a single pass**, so the source data is read once instead of once per level. This matters most for slow sources such as files on a network drive.
- With `compute=False`, `to_zarr` returns the unevaluated dask tasks instead of writing, and `num_workers` does not apply: you run them with your own scheduler.

Writing speed is also bounded by how fast the source can be read and decoded, which `num_workers` does not change.

## Progress bar

`to_zarr` shows a [tqdm](https://tqdm.github.io/) progress bar (in a terminal or a notebook) while the arrays are written, labelled with the dataset being written (`raw`, a group or a label). The CLI shows it too. All pyramid levels of a dataset share one bar, because they are computed in a single pass.

```python
dataset.to_zarr("out.zarr")                    # progress bar on (default)
dataset.to_zarr("out.zarr", progress=False)    # silent, e.g. for batch jobs and tests
