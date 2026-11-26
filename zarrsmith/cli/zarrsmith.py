from __future__ import annotations

import shlex
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import zarrsmith.microscope_manager as mm
from zarrsmith.utils.axes import UNRECORDED_SCALE_KEY, metadata_axes, metadata_axis_size
from zarrsmith.utils.chunks import (default_chunk_size, default_num_levels, layout_summary, level0_layout,
                                    output_summary_line)
from zarrsmith.utils.ngff import DEFAULT_COMPRESSOR, DEFAULT_SHARD_TARGET_MB
from zarrsmith.utils.output_path import output_refusal
from zarrsmith.cli.__arguments import (
    _parse_arguments,
    parse_bool,
    parse_color,
    parse_downscale_factor,
    parse_shard_exclude_axes_spec,
    parse_shards_spec,
    parse_subset_spec,
)


def _dataset_size_mb(metadata):
    dtype = np.dtype(metadata.get("dtype", "uint16"))
    nbytes = dtype.itemsize
    for size in metadata["size"][0]:
        nbytes *= int(size)
    return nbytes / 1024 / 1024


def _check_chunk_size(chunk_size, metadata):
    """Return ``chunk_size`` as a tuple, or raise if it does not have one value per dataset axis."""
    axes = metadata_axes(metadata)
    chunk_size = tuple(int(c) for c in chunk_size)
    if len(chunk_size) != len(axes):
        raise ValueError(
            f"chunk_size={chunk_size} has {len(chunk_size)} values, but the dataset has "
            f"{len(axes)} axes ({axes}, shape {tuple(metadata['size'][0])}). "
            f"Provide one chunk size per axis, in {axes.upper()} order."
        )
    return chunk_size


def _normalize_downscale_factor(value):
    if value is None:
        return tuple([2] * 3)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return tuple([int(value[0])] * 3)
        return tuple(int(v) for v in value)
    return tuple([int(value)] * 3)


def _present(value):
    return not (
        pd.isna(value)
        or (isinstance(value, str) and value.strip() == "")
        or value == "-1"
    )


def _normalize_shards(value):
    """Normalize a `shards` CLI/CSV/kwarg value into ``None``, ``"auto"`` or a tuple."""
    if value is None:
        return None
    if isinstance(value, str):
        return parse_shards_spec(value)
    if isinstance(value, (list, tuple)):
        return parse_shards_spec(value)
    raise TypeError(
        f"shards must be 'auto', a shard shape, or None; got {value!r}."
    )


def _compression_kwargs(compressor, compressor_level) -> Dict[str, Any]:
    """``to_zarr`` options from a `compressor` CLI/CSV/kwarg value; ``"none"`` or ``None`` is uncompressed."""
    if isinstance(compressor, str):
        compressor = compressor.strip().lower()
    kwargs: Dict[str, Any] = {"compressor": None if compressor == "none" else compressor}
    if compressor_level is not None:
        kwargs["compressor_level"] = int(compressor_level)
    return kwargs


def _normalize_shard_exclude_axes(value):
    """Normalize a `shard_exclude_axes` CLI/CSV/kwarg value into ``None`` or a tuple."""
    if value is None:
        return None
    if isinstance(value, (str, list, tuple)):
        return parse_shard_exclude_axes_spec(value)
    raise TypeError(
        f"shard_exclude_axes must be a sequence of axis names, 'none', or None; got {value!r}."
    )


def _normalize_subset(value):
    """Normalize a subset specification from the CLI or a batch CSV."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k).upper(): v for k, v in value.items()}
    if isinstance(value, str):
        parsed = parse_subset_spec(value)
        return None if parsed is None else parsed
    raise TypeError(
        "subset must be either a dict of axis selections or a compact subset string"
    )

def _channel_indices(selection, names: List[str]):
    """A ``--subset c=`` selection with channel names replaced by their indices in ``names``
    (as the napari converter selects channels by name); other selections are returned as they are."""
    if isinstance(selection, str):
        selection = [selection]
    if not isinstance(selection, list) or not any(isinstance(v, str) for v in selection):
        return selection
    indices = []
    for value in selection:
        if isinstance(value, str):
            if value not in names:
                raise ValueError(f"--subset c: no channel named {value!r}; the channels are {list(names)}.")
            indices.append(list(names).index(value))
        else:
            indices.append(int(value))
    return indices


#: How the refusals of the CLI say to allow overwriting.
_OVERWRITE_HINT = "pass --overwrite (or fill the overwrite column of a --batch CSV)"


def _refuse_output(zarr_path, *, overwrite, input_path=None) -> None:
    """Stop before any work is done if ``zarr_path`` must not be written to."""
    refusal = output_refusal(zarr_path, overwrite=bool(overwrite), sources=[input_path], hint=_OVERWRITE_HINT)
    if refusal is not None:
        raise FileExistsError(refusal)


def _report_output_path(zarr_path, *, overwrite, input_path=None) -> None:
    """For a dry run: say whether the real run would be refused at ``zarr_path``, instead of stopping."""
    refusal = output_refusal(zarr_path, overwrite=bool(overwrite), sources=[input_path], hint=_OVERWRITE_HINT)
    if refusal is not None:
        print(f"NOTE (dry run): the real run would stop here: {refusal}")


def _print_dry_run(zarr_path, output_line: str, layout_line: str, num_levels, extra: str = "") -> None:
    """The recap of ``--dry_run``: the lines the napari widgets show; nothing is written."""
    n = int(num_levels)
    print(f"\n--->Dry run: nothing written to {zarr_path}")
    print(f"Output: {output_line}{extra}")
    print(f"Zarr: {layout_line}, {n} pyramid level{'s' if n != 1 else ''}")


def _warn_rows_skipped(skipped: List[str]) -> None:
    """Recap the rows of a batch that were not run because of what is at their output."""
    if skipped:
        print(f"WARNING: {len(skipped)} row(s) skipped, nothing was written for them: {', '.join(skipped)}",
              file=sys.stderr)


def _resolve_zarr_manager(input_path: str, microscope: Optional[str] = None):
    """Resolve a manager class from an explicit microscope name or by inspecting the path."""
    names = ", ".join(f'"{key}"' for key in mm.READERS)
    if _present(microscope):
        key = str(microscope).strip().lower()
        if key not in mm.READERS:
            raise TypeError(f'Microscope {microscope} not recognized. Should be one of {names}.')
        return mm.READERS[key], key
    found = mm.detect_reader(input_path)
    if found is None:
        raise TypeError(
            'Could not auto-detect the microscope from the input path. '
            f'Pass --microscope explicitly with one of {names}.'
        )
    return found

def zarr_convert(
    input_path, 
    zarr_path, 
    microscope: Optional[str] = None, 
    max_size : Optional[float] = None,
    chunk_size : Optional[List[int]] = None,
    scene_index : Optional[int] = None,
    channel_names : Optional[List[str]] = None,
    channel_colors : Optional[List[str]] = None,
    zarr_format : Optional[int] = 3,
    downscale_factor: Optional[int] = 2,
    num_levels: Optional[int] = None,
    method: Optional[str] = None,
    subset: Optional[dict] = None,
    shards: Optional[Any] = "auto",
    shard_target_mb: Optional[float] = None,
    shard_exclude_axes: Optional[Any] = None,
    compressor: Optional[str] = DEFAULT_COMPRESSOR,
    compressor_level: Optional[int] = None,
    drop_singleton: Optional[bool] = True,
    num_workers: Optional[int] = None,
    overwrite: Optional[bool] = False,
    dry_run: bool = False,
):
    """Helper function for CLI to convert a dataset to zarr given some parameters.

    Parameters
    ----------
        input_path : str
            Input path for the data to be converted.
        zarr_path : str
            Output .zarr path.
        microscope : str
            Microscope used to acquire input data.
            One of \"luxendo\", \"opera\", \"viventis\", \"zeiss\", \"evident\", \"zarrv04\", \"zarr\".
        max_size : Optional[float]
            Max chunk size in MB. \n
            Default: None (about 4 MB, about cubic in micrometers, at most 32 z planes,
            see zarrsmith.utils.chunks.default_chunk_size)
        chunk_size : Optional[List[int]]
            Chunk size in TCZYX format, or whatever axes are present in the dataset.
            Default: None
        scene_index : Optional[int]
            Scene index for .czi and .vsi files. \n
            Default: None (the first scene)
        channel_names : Optional[List[str]]
            Name of channels.\n
            Example: \"-cn bf gfp rfp\"\n
            Default: None
        channel_colors : Optional[List[str]]
            Colors of channels (hex or matplotlib color name)\n
            Example: \"-cc 0000FF cyan 00ff00\")\n
            Default: None
        zarr_format : Optional[int]
            Output zarr format. Zarr v2 maps to NGFF 0.4 and Zarr v3 maps to NGFF 0.5.\n
            Default: 3
        downscale_factor : Optional[int]
            Pyramid downsampling factor.\n
            Example: \"-df 1 2 2\")\n
            Default: 2
        num_levels : Optional[int]
            Number of pyramid levels.\n
            Example: \"-nl 3\")\n
            Default: None
        method : Optional[str]
            Downsampling method of the pyramid levels: "mean", "nearest" or "mode".\n
            Example: \"-me nearest\"\n
            Default: None (mean for intensity data, nearest for labels)
        subset : Optional[dict]
            Axis subset to apply before chunk selection and pyramid generation.\n
            Example: "--subset y=10:100:2;x=20:80"\n
            Default: None
        shards : Optional[str | List[int]]
            Zarr v3 sharding for the output.

            "auto" sizes shards automatically; a shape (matching the dataset's
            axes) requests that shard shape explicitly; "none" disables sharding.
            With zarr_format=2 sharding is set to none with a warning.

            Example: \"-sh auto\", \"-sh 1 1 8 2160 4096\" or \"-sh none\"

            Default: "auto"
        shard_target_mb : Optional[float]
            Target uncompressed shard size in MB, used when shards="auto".\n
            Default: 1024 (zarrsmith's own default; unset here means "let zarrsmith choose")
        shard_exclude_axes : Optional[List[str]]
            Axes that shards="auto" never merges chunks along.\n
            Example: \"-sea t c\" (default) or \"-sea none\" to allow every axis to merge.\n
            Default: None (falls back to zarrsmith's own default of t, c)
        compressor : Optional[str]
            Codec of the output: "blosc" (Blosc with zstd and byte shuffle), "gzip"
            (zarr_format 2 only), or "none" or None for uncompressed chunks.\n
            Example: \"-cp none\"

            Default: "blosc"
        compressor_level : Optional[int]
            Compression level, 0 to 9.\n
            Default: None (falls back to zarrsmith's own default of 1)
        drop_singleton : Optional[bool]
            Remove singleton t/c/z axes from the output dataset.\n
            Example: \"-ds false\" to keep them.\n
            Default: True
        num_workers : Optional[int]
            Number of threads used to write the zarr in parallel.

            Example: \"-nw 4\"

            Default: None (half of the available cores, capped at 8)
        overwrite : Optional[bool]
            Replace ``zarr_path`` if it is an existing Zarr store. Otherwise an existing
            non-empty output raises ``FileExistsError``, before anything is read.

            Default: False
        dry_run : bool
            Print the output, its chunks, shards and pyramid levels (as the napari
            widget shows them) and return without writing anything.

            Default: False
    """
    if dry_run:
        _report_output_path(zarr_path, overwrite=overwrite, input_path=input_path)
    else:
        _refuse_output(zarr_path, overwrite=overwrite, input_path=input_path)

    manager, resolved_microscope = _resolve_zarr_manager(input_path, microscope)
    print(f'\n--->Using manager: {manager.__name__} ({resolved_microscope})')

    downscale_factor = _normalize_downscale_factor(downscale_factor)
    
    # --- Figure out chunks dimensions ---
    if resolved_microscope.lower() in ("zeiss", "evident"):
        dataset = manager(path=input_path, scene_index=0 if scene_index is None else scene_index)
    else:
        dataset = manager(path=input_path)
        
    # --- Show metadata summary ---
    print("\n--->Input dataset")
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print("CHUNK SIZE:", dataset.chunks)
    print("DATASET SIZE (MB):", _dataset_size_mb(dataset.metadata))

    subset_kwargs = _normalize_subset(subset)
    if subset_kwargs and "C" in subset_kwargs:
        subset_kwargs["C"] = _channel_indices(subset_kwargs["C"], list(dataset.metadata.get("channel_names") or []))
    if subset_kwargs:
        print(f"\n--->Applying subset: {subset_kwargs}")
        dataset.subset_dataset(**subset_kwargs, rebuild_pyramid=False)
        print("\n--->Dataset after subsetting")
        for i in dataset.metadata:
            print(f"{i.upper()}: {dataset.metadata[i]}")
        print("DATASET SIZE (MB):", _dataset_size_mb(dataset.metadata))
        
    # --- Select chunk size ---
    print("\n--->Select chunks.")
    if chunk_size is not None:
        chunk_size = _check_chunk_size(chunk_size, dataset.metadata)
        print(f"\tUsing user-provided chunk size: {chunk_size} ({metadata_axes(dataset.metadata)})")
        size_mb = _dataset_size_mb(dataset.metadata) * np.prod(chunk_size) / np.prod(dataset.metadata["size"][0])
        n_chunks = {ax: int(np.ceil(size / chunk)) for ax, size, chunk in zip(metadata_axes(dataset.metadata), dataset.metadata["size"][0], chunk_size)}
    else:
        chunk_size, size_mb, n_chunks = default_chunk_size(dataset.metadata, max_size)
        print(f"\tUsing max_size={max_size if max_size is not None else 'default'} MB to select chunk size.")

    print(f"Chunk size: {chunk_size}, {size_mb} MB.")
    print(f"N chunks: {n_chunks}.")

    dataset.data = [
        arr.rechunk(chunk_size) if arr.ndim == len(chunk_size) else arr
        for arr in dataset.data
    ]
    dataset.metadata["chunksize"] = [tuple(arr.chunksize) for arr in dataset.data]
    if hasattr(dataset, "chunks"):
        dataset.chunks = chunk_size

    # --- Build pyramid ---
    print("\n--->Selected pyramidal layers, lower layer should have dims<2048")
    if num_levels is None:
        num_levels = default_num_levels(dataset.metadata, downscale_factor)
    print(f"Number of levels: {num_levels}.")
    print(f"Downsampling method: {method or 'default (mean for intensity, nearest for labels)'}.")

    dataset.build_pyramid(
        num_levels=num_levels, 
        downscale_factor=downscale_factor,
        method=method,
    )

    # --- Modify metadata according to optional parameters ---
    
    # Metadata format:
    # metadata = {
    #         # "size": [(size_t, size_c, size_z, size_y, size_x)], # can't change
    #         # "scales": scales, # can't change
    #         # "units": units, # can't change
    #         # "time_increment": time_increment, # can't change
    #         # "time_increment_unit": time_unit, # can't change
    #         "channel_names": channel_names,
    #         "channel_colors": channel_colors,
    #         # "dtype": pixels.attrib.get("Type", "uint16"), # can't change
    #         # "axes": "tczyx" # can't change
    #     }

    print("\n--->Updating metadata to selected channel_names and channel_colors")
    metadata = {}
    if channel_names:
        if "c" not in metadata_axes(dataset.metadata):
            raise TypeError("channel_names were provided, but the dataset has no channel axis.")
        n_ch = metadata_axis_size(dataset.metadata, "c")
        if len(channel_names)!=n_ch:
            raise TypeError(f"Length of channel_names={channel_names} does not match dataset channels of length={n_ch}.")
        metadata["channel_names"] = channel_names
        if channel_colors:
            if len(channel_colors)!=n_ch:
                raise TypeError(f"Length of channel_colors={channel_colors} does not match dataset channels of length={n_ch}.")
            metadata["channel_colors"] = channel_colors
    dataset.update_metadata(metadata)

    print("\n--->Updating metadata to selected zarr_format and downscale_factor")
    ngff_version = '0.4' if int(zarr_format) == 2 else '0.5'

    # --- Resolve sharding options ---
    shards = _normalize_shards(shards)
    shard_exclude_axes = _normalize_shard_exclude_axes(shard_exclude_axes)
    drop_singleton = True if drop_singleton is None else parse_bool(drop_singleton)
    to_zarr_kwargs = {
        "zarr_format": int(zarr_format),
        "ngff_version": ngff_version,
        "drop_singleton": drop_singleton,
        "shards": shards,
        "overwrite": bool(overwrite),
        **_compression_kwargs(compressor, compressor_level),
    }
    if num_workers is not None:
        to_zarr_kwargs["num_workers"] = int(num_workers)
    if shards is not None:
        if shard_target_mb is not None:
            to_zarr_kwargs["shard_target_mb"] = float(shard_target_mb)
        if shard_exclude_axes is not None:
            to_zarr_kwargs["shard_exclude_axes"] = shard_exclude_axes

    # --- Show metadata summary ---
    print("\n--->Input dataset after adjustments")
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print(f"CHUNK SIZE: {dataset.chunks} , {size_mb} MB.")
    print(f"N CHUNKS: {n_chunks}.")
    print(f"PYRAMID LEVELS: {num_levels}.")
    print(f"ZARR FORMAT: {zarr_format}, NGFF VERSION: {ngff_version}.")
    print(f"COMPRESSOR: {to_zarr_kwargs['compressor']}, LEVEL: {to_zarr_kwargs.get('compressor_level', '(zarrsmith default: 1)')}.")
    print(f"DROP_SINGLETON: {drop_singleton}.")
    print(f"NUM_WORKERS: {num_workers if num_workers is not None else '(zarrsmith default)'}.")
    print(f"SHARDS: {shards}.")
    if shards is not None:
        print(f"SHARD_TARGET_MB: {to_zarr_kwargs.get('shard_target_mb', f'(zarrsmith default: {DEFAULT_SHARD_TARGET_MB:g})')}.")
        print(f"SHARD_EXCLUDE_AXES: {shard_exclude_axes if shard_exclude_axes is not None else '(zarrsmith default: t, c)'}.")

    if dry_run:
        md = dataset.metadata
        axes = metadata_axes(md)
        layout = level0_layout(axes, md["size"][0], md["dtype"], chunk_size, zarr_format=int(zarr_format),
                               shards=shards, shard_mb=to_zarr_kwargs.get("shard_target_mb"),
                               shard_exclude_axes=shard_exclude_axes)
        line = output_summary_line(axes, dict(zip(axes, md["size"][0])), md["dtype"], md["scales"][0],
                                   md.get("units") or (), drop_singleton=drop_singleton)
        _print_dry_run(zarr_path, line, layout_summary(axes, md["dtype"], layout), num_levels)
        return

    # --- Write to OME-Zarr format ---
    print("\n--->Writing to zarr")
    dataset.to_zarr(zarr_path, **to_zarr_kwargs)

    # --- Show metadata summary for updated dataset ---
    dataset = mm.ZarrManager(path=zarr_path)

def _batch_row_kwargs(v, columns) -> Dict[str, Any]:
    """``zarr_convert`` options of one batch CSV row (shared by ``convert --batch`` and ``fuse --batch``)."""
    kwargs: Dict[str, Any] = {}

    # Prioritize chunk_size over max_size if both are present
    if "chunk_size" in columns and _present(v.get("chunk_size")):
        kwargs["chunk_size"] = parse_downscale_factor(v["chunk_size"])
    elif "max_size(MB)" in columns and _present(v.get("max_size(MB)")):
        kwargs["max_size"] = float(v["max_size(MB)"])

    if "scene_index" in columns and _present(v.get("scene_index")):
        kwargs["scene_index"] = int(v["scene_index"])

    if "channel_names" in columns and _present(v.get("channel_names")):
        kwargs["channel_names"] = [c.strip() for c in str(v["channel_names"]).split(" ")]

    if "channel_colors" in columns and _present(v.get("channel_colors")):
        kwargs["channel_colors"] = [parse_color(c.strip()) for c in str(v["channel_colors"]).split(" ")]

    if "zarr_format" in columns and _present(v.get("zarr_format")):
        kwargs["zarr_format"] = int(v["zarr_format"])

    if "downscale_factor" in columns and _present(v.get("downscale_factor")):
        kwargs["downscale_factor"] = parse_downscale_factor(v["downscale_factor"])

    if "num_levels" in columns and _present(v.get("num_levels")):
        kwargs["num_levels"] = int(v["num_levels"])

    if "method" in columns and _present(v.get("method")):
        kwargs["method"] = str(v["method"]).strip().lower()

    for key in ("subset", "shards", "shard_exclude_axes"):
        if key in columns and _present(v.get(key)):
            kwargs[key] = v[key]

    if "shard_target_mb" in columns and _present(v.get("shard_target_mb")):
        kwargs["shard_target_mb"] = float(v["shard_target_mb"])

    if "compressor" in columns and _present(v.get("compressor")):
        kwargs["compressor"] = v["compressor"]

    if "compressor_level" in columns and _present(v.get("compressor_level")):
        kwargs["compressor_level"] = int(v["compressor_level"])

    if "drop_singleton" in columns and _present(v.get("drop_singleton")):
        kwargs["drop_singleton"] = parse_bool(v["drop_singleton"])

    if "num_workers" in columns and _present(v.get("num_workers")):
        kwargs["num_workers"] = int(v["num_workers"])

    if "overwrite" in columns and _present(v.get("overwrite")):
        kwargs["overwrite"] = parse_bool(v["overwrite"])

    return kwargs


#: Parsed arguments that are not options of the conversion functions.
_NOT_CONVERSION_OPTIONS = {"runmode", "mode", "command", "renamed_to", "batch", "input_file"}


def convert_batch(args):
    """Runmode to convert batch of imaged to zarr

    Args:
        args (args): parsed arguments
    """
    csv_path = getattr(args, "input_file", args)  # parsed arguments or a path
    print(f"Converting batch.\nRunning through: zarrsmith convert --batch {csv_path}")

    database = pd.read_csv(csv_path)
    print(database)
    skipped = []

    for i, v in database.iterrows():
        print("-" * 20)

        input_f = v["input"]
        output = v["output"]
        microscope = v["microscope"]

        conv_kwargs = {
            "input_path": input_f,
            "zarr_path": output,
            "microscope": microscope,
            **_batch_row_kwargs(v, database.columns),
            "dry_run": bool(getattr(args, "dry_run", False)),
        }

        # An output in the way only stops its own row: the others still run.
        try:
            zarr_convert(**conv_kwargs)
        except FileExistsError as error:
            print(f"WARNING: row skipped: {error}", file=sys.stderr)
            skipped.append(str(output))
    _warn_rows_skipped(skipped)

def convert_single(args):
    """Runmode to convert a single image to zarr

    Args:
        args (args): parsed arguments
    """
    cli = (
        f'zarrsmith convert --input_path {args.input_path} --zarr_path {args.zarr_path} '
        f'--microscope {args.microscope} --max_size {args.max_size} '
        f'--scene_index {args.scene_index} --channel_names {args.channel_names} '
        f'--channel_colors {args.channel_colors} --zarr_format {args.zarr_format} '
        f'--num_levels {args.num_levels} --downscale_factor {args.downscale_factor} '
        f'--method {args.method} '
        f'--chunk_size {args.chunk_size} --subset {args.subset} '
        f'--shards {args.shards} --shard_target_mb {args.shard_target_mb} '
        f'--shard_exclude_axes {args.shard_exclude_axes} '
        f'--compressor {args.compressor} --compressor_level {args.compressor_level} '
        f'--drop_singleton {args.drop_singleton} --num_workers {args.num_workers}'
        f'{" --overwrite" if args.overwrite else ""}'
    )
    print(f'Converting single file.\nRunning through: {cli}')
    exclude = _NOT_CONVERSION_OPTIONS
    conv_kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in exclude and v is not None and not (isinstance(v, str) and v.strip() == "")
    }
    if "chunk_size" in conv_kwargs:
        conv_kwargs.pop("max_size", None)

    # Convert 2 zarr
    zarr_convert(**conv_kwargs)


# ---------------------------------------------------------------------------
# Mosaics: fuse / fuse --batch
# ---------------------------------------------------------------------------

def _mosaic_manager_name_map() -> Dict[str, Any]:
    from zarrsmith.mosaic_manager import MOSAIC_READERS

    return dict(MOSAIC_READERS)


def _resolve_mosaic_manager(input_path: str, microscope: Optional[str] = None):
    """Mosaic reader class from an explicit microscope name or by inspecting the folder.

    An explicit microscope that does not match a recognised folder is refused when
    the reader is created (:class:`~zarrsmith.mosaic_manager.WrongReaderError`)."""
    from zarrsmith.mosaic_manager import detect_microscope

    readers = _mosaic_manager_name_map()
    if _present(microscope):
        key = str(microscope).strip().lower()
        if key not in readers:
            raise TypeError(f'Mosaic microscope {microscope} not recognized. Should be one of {sorted(readers)}.')
        return readers[key], key
    key = detect_microscope(input_path)
    if key is not None:
        return readers[key], key
    raise TypeError(
        'Could not auto-detect the mosaic type from the input path. '
        f'Pass --microscope explicitly with one of {sorted(readers)}.'
    )


def _int_triplet(value, default=(1, 1, 1)) -> Tuple[int, int, int]:
    """``None`` -> default, ``n`` -> ``(n, n, n)``, ``(y, x)`` -> ``(1, y, x)``, ``(z, y, x)`` as is."""
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        value = parse_downscale_factor(value)
    if isinstance(value, (int, np.integer)):
        return (int(value),) * 3
    value = tuple(int(v) for v in value)
    if len(value) == 1:
        return value * 3
    if len(value) == 2:
        return (1,) + value
    if len(value) == 3:
        return value
    raise ValueError(f"Expected 1, 2 or 3 integers, got {value}.")


def _subset_to_crop(subset, channel_names: List[str]):
    """Split a convert-style subset into a fusion crop (z/y/x) and a channel selection (c)."""
    subset = _normalize_subset(subset) or {}
    if "T" in subset:
        raise ValueError("Choose the timepoints of a mosaic with --timepoints: remove t from --subset.")
    channels = list(channel_names)
    if "C" in subset:
        sel = _channel_indices(subset["C"], channels)
        if isinstance(sel, slice):
            channels = channels[sel]
        elif isinstance(sel, list):
            channels = [channels[i] for i in sel]
        else:
            channels = [channels[int(sel)]]
    crop = {}
    for axis in "ZYX":
        if axis in subset:
            sel = subset[axis]
            if isinstance(sel, list):
                raise ValueError(f"--subset {axis.lower()}: use a range such as 10:200 (index lists are not supported for mosaics).")
            crop[axis.lower()] = sel
    return crop or None, channels


def mosaic_convert(
    input_path,
    zarr_path,
    microscope: Optional[str] = None,
    max_size: Optional[float] = None,
    chunk_size: Optional[List[int]] = None,
    scene_index: Optional[int] = None,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[str]] = None,
    zarr_format: Optional[int] = 3,
    downscale_factor: Optional[Any] = 2,
    num_levels: Optional[int] = None,
    method: Optional[str] = None,
    subset: Optional[Any] = None,
    shards: Optional[Any] = "auto",
    shard_target_mb: Optional[float] = None,
    shard_exclude_axes: Optional[Any] = None,
    compressor: Optional[str] = DEFAULT_COMPRESSOR,
    compressor_level: Optional[int] = None,
    drop_singleton: Optional[bool] = True,
    num_workers: Optional[int] = None,
    read_downsample: Optional[Any] = None,
    projection: Optional[str] = "none",
    blend: Optional[str] = None,
    sigmoid_steepness: Optional[float] = None,
    channels: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    cameras: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    tiles: Optional[List[str]] = None,
    skip_tiles: Optional[List[str]] = None,
    timepoints: Optional[List[int]] = None,
    reference_timepoint: Union[int, str, None] = None,
    midline_axis: Optional[str] = None,
    out_dtype: Optional[str] = None,
    data_type: Optional[str] = None,
    label_offsets: Optional[str] = None,
    label_mapping: Optional[str] = None,
    read_workers: int = 0,
    temp_dir: Optional[str] = None,
    overwrite: bool = False,
    flatfield: bool = True,
    transforms: Optional[str] = None,
    transforms_only: bool = False,
    register_cameras: bool = False,
    register_channel: Optional[str] = None,
    register_model: str = "rigid",
    dry_run: bool = False,
):
    """Fuse a mosaic into an OME-Zarr image (CLI helper for ``zarrsmith fuse``).

    The ``convert`` options (``max_size`` ... ``num_workers``, ``method``) have the same meaning as in
    :func:`zarr_convert`, applied to the fused output; ``subset`` crops it (z/y/x in
    output voxels) and selects channels (c). ``tiles`` lists the only tiles to fuse
    (ids as for ``skip_tiles``); the output then fits them. ``fields`` keeps only the
    tiles of these fields (connected groups of touching tiles, ``field0``, ``field1``...);
    without ``tiles`` and ``fields`` the first field is fused. Tiles from several fields
    are not fused: an error is printed and ``False`` returned (``True`` once the output
    is saved). ``timepoints`` and ``reference_timepoint`` choose the timepoints to fuse and
    the one whose geometry they all use (``"all"``: each at its own positions). The other options are those of
    :func:`zarrsmith.mosaic_manager.fuse_tiles`: ``read_downsample`` is its ``ds`` and
    ``read_workers`` its ``workers`` (0: automatic, see :func:`~zarrsmith.mosaic_manager.fusion.fusion_workers`). ``flatfield`` is passed to readers that correct
    the tiles (Opera); the others ignore it.

    ``label_offsets`` (CSV ``store,offset``) and ``label_mapping`` (JSON ``{id: new_id}``)
    renumber label tiles as they are read, see :func:`_relabel_tiles`; either implies
    ``data_type="label"``.

    The transforms (where every tile goes) are written to
    :func:`~zarrsmith.mosaic_manager.default_transforms_path` of ``zarr_path`` before
    fusing; with ``transforms_only`` nothing else is done. ``transforms`` is such a file
    to place the tiles with instead of the stage positions (its reference timepoint is
    used unless ``reference_timepoint`` is given). ``register_cameras`` registers the views
    of opposing cameras on the images before fusing (``register_model`` ``"rigid"`` or
    ``"translation"``), measured on the reference channel ``register_channel`` (default: the
    first; ``"all"``: every channel) at the reference timepoint (every timepoint with
    ``reference_timepoint="all"``) and applied to the others, see :meth:`MosaicManager.register_cameras
    <zarrsmith.mosaic_manager.MosaicManager.register_cameras>`.

    ``dry_run`` prints the output, its chunks, shards, pyramid levels and fusion workers (as
    the napari widget shows them) and returns ``True`` without writing anything (camera
    registration, which reads the images, is skipped).
    """
    from zarrsmith.mosaic_manager import (default_transforms_path, fuse_tiles, plan_fusion, reader_options,
                                          transforms_document, write_transforms)
    from zarrsmith.mosaic_manager.fusion import SIGMOID_STEEPNESS, default_fusion_levels

    transforms_only = bool(parse_bool(transforms_only)) if transforms_only is not None else False
    relabel = _present(label_offsets) or _present(label_mapping)
    # The reader picks stores by the data type asked for; tiles renumbered as labels may come
    # from stores that do not say they hold labels (e.g. written by another tool), so only an
    # explicit --data_type restricts it.
    reader_data_type = data_type
    if relabel:
        if data_type not in (None, "label"):
            raise ValueError("--label_offsets/--label_mapping renumber label tiles: they need --data_type label.")
        data_type = "label"
    if dry_run:
        _report_output_path(zarr_path, overwrite=overwrite, input_path=input_path)
    elif not transforms_only:
        _refuse_output(zarr_path, overwrite=overwrite, input_path=input_path)

    reader, resolved = _resolve_mosaic_manager(input_path, microscope)
    print(f"\n--->Using mosaic manager: {reader.__name__} ({resolved})")
    if scene_index is not None and resolved != "zeiss":
        print(f"\tscene_index={scene_index} is ignored for {resolved} mosaics.")
    options = reader_options(reader, flatfield=flatfield, scene_index=scene_index, data_type=reader_data_type)
    if not flatfield and "flatfield" not in options:
        print(f"\t--no_flatfield is ignored for {resolved} mosaics (no flat-field correction).")
    mosaic = reader(input_path, **options)
    print(f"{len(mosaic.tiles)} tiles, channels {mosaic.channels}, groups {mosaic.groups}, "
          f"fields {mosaic.fields}")
    if len(mosaic.timepoints) > 1:
        print(f"{len(mosaic.timepoints)} timepoints: {mosaic.timepoints}")
    if len(mosaic.cameras) > 1:
        print(f"{len(mosaic.cameras)} cameras: {mosaic.cameras}")
    if len(mosaic.fields) > 1:
        for f in mosaic.fields:
            stacks = sorted({mosaic.tile_id(t).split("/")[1] for t in mosaic.field_tiles(f)})
            print(f"\t{f}: {len(stacks)} stack position(s): {' '.join(stacks)}")
    if _present(transforms):
        doc = mosaic.apply_transforms(transforms)
        print(f"Tile positions from {transforms} ({doc.get('method')}, {len(doc['tiles'])} tile(s))")
        if reference_timepoint is None and doc.get("reference_timepoint") is not None:
            reference_timepoint = doc["reference_timepoint"]

    use = mosaic.fusion_tiles(list(tiles) if tiles else None, list(fields) if fields else None)
    if tiles:
        print(f"Fusing only {len(use)} tile(s) of {list(tiles)}")
    if fields:
        print(f"Fusing only fields {list(fields)}: {len(use)} tile(s)")
    elif not tiles and len(mosaic.fields) > 1:
        print(f"Fusing the first field, {mosaic.fields[0]} ({len(use)} tiles); choose another with --fields.")
    if groups:
        wanted = {str(g) for g in groups}
        use = [t for t in use if str(t.group) in wanted]
        if not use:
            raise ValueError(f"No tile in groups {sorted(wanted)}; available: {mosaic.groups}.")
    if cameras:
        wanted = {str(c) for c in cameras}
        use = [t for t in use if str(t.camera) in wanted]
        if not use:
            raise ValueError(f"No tile of cameras {sorted(wanted)}; available: {mosaic.cameras}.")
    if relabel:
        _relabel_tiles(use, label_offsets if _present(label_offsets) else None,
                       label_mapping if _present(label_mapping) else None, out_dtype or "uint32")

    if register_cameras and parse_bool(register_cameras) and dry_run:
        print("--register_cameras: skipped in a dry run (it reads the images); the output may move a little.")
    elif register_cameras and parse_bool(register_cameras):
        if len({t.camera for t in use}) < 2:
            print("--register_cameras: the fused tiles come from one camera, nothing to register.")
        else:
            model = register_model or "rigid"
            print(f"\n--->Registering the camera views ({model}, at the camera handover)")
            results = mosaic.register_cameras(use, reference_channel=register_channel or None,
                                              reference_timepoint=reference_timepoint, model=model)
            for r in results.values():
                print(f"\t{r.summary()}")

    skip = mosaic.tiles_by_ids(list(skip_tiles)) if skip_tiles else None
    if skip:
        print(f"Leaving {len(skip)} tile(s) blank: {[mosaic.tile_id(t) for t in skip]}")

    ds = _int_triplet(read_downsample)
    projection = None if projection in (None, "none", "None", "") else projection
    crop, selected = _subset_to_crop(subset, channels or list(dict.fromkeys(t.channel for t in use)))
    plan = plan_fusion(use, ds=ds, projection=projection, blend=blend, channels=selected,
                       crop=crop, skip=skip, out_dtype=out_dtype, midline_axis=midline_axis,
                       timepoints=timepoints, reference_timepoint=reference_timepoint, data_type=data_type)
    print(f"\n--->Output: {plan.summary()} ({plan.data_type} data, blend {plan.blend})")
    if plan.error:
        print(f"ERROR: {plan.error}")
        return False
    transforms_path = default_transforms_path(zarr_path)
    if transforms_only and not dry_run:
        write_transforms(transforms_path, transforms_document(plan, tile_id=mosaic.tile_id, source=str(input_path)))
        print(f"Transforms written to {transforms_path} (nothing fused: --transforms_only)")
        return True

    plan_md = {"axes": plan.axes, "size": [plan.shape], "dtype": str(plan.dtype), "scales": [plan.scale],
               UNRECORDED_SCALE_KEY: tuple(ax for ax, v in plan.voxel_size.items() if v is None)}
    if chunk_size is not None:
        chunks = tuple(int(c) for c in chunk_size)
        print(f"\tUsing user-provided chunk size: {chunks}")
    else:
        chunks, size_mb, _ = default_chunk_size(plan_md, max_size)
        print(f"\tUsing max_size={max_size if max_size is not None else 'default'} MB: "
              f"chunk size {chunks}, {size_mb:.1f} MB.")

    factor = _normalize_downscale_factor(downscale_factor)
    if num_levels is None:
        num_levels = default_fusion_levels(plan, factor)  # as fuse_tiles and the napari widget

    shards = _normalize_shards(shards)
    shard_exclude_axes = _normalize_shard_exclude_axes(shard_exclude_axes)
    write_kwargs = {
        "zarr_format": int(zarr_format),
        "ngff_version": "0.4" if int(zarr_format) == 2 else "0.5",
        "drop_singleton": True if drop_singleton is None else parse_bool(drop_singleton),
        "shards": shards,
        "chunks": chunks,
        **_compression_kwargs(compressor, compressor_level),
    }
    if num_workers is not None:
        write_kwargs["num_workers"] = int(num_workers)
    if shards is not None:
        if shard_target_mb is not None:
            write_kwargs["shard_target_mb"] = float(shard_target_mb)
        if shard_exclude_axes is not None:
            write_kwargs["shard_exclude_axes"] = shard_exclude_axes

    if dry_run:
        from zarrsmith.mosaic_manager.fusion import FUSION_SHARD_TARGET_MB, fusion_workers

        target = write_kwargs.get("shard_target_mb", FUSION_SHARD_TARGET_MB)
        layout = level0_layout(plan.axes, plan.shape, plan.dtype, chunks, zarr_format=int(zarr_format),
                               shards=shards, shard_mb=target, shard_exclude_axes=shard_exclude_axes)
        with warnings.catch_warnings():  # the number is shown; the real run warns
            warnings.simplefilter("ignore")
            layout["workers"] = fusion_workers(plan, layout["shards"] or layout["chunks"], int(read_workers))
        layout["auto_workers"] = int(read_workers) <= 0
        _print_dry_run(zarr_path, plan.summary(), layout_summary(plan.axes, plan.dtype, layout), num_levels,
                       f" ({plan.data_type} data, blend {plan.blend})")
        return True

    print(f"PYRAMID LEVELS: {num_levels}, DOWNSCALE FACTOR: {factor}, "
          f"METHOD: {method or 'default (mean for intensity, nearest for labels)'}.")
    print(f"WRITE OPTIONS: {write_kwargs}")
    print("\n--->Fusing to zarr")
    fuse_tiles(
        use,
        zarr_path,
        ds=ds,
        projection=projection,
        blend=blend,
        channels=plan.channels,
        channel_names=channel_names,
        channel_colors=channel_colors,
        crop=crop,
        skip=skip,
        levels=int(num_levels),
        downscale_factor=factor,
        pyramid_method=method,
        workers=int(read_workers),
        out_dtype=out_dtype,
        data_type=data_type,
        midline_axis=midline_axis,
        sigmoid_steepness=SIGMOID_STEEPNESS if sigmoid_steepness is None else float(sigmoid_steepness),
        timepoints=timepoints,
        reference_timepoint=reference_timepoint,
        overwrite=bool(overwrite),
        temp_dir=temp_dir,
        transforms_path=transforms_path,
        tile_id=mosaic.tile_id,
        source=str(input_path),
        **write_kwargs,
    )
    print(f"Saved {zarr_path} (transforms: {transforms_path})")
    return True


def _relabel_tiles(tiles, offsets_csv: Optional[str], mapping_json: Optional[str], dtype: str = "uint32") -> None:
    """Renumber label tiles in place as they are read (``zarrsmith fuse --label_offsets/--label_mapping``).

    ``offsets_csv`` has the columns ``store`` (a store name, with or without ``.ome.zarr``)
    and ``offset``: every non-zero id of that store's tiles gets the offset added, so
    tiles numbered separately do not clash. Every tile needs one. ``mapping_json``
    (``{"id": new_id}``) is applied afterwards, e.g. the objects a label solver merged
    across tile edges; ids it does not list are kept. Each tile is given only the
    entries between its offset and the next larger one (the ids it can hold), so the
    workers do not each carry the whole mapping. The tiles are marked as label data.
    """
    import json

    from zarrsmith.mosaic_manager import RelabeledArray

    offsets: Dict[str, int] = {}
    if offsets_csv is not None:
        table = pd.read_csv(offsets_csv)
        missing_cols = {"store", "offset"} - set(table.columns)
        if missing_cols:
            raise ValueError(f"{offsets_csv} needs the columns store and offset; missing {sorted(missing_cols)}.")
        for store, offset in zip(table["store"].astype(str), table["offset"]):
            name = store.strip()
            offsets[name[:-len(".ome.zarr")] if name.endswith(".ome.zarr") else name] = int(offset)
    mapping: Dict[int, int] = {}
    if mapping_json is not None:
        with open(mapping_json, encoding="utf-8") as handle:
            mapping = {int(k): int(v) for k, v in json.load(handle).items()}

    def stack_of(tile) -> str:
        name = str(tile.attrs.get("store") or tile.attrs.get("stack") or "")
        return name[:-len(".ome.zarr")] if name.endswith(".ome.zarr") else name

    if offsets:
        unknown = sorted({stack_of(t) for t in tiles} - set(offsets))
        if unknown:
            raise ValueError(f"No offset for the tile store(s) {unknown} in {offsets_csv}.")
    bounds = sorted(set(offsets.values()))
    keys = np.array(sorted(mapping), dtype=np.int64)
    for tile in tiles:
        offset = offsets.get(stack_of(tile), 0)
        if offsets:
            i = bounds.index(offset)
            upper = bounds[i + 1] if i + 1 < len(bounds) else None
            lo = np.searchsorted(keys, offset, side="right")
            hi = len(keys) if upper is None else np.searchsorted(keys, upper, side="left")
            own = {int(k): mapping[int(k)] for k in keys[lo:hi]}
        else:
            own = mapping
        tile.data = RelabeledArray(tile.data, offset=offset, mapping=own, dtype=dtype)
        tile.attrs["data_type"] = "label"
    print(f"Renumbered {len(tiles)} label tile(s): {len(offsets)} offset(s), {len(mapping)} mapped id(s)")


def _apply_pyramid_like(kwargs: Dict[str, Any]) -> None:
    """``--pyramid_like IMAGE --pyramid_start_level K``: the number of levels and the factor of
    IMAGE from level K on, unless --num_levels/--downscale_factor are given (in place)."""
    from zarrsmith.utils.dataset_ops import pyramid_like

    like = kwargs.pop("pyramid_like", None)
    start = int(kwargs.pop("pyramid_start_level", 0) or 0)
    if like is None:
        return
    levels, factor = pyramid_like(mm.ZarrManager(str(like), mode="r").metadata, start)
    kwargs.setdefault("num_levels", levels)
    kwargs.setdefault("downscale_factor", list(factor))
    print(f"Pyramid like {like} from level {start}: {kwargs['num_levels']} level(s), "
          f"factor {kwargs['downscale_factor']}")


def _fusion_kwargs(args) -> Dict[str, Any]:
    kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in _NOT_CONVERSION_OPTIONS and v is not None and not (isinstance(v, str) and v.strip() == "")
    }
    if "chunk_size" in kwargs:
        kwargs.pop("max_size", None)
    _apply_pyramid_like(kwargs)
    return kwargs


def convert_mosaic_single(args):
    """Runmode to fuse one mosaic to zarr."""
    print(f"Fusing mosaic {args.input_path} -> {args.zarr_path}")
    if not mosaic_convert(**_fusion_kwargs(args)):
        raise SystemExit(1)


def fuse_labels_single(args):
    """``zarrsmith fuse-labels``: the label stores of a folder (one per tile), placed at their NGFF
    translation, renumbered by --label_offsets/--label_mapping, fused without blending."""
    print(f"Fusing label tiles {args.input_path} -> {args.zarr_path}")
    kwargs = _fusion_kwargs(args)
    kwargs["microscope"] = "zarr"
    if "label_offsets" not in kwargs and "label_mapping" not in kwargs:
        kwargs["data_type"] = "label"  # ids already agree: read the label stores as they are
    if not mosaic_convert(**kwargs):
        raise SystemExit(1)


def _whole_voxels(value, name: str):
    """Voxel lengths for plan_tiles must be whole: 512.0 is fine, 512.5 is refused."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {ax: _whole_voxels(v, f"{name} {ax}") for ax, v in value.items()}
    if float(value) != int(value):
        raise ValueError(f"{name} in voxels must be a whole number, got {value}; use --physical for units.")
    return int(value)


def plan_tiles_single(args):
    """``zarrsmith plan-tiles``: plan the tiles of one level of an OME-Zarr image, write the tiles CSV."""
    import math

    from zarrsmith.utils.tiling import write_tiles_csv

    image = mm.ZarrManager(str(args.input_path), mode="r")
    level = int(args.level)
    tile_size, overlap, halo, roi = args.tile_size, args.overlap, args.halo, args.roi
    if not args.physical:
        tile_size = _whole_voxels(tile_size, "--tile_size")
        overlap = _whole_voxels(overlap, "--overlap")
        halo = _whole_voxels(halo, "--halo")
        if roi:  # voxel bounds widened to whole voxels
            roi = {ax: (None if lo is None else math.floor(lo + 1e-9), None if hi is None else math.ceil(hi - 1e-9))
                   for ax, (lo, hi) in roi.items()}
    plan = image.plan_tiles(level, tile_size=tile_size, overlap=overlap, halo=halo, roi=roi,
                            physical=bool(args.physical), max_tile_voxels=args.max_tile_voxels, resize=args.resize)

    counts = " x ".join(f"{plan.counts[ax]} ({ax})" for ax in plan.axes)
    print(f"{len(plan.tiles)} tile(s): {counts}; tile size {plan.tile_size}, overlap {plan.overlap}, "
          f"halo {plan.halo}; largest read {plan.max_load_voxels} voxels (halo included)")
    if args.dry_run:
        print("Dry run: nothing written.")
        return
    name = Path(str(args.input_path).rstrip("/")).name
    source_id = args.source_id or (name[:-len(".ome.zarr")] if name.endswith(".ome.zarr") else Path(name).stem)
    write_tiles_csv(plan, args.output_csv, source_id, str(args.input_path))
    print(f"Wrote {args.output_csv}")


def _split_names(cell, column: str) -> List[str]:
    """Space separated names of a batch CSV cell; quote a name with spaces: "Alexa 488" Brightfield."""
    try:
        return shlex.split(str(cell))
    except ValueError:
        raise ValueError(
            f"Unbalanced quote in the {column!r} cell {str(cell)!r} of the batch CSV. Quote names with spaces "
            f"with double quotes, e.g. \"Alexa 488\" Brightfield (Excel drops a leading ' when it saves)."
        ) from None


def convert_mosaic_batch(args):
    """Runmode to fuse a batch of mosaics listed in a CSV file."""
    csv_path = getattr(args, "input_file", args)  # parsed arguments or a path
    print(f"Fusing a batch of mosaics from {csv_path}")
    database = pd.read_csv(csv_path)
    print(database)
    columns = database.columns
    not_fused = []
    skipped = []
    for _, v in database.iterrows():
        print("-" * 20)
        kwargs = {
            "input_path": v["input"],
            "zarr_path": v["output"],
            "microscope": v["microscope"] if "microscope" in columns and _present(v.get("microscope")) else None,
            **_batch_row_kwargs(v, columns),
        }
        for key in ("downsample", "read_downsample"):  # read_downsample: the earlier column name
            if key in columns and _present(v.get(key)):
                kwargs["read_downsample"] = _int_triplet(str(v[key]))
        for key in ("transforms_only", "register_cameras"):
            if key in columns and _present(v.get(key)):
                kwargs[key] = parse_bool(v[key])
        for key in ("projection", "blend", "midline_axis", "out_dtype", "data_type", "label_offsets",
                    "label_mapping", "temp_dir", "transforms", "register_model", "register_channel"):
            if key in columns and _present(v.get(key)):
                kwargs[key] = str(v[key]).strip()
        for key in ("channels", "groups", "cameras", "fields", "tiles", "skip_tiles"):
            if key in columns and _present(v.get(key)):
                kwargs[key] = _split_names(v[key], key)
        # float: pandas reads a number in a column with empty cells as 3.0
        if "timepoints" in columns and _present(v.get("timepoints")):
            kwargs["timepoints"] = [int(float(t)) for t in str(v["timepoints"]).split()]
        if "reference_timepoint" in columns and _present(v.get("reference_timepoint")):
            kwargs["reference_timepoint"] = _timepoint_cell(v["reference_timepoint"])
        if "read_workers" in columns and _present(v.get("read_workers")):
            kwargs["read_workers"] = int(v["read_workers"])
        if "sigmoid_steepness" in columns and _present(v.get("sigmoid_steepness")):
            kwargs["sigmoid_steepness"] = float(v["sigmoid_steepness"])
        if "flatfield" in columns and _present(v.get("flatfield")):
            kwargs["flatfield"] = parse_bool(v["flatfield"])
        kwargs["dry_run"] = bool(getattr(args, "dry_run", False))
        try:
            if not mosaic_convert(**kwargs):
                not_fused.append(str(v["output"]))
        except FileExistsError as error:
            print(f"WARNING: row skipped: {error}", file=sys.stderr)
            skipped.append(str(v["output"]))
    if not_fused:
        print(f"ERROR: {len(not_fused)} row(s) not fused (tiles from several fields): {', '.join(not_fused)}")
    _warn_rows_skipped(skipped)


# ---------------------------------------------------------------------------
# Chromatic shift: chromatic-shift / chromatic-shift --batch
# ---------------------------------------------------------------------------

#: Options of ``estimate_chromatic_shift`` exposed on the command line.
_ESTIMATE_OPTIONS = ("timepoint", "max_shift", "coarse_level", "refine_level", "n_regions", "crop_yx",
                     "min_z_normal", "min_correlation", "threshold")


def _timepoint_cell(value) -> Union[int, str]:
    """A timepoint or ``all`` from a batch CSV cell (pandas reads 3 in a column with empty cells as 3.0)."""
    text = str(value).strip().lower()
    return "all" if text == "all" else int(float(text))


def _channel_index(channel, channel_names: List[str]) -> int:
    """Channel given as an index (``1``) or a name (``"cFos-647"``)."""
    text = str(channel).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    if text in channel_names:
        return channel_names.index(text)
    raise ValueError(f"Reference channel {channel!r} not found; channels are {channel_names}.")


def _parse_shifts(value) -> Any:
    """``--shifts`` values or batch cell: one number per channel, or the path to a JSON report."""
    tokens = [str(v) for v in value] if isinstance(value, (list, tuple)) else str(value).split()
    if len(tokens) == 1 and Path(tokens[0]).is_file():
        import json

        return json.loads(Path(tokens[0]).read_text())
    try:
        return [float(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(
            f"Invalid shifts {value!r}: give one number per channel or the path to a JSON report."
        ) from exc


def _print_shift_report(report: Dict[str, Any]) -> None:
    names = report.get("channel_names") or []
    unit = report.get("unit") or ""
    ref = report.get("reference_channel", 0)
    for c, shift in enumerate(report["shifts"]):
        name = names[c] if c < len(names) else f"channel {c}"
        if c == ref:
            print(f"  {name}: reference")
            continue
        surfaces = (report.get("surface_shifts") or [None] * (c + 1))[c] or {}
        print(
            f"  {name}: {shift:+.2f} {unit} (coarse {report['coarse_shifts'][c]:+.2f}, "
            f"{report['n_regions_used'][c]} regions, spread {report['spread'][c]}, "
            f"lower {surfaces.get('lower')}, upper {surfaces.get('upper')})"
        )
    for warning in report.get("warnings") or []:
        print(f"  WARNING: {warning}")


def chromatic_shift_correct(
    input_path: str,
    zarr_path: Optional[str] = None,
    *,
    report_path: Optional[str] = None,
    shifts: Any = None,
    reference_channel: Any = 0,
    extent: str = "union",
    order: int = 1,
    include_groups: bool = True,
    include_labels: bool = True,
    overwrite: bool = False,
    num_workers: Optional[int] = None,
    fail_on_warning: bool = False,
    **estimate_kwargs,
) -> Dict[str, Any]:
    """Measure the Z chromatic shift of an OME-Zarr image and write a corrected copy.

    Without ``zarr_path`` only the estimate runs (and is saved to
    ``report_path`` when given). ``shifts`` (one value per channel, a report
    dict or the path to a JSON report) skips the estimate. With
    ``fail_on_warning`` a report with warnings stops before anything is
    written except the report. Returns the report, or the given shifts. An existing
    output or report raises ``FileExistsError`` unless ``overwrite`` is given.
    """
    if zarr_path is not None:
        _refuse_output(zarr_path, overwrite=overwrite, input_path=input_path)
    if report_path is not None and Path(report_path).exists() and not overwrite:
        raise FileExistsError(f"{report_path} already exists; {_OVERWRITE_HINT} to replace it, or choose another path.")

    z = mm.ZarrManager(path=str(input_path))
    names = list(z.metadata.get("channel_names") or [])

    if shifts is not None:
        report = shifts if isinstance(shifts, dict) else _parse_shifts(shifts)
        print(f"Using the given shifts: {report['shifts'] if isinstance(report, dict) else report}")
    else:
        ref = _channel_index(reference_channel, names)
        options = {k: v for k, v in estimate_kwargs.items() if k in _ESTIMATE_OPTIONS and v is not None}
        if options.get("refine_level") == "none":
            options["refine_level"] = None
        print(f"\n--->Estimating the chromatic shift of {input_path} (reference: {names[ref] if ref < len(names) else ref}, "
              f"timepoint: {options.get('timepoint', 0)})")
        report = z.estimate_chromatic_shift(reference_channel=ref, **options)
        _print_shift_report(report)

    if report_path is not None:
        import json

        if Path(report_path).exists():
            print(f"WARNING: replacing the existing report {report_path}", file=sys.stderr)
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2))
        print(f"Report saved to {report_path}")

    if fail_on_warning and isinstance(report, dict) and report.get("warnings"):
        raise SystemExit(f"Chromatic shift estimate of {input_path} has warnings; nothing written (--fail_on_warning).")

    if zarr_path is None:
        return report

    print(f"\n--->Writing the corrected dataset to {zarr_path}")
    z.apply_chromatic_shift(
        zarr_path,
        report,
        extent=extent,
        order=int(order),
        include_groups=bool(include_groups),
        include_labels=bool(include_labels),
        overwrite=bool(overwrite),
        num_workers=num_workers,
    )
    print(f"Saved {zarr_path}")
    return report


def _chromatic_shift_kwargs(args) -> Dict[str, Any]:
    kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in _NOT_CONVERSION_OPTIONS and v is not None and not (isinstance(v, str) and v.strip() == "")
    }
    if "report" in kwargs:
        kwargs["report_path"] = kwargs.pop("report")
    return kwargs


def chromatic_shift_single(args):
    """Runmode to correct the chromatic shift of one OME-Zarr image."""
    chromatic_shift_correct(**_chromatic_shift_kwargs(args))


def chromatic_shift_batch(args):
    """Runmode to correct the chromatic shift of every OME-Zarr image listed in a CSV file."""
    csv_path = getattr(args, "input_file", args)  # parsed arguments or a path
    print(f"Correcting the chromatic shift of a batch from {csv_path}")
    database = pd.read_csv(csv_path)
    print(database)
    columns = database.columns

    def cell(v, key):
        return v.get(key) if key in columns and _present(v.get(key)) else None

    skipped = []
    for _, v in database.iterrows():
        print("-" * 20)
        kwargs: Dict[str, Any] = {"input_path": v["input"], "zarr_path": cell(v, "output")}
        for key, cast in (("report", str), ("shifts", str), ("reference_channel", str), ("extent", str),
                          ("refine_level", str), ("timepoint", int), ("coarse_level", int), ("n_regions", int),
                          ("crop_yx", int), ("order", int), ("num_workers", int), ("max_shift", float),
                          ("min_z_normal", float), ("min_correlation", float), ("threshold", float)):
            value = cell(v, key)
            if value is not None:
                kwargs["report_path" if key == "report" else key] = cast(value).strip() if cast is str else cast(value)
        if kwargs.get("refine_level") not in (None, "none"):
            kwargs["refine_level"] = int(kwargs["refine_level"])
        for key in ("include_groups", "include_labels", "overwrite", "fail_on_warning"):
            value = cell(v, key)
            if value is not None:
                kwargs[key] = parse_bool(value)
        try:
            chromatic_shift_correct(**kwargs)
        except FileExistsError as error:
            print(f"WARNING: row skipped: {error}", file=sys.stderr)
            skipped.append(str(v["input"]))
    _warn_rows_skipped(skipped)


# ---------------------------------------------------------------------------
# Pixel size: resample / resample --batch
# ---------------------------------------------------------------------------

def resample_zarr(
    input_path: str,
    zarr_path: str,
    pixel_size: Any,
    *,
    fast: bool = False,
    include_groups: bool = True,
    include_labels: bool = True,
    overwrite: bool = False,
    num_workers: Optional[int] = None,
) -> None:
    """Write a copy of an OME-Zarr image with a new pixel size (one value, or three in Z Y X order)."""
    _refuse_output(zarr_path, overwrite=overwrite, input_path=input_path)
    sizes = [float(v) for v in (str(pixel_size).split() if isinstance(pixel_size, str) else pixel_size)]
    if len(sizes) not in (1, 3):
        raise ValueError(f"Invalid pixel size {pixel_size!r}: give one value, or three in Z Y X order.")

    z = mm.ZarrManager(path=str(input_path))
    print(f"\n--->Resampling {input_path} from {z.metadata['scales'][0]} to {sizes} ({z.metadata.get('units')})")
    z.resample_to_zarr(
        zarr_path,
        sizes[0] if len(sizes) == 1 else sizes,
        fast=bool(fast),
        include_groups=bool(include_groups),
        include_labels=bool(include_labels),
        overwrite=bool(overwrite),
        num_workers=num_workers,
    )
    print(f"Saved {zarr_path}")


def resample_single(args):
    """Runmode to resample one OME-Zarr image."""
    resample_zarr(**{k: v for k, v in vars(args).items() if k not in _NOT_CONVERSION_OPTIONS})


def resample_batch(args):
    """Runmode to resample every OME-Zarr image listed in a CSV file."""
    csv_path = getattr(args, "input_file", args)  # parsed arguments or a path
    print(f"Resampling a batch from {csv_path}")
    database = pd.read_csv(csv_path)
    print(database)
    columns = database.columns

    skipped = []
    for _, v in database.iterrows():
        print("-" * 20)
        kwargs: Dict[str, Any] = {"input_path": v["input"], "zarr_path": v["output"], "pixel_size": str(v["pixel_size"])}
        for key in ("fast", "include_groups", "include_labels", "overwrite"):
            if key in columns and _present(v.get(key)):
                kwargs[key] = parse_bool(v[key])
        if "num_workers" in columns and _present(v.get("num_workers")):
            kwargs["num_workers"] = int(v["num_workers"])
        try:
            resample_zarr(**kwargs)
        except FileExistsError as error:
            print(f"WARNING: row skipped: {error}", file=sys.stderr)
            skipped.append(str(v["input"]))
    _warn_rows_skipped(skipped)


# ---------------------------------------------------------------------------
# Empty stores and in-place pyramids: allocate / pyramid
# ---------------------------------------------------------------------------

def _single_or_tuple(values):
    """A one-value CLI list as a scalar, else a tuple (``None`` stays ``None``)."""
    if values is None:
        return None
    values = list(values)
    return values[0] if len(values) == 1 else tuple(values)


def allocate_single(args):
    """Runmode creating an empty OME-Zarr image from a template or an explicit geometry."""
    from zarrsmith.utils.allocate import allocate_zarr

    _refuse_output(args.out, overwrite=args.overwrite, input_path=args.template)
    translation = args.translation
    if translation is not None:
        translation = True if len(translation) == 0 else tuple(translation)
    compression = _compression_kwargs(args.compressor, args.compressor_level)
    units = None if args.units is None else (args.units[0] if len(args.units) == 1 else tuple(args.units))
    root = allocate_zarr(
        args.out,
        template=args.template,
        size=args.size,
        axes=args.axes,
        scale=_single_or_tuple(args.scale),
        units=units,
        time_increment=args.time_increment,
        dtype=args.dtype,
        data_type=args.data_type,
        channel_names=args.channel_names,
        channel_colors=args.channel_colors,
        num_channels=args.num_channels,
        crop=args.crop,
        chunk_size=args.chunk_size,
        shards=_normalize_shards(args.shards),
        shard_target_mb=args.shard_target_mb,
        num_levels=args.num_levels,
        downscale_factor=_single_or_tuple(args.downscale_factor),
        start_level=args.start_level,
        translation=translation,
        zarr_format=args.zarr_format,
        extra_attributes=args.attrs,
        name=args.name,
        overwrite=args.overwrite,
        **compression,
    )
    level0 = root["0"]
    print(f"Created {args.out}: zarr v{root.metadata.zarr_format}, shape {tuple(level0.shape)}, "
          f"dtype {level0.dtype}, chunks {tuple(level0.chunks)}, shards {level0.shards}")


def _pyramid_groups(z, groups) -> List[Optional[str]]:
    """Group paths of ``zarrsmith pyramid --group`` (``None`` is the root image)."""
    if not groups:
        return [None]
    if any(str(g).lower() == "all" for g in groups):
        return [None] + list(z.groups) + [f"labels/{name}" for name in z.labels]
    return [None if g in ("/", "") else g for g in groups]


def pyramid_single(args):
    """Runmode building the pyramid of an OME-Zarr image in place."""
    z = mm.ZarrManager(path=str(args.zarr_path), mode="r+")
    for group in _pyramid_groups(z, args.group):
        print(f"\n--->Building the pyramid of {args.zarr_path}{'' if group is None else '/' + group}")
        z.populate_pyramid(
            group=group,
            num_levels=args.num_levels,
            downscale_factor=_single_or_tuple(args.downscale_factor),
            method=args.method,
            start_level=args.start_level,
            num_workers=args.num_workers,
        )
    print(f"Saved {args.zarr_path}")


def main(argv=None):
    """Entry point of the ``zarrsmith`` command: ``convert``, ``fuse``, ``resample``, ``chromatic-shift``,
    ``plan-tiles``, ``allocate``, ``fuse-labels`` or ``pyramid``."""
    args = _parse_arguments(argv)
    if args.renamed_to:
        print(f"Note: `zarrsmith {args.command}` is now `{args.renamed_to}`; the old name still works for now.",
              file=sys.stderr)
    runners = {
        "convert": convert_single,
        "convert_batch": convert_batch,
        "fuse": convert_mosaic_single,
        "fuse_batch": convert_mosaic_batch,
        "chromatic_shift": chromatic_shift_single,
        "chromatic_shift_batch": chromatic_shift_batch,
        "resample": resample_single,
        "resample_batch": resample_batch,
        "allocate": allocate_single,
        "pyramid": pyramid_single,
        "plan_tiles": plan_tiles_single,
        "fuse_labels": fuse_labels_single,
    }
    runners[args.mode](args)

if __name__ == "__main__":
    main()