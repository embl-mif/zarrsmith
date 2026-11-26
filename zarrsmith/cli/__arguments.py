__all__ = ['_parse_arguments']

import argparse
import os
import re
import sys
import textwrap

from zarrsmith import __version__
from zarrsmith.utils.colors import parse_color as _parse_color
from zarrsmith.utils.ngff import DEFAULT_SHARD_TARGET_MB


class MultilineDefaultsHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Formatter that preserves line breaks while still displaying defaults."""
    pass


# Valid type of's
def valid_input_path(x):
    """Validate an existing input file or directory argument."""
    if x is not None:
        if os.path.isdir(x) or os.path.isfile(x):
            return os.path.abspath(x)
        raise argparse.ArgumentTypeError(f'Input path {x} is not a valid directory')
    return None


def parse_color(value: str) -> str:
    """A CLI color (6-digit hex code, ``#`` optional, or a matplotlib color name) as ``RRGGBB``;
    see :func:`zarrsmith.utils.colors.parse_color`."""
    try:
        return _parse_color(value)
    except TypeError as error:
        raise argparse.ArgumentTypeError(str(error)) from None

def parse_downscale_factor(value: str):
    """Parse a scalar or anisotropic downscale factor from a CSV cell.

    Examples
    --------
    ``2`` -> 2
    ``1 2 2`` -> (1, 2, 2)
    ``1,2,2`` -> (1, 2, 2)
    """
    if value is None:
        return 2

    text = str(value).strip()
    if text in {'', '-1', 'none', 'null', 'nan'}:
        return 2

    parts = [p for p in re.split(r'[\s,;]+', text) if p]
    if not parts:
        return 2

    try:
        factors = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid downscale factor '{value}'. Use a scalar like '2' or a sequence like '1 2 2'."
        ) from exc

    if any(f <= 0 for f in factors):
        raise argparse.ArgumentTypeError(
            f"Invalid downscale factor '{value}'. All values must be positive integers."
        )

    return factors[0] if len(factors) == 1 else tuple(factors)


def parse_bool(value) -> bool:
    """Parse a --drop_singleton value or batch CSV cell into a bool."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use true or false."
    )


def parse_shards_spec(value):
    """Parse a --shards value or batch CSV cell into "auto" or a shard shape.

    Examples
    --------
    ``auto`` -> ``"auto"``
    ``1 1 8 2160 4096`` -> ``(1, 1, 8, 2160, 4096)``
    ``1,1,8,2160,4096`` -> ``(1, 1, 8, 2160, 4096)``
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        tokens = [str(v).strip() for v in value]
        if len(tokens) == 1 and tokens[0].lower() in {"none", "null", "nan", "-1", ""}:
            return None
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null", "nan", "-1"}:
            return None
        tokens = [p for p in re.split(r'[\s,;]+', text) if p]

    if len(tokens) == 1 and tokens[0].lower() == "auto":
        return "auto"

    try:
        return tuple(int(t) for t in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid shards value '{value}'. Use 'auto' or a shard shape matching "
            "the dataset's axes, e.g. '1 1 8 2160 4096'."
        ) from exc


def parse_shard_exclude_axes_spec(value):
    """Parse a --shard_exclude_axes value or batch CSV cell into an axis tuple.

    Examples
    --------
    ``t c`` -> ``('t', 'c')``
    ``none`` -> ``()`` (allow every axis to be merged by shards="auto")
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        tokens = [str(v).strip().lower() for v in value]
    else:
        text = str(value).strip()
        if text == "":
            return None
        tokens = [p for p in re.split(r'[\s,;]+', text) if p]
        tokens = [t.lower() for t in tokens]

    if len(tokens) == 1 and tokens[0] in {"none", "null", "nan", "-1"}:
        return ()

    valid = {"t", "c", "z", "y", "x"}
    invalid = [t for t in tokens if t not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid shard_exclude_axes {invalid!r}. Must be a subset of t, c, z, y, x, "
            "or 'none' to allow every axis to merge."
        )
    return tuple(tokens)


def _parse_subset_selection(selection: str, axis: str):
    """Parse a per-axis selection from a CLI subset specification."""
    text = selection.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    if text == "" or text.lower() in {"all", "none", "null", "nan", "*"}:
        return slice(None)

    if axis == "c" and ":" not in text:
        # channels by name (c=DAPI,GFP), or names and indices mixed; numbers stay indices
        tokens = [part.strip().strip('"').strip("'") for part in text.split(",") if part.strip()]
        if any(not part.lstrip("-").isdigit() for part in tokens):
            return [int(part) if part.lstrip("-").isdigit() else part for part in tokens]

    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) > 3:
            raise argparse.ArgumentTypeError(
                f"Invalid subset selection '{selection}' for axis '{axis}'. "
                "Use 'start:stop[:step]', a comma-separated index list, or a single integer."
            )

        def _maybe_int(value: str):
            return None if value == "" else int(value)

        try:
            start = _maybe_int(parts[0])
            stop = _maybe_int(parts[1]) if len(parts) > 1 else None
            step = _maybe_int(parts[2]) if len(parts) > 2 else None
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid subset selection '{selection}' for axis '{axis}'. "
                "Slice bounds must be integers."
            ) from exc

        if step == 0:
            raise argparse.ArgumentTypeError(
                f"Invalid subset selection '{selection}' for axis '{axis}'. Step cannot be 0."
            )
        return slice(start, stop, step)

    if "," in text:
        try:
            return [int(part.strip()) for part in text.split(",") if part.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid subset selection '{selection}' for axis '{axis}'. "
                "Use integers separated by commas."
            ) from exc

    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid subset selection '{selection}' for axis '{axis}'. "
            "Use a single integer, a comma-separated list, or a slice like '0:10:2'."
        ) from exc


def parse_subset_spec(value: str):
    """Parse a compact axis subset string into kwargs for subset_dataset().

    Examples
    --------
    ``y=10:100:2;x=20:80`` -> ``{"Y": slice(10, 100, 2), "X": slice(20, 80, None)}``
    ``z=0:10;c=0,2`` -> ``{"Z": slice(0, 10, None), "C": [0, 2]}``
    ``c=DAPI,GFP`` -> ``{"C": ["DAPI", "GFP"]}`` (channel names, resolved against the dataset)
    """
    if value is None:
        return None

    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "nan", "-1"}:
        return None

    subset = {}
    for raw_spec in re.split(r"[;\n]+", text):
        spec = raw_spec.strip()
        if not spec:
            continue
        if "=" not in spec:
            raise argparse.ArgumentTypeError(
                f"Invalid subset specification '{value}'. "
                "Use axis=selection pairs separated by semicolons, e.g. 'y=10:100:2;x=20:80'."
            )
        axis, selection = spec.split("=", 1)
        axis = axis.strip().lower()
        if axis not in {"t", "c", "z", "y", "x"}:
            raise argparse.ArgumentTypeError(
                f"Invalid axis '{axis}' in subset specification '{value}'. "
                "Supported axes are t, c, z, y, and x."
            )
        subset[axis.upper()] = _parse_subset_selection(selection, axis)

    return subset



_OVERWRITE_HELP = ('Replace an existing output. Without it a non-empty output is refused. Only a Zarr store '
                   'is ever replaced: any other non-empty folder is refused in all cases.')


# ---------------------------------------------------------------------------------------
# Help layout: every command lists its options in the same sections, in this order:
#   Input and output | Input | What to read | <the command's own sections> | Pyramid |
#   Storage layout | Channel metadata | Performance | Run
# Options marked advanced (most of Storage layout, Channel metadata, Performance, and a few
# fine-tuning options) are only shown by `zarrsmith <command> --help-all`; they work either way.
# ---------------------------------------------------------------------------------------

_ADVANCED = '_zarrsmith_advanced'


def _section(parser, title, advanced=False):
    group = parser.add_argument_group(title)
    setattr(group, _ADVANCED, advanced)
    return group


def _add(group, *flags, advanced=None, **kwargs):
    """``group.add_argument`` that marks the option as advanced (hidden without --help-all)."""
    action = group.add_argument(*flags, **kwargs)
    setattr(action, _ADVANCED, getattr(group, _ADVANCED, False) if advanced is None else advanced)
    return action


class _DeprecatedOption(argparse.Action):
    """An old spelling of an option: still works, with a warning naming the new one; not in the help."""

    def __init__(self, option_strings, dest, new=None, base='store', **kwargs):
        self.new, self.base = new, base
        if base in ('store_true', 'store_false'):
            kwargs['nargs'] = 0
        kwargs.setdefault('default', argparse.SUPPRESS)
        kwargs['help'] = argparse.SUPPRESS
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(f"Note: {option_string} is now {self.new}; the old spelling still works for now.", file=sys.stderr)
        value = {'store_true': True, 'store_false': False}.get(self.base, values)
        setattr(namespace, self.dest, value)


def _deprecated(group, *flags, dest, new, **kwargs):
    group.add_argument(*flags, dest=dest, action=_DeprecatedOption, new=new, **kwargs)


class _DtypeOrDataType(argparse.Action):
    """fuse ``-dt/--dtype``: the output dtype. ``-dt`` used to be ``--data_type``, so ``-dt label``
    and ``-dt intensity`` still set the data type, with a note."""

    def __call__(self, parser, namespace, values, option_string=None):
        if str(values).lower() in ('intensity', 'label'):
            print(f"Note: {option_string} {values} sets the data type: {option_string} is now the output dtype, "
                  f"use -dty/--data_type {values}.", file=sys.stderr)
            namespace.data_type = str(values).lower()
        else:
            namespace.out_dtype = values


def _finish_help(parser, command, show_all):
    """Hide the advanced options unless --help-all, and say how many there are."""
    hidden = [a for a in parser._actions if getattr(a, _ADVANCED, False) and a.help is not argparse.SUPPRESS]
    _add(parser._optionals, '--help-all', action='store_true',
         help='Show every option, including storage layout, channel metadata and performance options.')
    if show_all or not hidden:
        return
    for action in hidden:
        action.help = argparse.SUPPRESS
    note = f"{len(hidden)} more options (storage layout, channel metadata, performance, fine tuning): " \
           f"`zarrsmith {command} --help-all`."
    parser.epilog = f"{parser.epilog}\n\n{note}" if parser.epilog else note


# ---- Sections shared by several commands -----------------------------------------------

def _io_section(parser, *, input_help, output_help='Path to the output .ome.zarr.', batch=True,
                required=False, output_type=os.path.abspath, input_type=None):
    group = _section(parser, 'Input and output' + (' (or --batch)' if batch and not required else ''))
    _add(group, '-i', '--input_path', required=required, type=input_type or valid_input_path, help=input_help)
    _add(group, '-z', '--zarr_path', required=required, type=output_type, help=output_help)
    if batch and not required:
        _add(group, '--batch', metavar='CSV', type=valid_input_path,
             help='Run every row of this CSV file instead (see the description above).')
    return group


def _pyramid_section(parser, *, infer=False, start_level=False, defaults=None, method=True,
                     num_levels_help='Number of pyramidal levels.',
                     factor_help='Pyramid downsampling factor. Use one value for isotropic downsampling or three '
                                 'values for anisotropic Z Y X factors, e.g. -df 1 2 2.'):
    defaults = defaults or {}
    group = _section(parser, 'Pyramid')
    _add(group, '-nl', '--num_levels', type=int, metavar='N', help=num_levels_help, **defaults.get('num_levels', {}))
    _add(group, '-df', '--downscale_factor', nargs='+', type=int, metavar='FACTOR', help=factor_help,
         **defaults.get('downscale_factor', {}))
    if method:
        _add(group, '-me', '--method', choices=['mean', 'nearest', 'mode'],
             help='Downsampling method of the pyramid levels. Default: mean for intensity data, nearest for labels.')
    if infer:
        _add(group, '-idf', '--infer_downscale_factors', metavar='IMAGE', type=valid_input_path,
             help='Make the pyramid follow this OME-Zarr image from --start_level on: the same number of '
                  'levels below it and the same factor between them, so level k of the output matches level '
                  'start + k of the image (e.g. labels of an image segmented at a coarser level). '
                  '--num_levels/--downscale_factor override it.')
        _add(group, '-sl', '--start_level', type=int, default=0, metavar='LEVEL',
             help='Level of --infer_downscale_factors that the output\'s level 0 matches.')
    if start_level:
        _add(group, '-sl', '--start_level', default='0', help=start_level)
    return group


def _storage_section(parser, *, chunk_help='Chunk size in TCZYX format, or whatever axes are present in the dataset.',
                     max_size_help='Max chunk size in MB (uncompressed). Default: about 4 MB, about cubic in '
                                   'micrometers (from the voxel size) with at most 32 z planes. Ignored if '
                                   '--chunk_size is provided.',
                     shard_target_help='Target uncompressed shard size in MB, used when --shards auto. Default: '
                                       '512 for convert, 256 for fuse (each fusion worker holds one shard).'):
    group = _section(parser, 'Storage layout', advanced=True)
    _add(group, '-zf', '--zarr_format', choices=[2, 3], type=int, advanced=False,
         help='Output zarr format. Zarr v2 maps to NGFF 0.4 and Zarr v3 maps to NGFF 0.5.')
    _add(group, '-cs', '--chunk_size', nargs='+', type=int, advanced=False, help=chunk_help)
    _add(group, '-ms', '--max_size', type=float, help=max_size_help)
    _add(group, '-sh', '--shards', nargs='+',
         help='Zarr v3 sharding for the output. Use "auto" to size shards automatically, a shard shape matching '
              'the dataset axes, e.g. "-sh 1 1 8 2160 4096", or "none" to disable sharding. With --zarr_format 2 '
              'sharding is not supported and is set to none with a warning. Default: auto.')
    _add(group, '-stm', '--shard_target_mb', type=float, help=shard_target_help)
    _add(group, '-sea', '--shard_exclude_axes', nargs='+',
         help='Axes that --shards auto never merges chunks along, e.g. "-sea t c". Use "none" to allow every '
              'axis to merge. Default: t c.')
    _add(group, '-cp', '--compressor', type=str.lower, choices=['blosc', 'gzip', 'none'],
         help='Codec of the output. "blosc" is Blosc with zstd and byte shuffle, "gzip" needs --zarr_format 2, '
              '"none" stores the chunks uncompressed. Default: blosc.')
    _add(group, '-cpl', '--compressor_level', type=int,
         help='Compression level, 0 to 9. Higher levels write much more slowly and rarely save much space. Default: 1.')
    _add(group, '-ds', '--drop_singleton', type=parse_bool,
         help='Remove singleton t/c/z axes from the output ("true" or "false"). Default: true.')
    return group


def _channel_metadata_section(parser, names_help='Name of channels. Example: -cn bf gfp rfp'):
    group = _section(parser, 'Channel metadata', advanced=True)
    _add(group, '-cn', '--channel_names', nargs='+', help=names_help)
    _add(group, '-cc', '--channel_colors', type=parse_color, nargs='+',
         help='Color(s) of channel(s). It needs to be hex or matplotlib color name. Example: -cc 0000FF cyan 00ff00')
    return group


def _performance_section(parser, *, read_workers=False, temp_dir=False, num_workers_default=None,
                         num_workers_help='Number of threads used to write the zarr in parallel. Default: half of '
                                          'the available cores, capped at 8.'):
    group = _section(parser, 'Performance', advanced=True)
    kwargs = {} if num_workers_default is None else {'default': num_workers_default}
    _add(group, '-nw', '--num_workers', type=int, help=num_workers_help, **kwargs)
    if read_workers:
        _add(group, '-rw', '--read_workers', type=int, default=0,
             help='Regions fused in parallel (worker processes). Default: 0, chosen from the cores, the regions and '
                  'the free memory, as in the napari widget. Not to be confused with -nw (write threads).')
    if temp_dir:
        _add(group, '-td', '--temp_dir', help=argparse.SUPPRESS)  # deprecated and ignored
    return group


def _run_section(parser, *, dry_run=True, overwrite=True, legacy_overwrite=False, dry_run_help=None):
    group = _section(parser, 'Run')
    if dry_run:
        _add(group, '-dr', '--dry_run', action='store_true', help=dry_run_help or (
            'Print what would be written and stop, writing nothing: the output (axes, size, voxel), its chunks '
            'and shards (size and count), the pyramid levels and, for fuse, the fusion workers; the same recap '
            'as the napari widgets. Only metadata is read (fuse: camera registration is skipped). With --batch: '
            'every row.'))
    if legacy_overwrite:
        _add(group, '-ow', '--overwrite', type=parse_bool, default=False,
             help='Replace an existing output ("true" or "false").')
    elif overwrite:
        _add(group, '-ow', '--overwrite', action='store_true', help=_OVERWRITE_HELP)
    return group


# ---- convert ----------------------------------------------------------------------------

CONVERT_HELP = """\
    Convert a dataset to OME-Zarr, or every dataset listed in a CSV file (--batch).

    Examples:
      zarrsmith convert -i /path/to/dataset -z out.ome.zarr
      zarrsmith convert -i /path/to/file.czi -m zeiss -si 1 -z out.ome.zarr -nl 4 -sh auto
      zarrsmith convert --batch datasets.csv

    The --batch CSV file has the columns input | microscope | output, plus any of
    chunk_size | max_size(MB) | scene_index | zarr_format | downscale_factor | subset | channel_colors |
    channel_names | num_levels | method | shards | shard_target_mb | shard_exclude_axes | compressor | compressor_level |
    drop_singleton | num_workers | overwrite
    with the same meaning as the options below, e.g.
      input            | microscope | output          | chunk_size    | zarr_format | downscale_factor | subset             | num_levels | shards
      /path/to/input_1 | opera      | /path/to/zarr_1 | 1 1 2 512 512 | 3           | 1 2 2            | y=10:100:2;x=20:80 | 3          | auto
      /path/to/input_2 | viventis   | /path/to/zarr_2 |               | 2           | 2                | z=0:10;c=0,2       |            |
    Leave a cell empty for the default. channel_colors can be hex codes or matplotlib color names.
    shards: "auto" (default when empty), an explicit shard shape, or "none". Not supported with zarr_format 2.
    shard_target_mb only applies to shards=auto (default 512). shard_exclude_axes defaults to "t c"
    ("none" lets every axis merge). compressor: "blosc" (default when empty), "gzip" (zarr_format 2 only) or
    "none", in lowercase; compressor_level goes from 0 to 9 (default 1).
    drop_singleton (true/false, default true) removes singleton t/c/z axes.
    method: "mean", "nearest" or "mode" (default when empty: mean for intensity data, nearest for labels).
    num_workers is the number of write threads (default: half of the available cores, capped at 8).
    overwrite (true/false, default false) replaces an existing output; a row whose output is in the way is
    skipped with a warning, and the other rows still run.
"""

CONVERT_MICROSCOPES = ['luxendo', 'opera', 'viventis', 'zeiss', 'evident', 'zarrv04', 'zarr', 'scape']


def _add_convert_arguments(parser, *, legacy=False):
    _io_section(parser, required=legacy, batch=not legacy,
                input_help='Path to input file.' if legacy else 'Path to the input dataset (file or folder).')
    read = _section(parser, 'What to read')
    _add(read, '-m', '--microscope', default=None, choices=CONVERT_MICROSCOPES, type=str,
         help='Microscope used for imaging. Leave empty to auto-detect from the input path.')
    _add(read, '-si', '--scene_index', type=int, help='Scene index for .czi and .vsi files. Default: the first scene.')
    _add(read, '-s', '--subset', type=str,
         help='Subset the input before pyramid generation. Format: axis=selection pairs separated by semicolons, '
              'e.g. "y=10:100:2;x=20:80;c=DAPI,GFP". Use integers, comma-separated indices, or slices like '
              '0:10:2; channels (c) also by name.')
    _pyramid_section(parser)
    _storage_section(parser)
    _channel_metadata_section(parser)
    _performance_section(parser)
    _run_section(parser)


# ---- fuse -------------------------------------------------------------------------------

FUSE_HELP = """\
    Fuse the raw tiles of a mosaic (a raw Luxendo acquisition, an Opera Harmony export, a Zeiss .czi,
    a folder of OME-Zarr stores) into one OME-Zarr image, 3D or a 2D projection (-p max), or every
    mosaic listed in a CSV file (--batch). The output options are those of `zarrsmith convert`;
    -s/--subset crops the fused output (z/y/x in output voxels, after --downsample) and selects
    channels (c). Label tiles of a tiled segmentation: see `zarrsmith fuse-labels`.

    Examples:
      zarrsmith fuse -i /path/to/2026-10-14_103957 -z out_3D.ome.zarr -rds 4 8 8 -nl 3
      zarrsmith fuse -i /path/to/2026-10-14_103957 -z out_MIP.ome.zarr -rds 1 8 8 -p max -b midline
      zarrsmith fuse -i /path/to/2026-10-14_103957 -z field1_MIP.ome.zarr -rds 1 8 8 -p max -fi field1 -st 3-x01-y00
      zarrsmith fuse -i "/path/to/plate-Measurement 1" -z A3_MIP.ome.zarr -p max -fi A3_field1
      zarrsmith fuse -i /path/to/timelapse -z out_t.ome.zarr -rds 4 8 8 -tp 0 1 2 3 -rtp 0
      zarrsmith fuse --batch mosaics.csv

    Tiles that touch or overlap form a field (field0, field1, ... ordered by position; for Opera plates
    one set per well: A3_field0, A3_field1, ...); the fields are listed when the mosaic is loaded. Only
    one field is fused at a time: the first one by default, another with -fi. Tiles from separate fields
    are never fused together: an error is printed and the fusion is not done (in a batch, the other
    rows still run). Opera tiles are flat-field corrected as in Harmony, unless -nff is given.

    A mosaic with several timepoints gives an output with a t axis. The geometry comes from one
    reference timepoint (-rtp, default: the first) and every timepoint is fused with it; -rtp all places
    every timepoint at its own stage positions instead. -tp selects the timepoints to fuse.

    MuVi SPIM acquisitions (two opposing cameras) get one field per stack (sample): field0 for stack 0
    (0-x00-y00, 0-x01-y00, ...). The two camera views are fused into one image: the right one is
    mirrored back, and they are blended along z, each camera on the side it images sharply. Tile ids
    then end with the camera (DAPI/0-x00-y00/right); DAPI/0-x00-y00 still names both cameras.

    Before fusing, the position of every tile at every timepoint is written to
    <output>.transforms.json (fused.ome.zarr -> fused.transforms.json). -tfo only writes this file;
    -tf FILE fuses with the positions of such a file instead of the stage positions.

    The --batch CSV file has the columns input | microscope | output, plus any of the output columns of
    `zarrsmith convert --batch` (chunk_size | max_size(MB) | zarr_format | downscale_factor | subset | channel_colors |
    channel_names | num_levels | method | shards | shard_target_mb | shard_exclude_axes | compressor | compressor_level |
    drop_singleton | num_workers | overwrite)
    and of the fusion columns
    downsample | projection | blend | sigmoid_steepness | channels | groups | cameras | fields | tiles | skip_tiles | timepoints |
    reference_timepoint | midline_axis | out_dtype | data_type | label_offsets | label_mapping | read_workers | temp_dir |
    transforms | transforms_only (true/false) | register_cameras (true/false) | register_model | register_channel |
    flatfield (true/false, Opera only, default true), e.g.
      input                  | microscope | output                 | downsample | projection | blend   | num_levels
      /path/to/acquisition_1 | luxendo    | /path/to/acq1_3D.zarr  | 4 8 8      | none       | sigmoid | 3
      /path/to/acquisition_1 | luxendo    | /path/to/acq1_MIP.zarr | 1 8 8      | max        | midline |
    Leave a cell empty for the default. Lists (channels, groups, fields, tiles, skip_tiles, timepoints, channel_names...) are space
    separated; put a name that contains spaces in double quotes in channels, groups, cameras, fields, tiles and
    skip_tiles, e.g. "Alexa 488" Brightfield.
    The napari "Mosaic Fusion" widget writes such files with "Append to batch CSV".
"""

FUSE_MICROSCOPES = ['luxendo', 'opera', 'zeiss', 'zarr']

_MOSAIC_STORAGE_HELP = dict(
    chunk_help='Output chunk size in C Z Y X (C Y X for projections) order, after T for several timepoints.',
    max_size_help='Max output chunk size in MB (default about 4 MB). Ignored if --chunk_size is provided.',
)


def _add_fuse_arguments(parser, *, legacy=False):
    _io_section(parser, required=legacy, batch=not legacy,
                input_help='Acquisition folder.' if legacy else
                'Acquisition folder (for Luxendo: the folder containing raw/; for Opera: the Harmony export '
                'containing Images/).')

    read = _section(parser, 'What to read')
    _add(read, '-m', '--microscope', default=None, choices=FUSE_MICROSCOPES, type=str,
         help='Microscope of the mosaic. Leave empty to auto-detect (raw Luxendo acquisitions, Opera Harmony '
              'exports with Images/Index.xml, Zeiss .czi files, folders of OME-Zarr stores placed at their '
              'NGFF translation).')
    _add(read, '-si', '--scene_index', type=int,
         help='Scene of a Zeiss mosaic to load (default: all of them, one field per scene).')
    _add(read, '-nff', '--no_flatfield', dest='flatfield', action='store_false',
         help='Opera: do not apply the flat-field correction stored by Harmony (applied by default, as in the '
              'images stitched by Harmony).')
    _add(read, '-ch', '--channels', nargs='+', help='Tile channels to fuse, in output order. Default: all.')
    _add(read, '-gr', '--groups', nargs='+', help='Tile groups to fuse (e.g. L R). Default: all.')
    _deprecated(read, '-g', dest='groups', new='-gr/--groups', nargs='+')
    _add(read, '-cam', '--cameras', nargs='+',
         help='Cameras to fuse (MuVi SPIM: left right). Default: all, blended into one image.')
    _add(read, '-fi', '--fields', nargs='+',
         help='Field to fuse (the output fits it): fields are groups of touching tiles, named field0, field1, '
              '... by position and listed when the mosaic is loaded. Tiles from several fields are not fused '
              '(an error is printed). Default: the first field (field0), unless --tiles is given.')
    _deprecated(read, '-f', dest='fields', new='-fi/--fields', nargs='+')
    _add(read, '-tl', '--tiles', nargs='+',
         help='Only fuse these tiles (the output fits them): stacks such as 1-x00-y00 (every channel), or '
              '"channel/stack" ids such as cFos-647/1-x00-y00. Default: all.')
    _deprecated(read, '-t', dest='tiles', new='-tl/--tiles', nargs='+')
    _add(read, '-st', '--skip_tiles', nargs='+',
         help='Tiles to leave blank (not read; the output keeps its full size): "channel/stack" ids such as '
              'cFos-647/1-x00-y00, or a stack such as 1-x00-y00 for every channel.')
    _add(read, '-tp', '--timepoints', nargs='+', type=int,
         help='Timepoints to fuse, as listed when the mosaic is loaded. More than one gives an output with a '
              't axis. Default: all.')
    _add(read, '-rtp', '--reference_timepoint', type=parse_timepoint_or_all,
         help='Timepoint whose tiles set the geometry (output size, tile positions, transitions): every '
              'timepoint is fused with it. It need not be among --timepoints. "all" places every timepoint '
              'at its own stage positions instead. Default: the first timepoint.')
    _add(read, '-s', '--subset', type=str,
         help='Crop the fused output and/or select channels: axis=selection pairs separated by semicolons, '
              'e.g. "z=0:100;y=200:900;c=1" or "c=DAPI,GFP" (channels by index or name). z/y/x are in output '
              'voxels (after --downsample); slices may not have a step. Timepoints are chosen with --timepoints.')

    fusion = _section(parser, 'Fusion')
    _add(fusion, '-rds', '--read_downsample' if legacy else '--downsample', dest='read_downsample', nargs='+',
         type=int, metavar='FACTOR',
         help='Downsampling applied while reading the tiles (block mean): one value or Z Y X, e.g. -rds 4 8 8. '
              'Default: 1 (full resolution).')
    _add(fusion, '-p', '--projection', choices=['none', 'max', 'sum'], default='none',
         help='"none" writes a 3D volume; "max"/"sum" write a 2D projection along z.')
    _add(fusion, '-b', '--blend', choices=['sigmoid', 'max', 'sum', 'midline', 'none'], default=None,
         help='How overlapping tiles are merged: sigmoid (default for images: smooth transition across every '
              'overlap, between tiles and between tile groups), max, sum (overlaps brighter), midline (each tile '
              'group, e.g. illumination side, up to the middle of the overlap, with a sharp switch there), or none '
              '(no blending: every voxel is copied unchanged from one tile, cut in the middle of each overlap; '
              'the default and the only choice for label data).')
    _add(fusion, '-ss', '--sigmoid_steepness', type=float, advanced=True,
         help='Steepness of the blend=sigmoid transition: higher is sharper. Default: 4 (the switch takes the '
              'central ~27%% of the overlap).')
    _add(fusion, '-ma', '--midline_axis', choices=['x', 'y', 'z'], advanced=True,
         help='Axis along which blend=midline/sigmoid separates the groups (or the cameras). Default: z for the '
              'opposing cameras of a MuVi SPIM, x otherwise.')
    _add(fusion, '-dt', '--dtype', dest='out_dtype', action=_DtypeOrDataType,
         help='Output dtype, e.g. uint16 or float32. Default: input dtype (float32 for blend=sum).')
    _deprecated(fusion, '-od', '--out_dtype', dest='out_dtype', new='-dt/--dtype')
    _add(fusion, '-dty', '--data_type', choices=['intensity', 'label'], default=None,
         help='Write the output as an intensity image or as a label image (tiles of label IDs that already '
              'agree across tiles; needs --blend none, the default for labels, and an integer dtype). Default: '
              'label when every tile is label data (a folder of label stores), else intensity. For a folder of '
              'OME-Zarr stores, it also chooses which stores are read.')
    _deprecated(fusion, '-lo', '--label_offsets', dest='label_offsets', default=None,
                new='`zarrsmith fuse-labels --label_offsets`')
    _deprecated(fusion, '-lm', '--label_mapping', dest='label_mapping', default=None,
                new='`zarrsmith fuse-labels --label_mapping`')

    place = _section(parser, 'Tile positions')
    _add(place, '-tf', '--transforms',
         help='Transforms JSON (written by an earlier fuse, possibly edited) giving the tile positions to use '
              'instead of the stage positions. Tiles not listed keep their stage positions.')
    _add(place, '-tfo', '--transforms_only', action='store_true',
         help='Only compute the transforms and write them (next to the output, <output>.transforms.json): '
              'nothing is fused. Every fuse writes this file before fusing.')
    _add(place, '-rc', '--register_cameras', action='store_true',
         help='MuVi SPIM: register the camera views on the images before fusing. The transform between the right '
              'and the left view (translation and rotation, see --register_model) is measured around the z centre '
              'of the stack, where the fusion hands over from one camera to the other, and applied to the right '
              'view (the detection paths of the two cameras never match exactly, so the stage positions alone '
              'leave a jump). The transforms file records the registration: reuse it with --transforms.')
    _add(place, '-rm', '--register_model', choices=['rigid', 'translation'], default='rigid', advanced=True,
         help='Transform fitted by --register_cameras: rigid (translation and rotation, default; the right view is '
              'resampled on the left grid) or translation (the right view is only moved).')
    _add(place, '-rch', '--register_channel', advanced=True,
         help='Reference channel of --register_cameras (e.g. a nuclear stain such as DAPI): the transform is '
              'measured on it and applied to every channel. "all" registers every channel on its own images (also '
              'corrects a chromatic difference between the cameras; a channel that cannot be registered takes the '
              'transform of the best registered one). Default: the first channel. Timepoints follow '
              '--reference_timepoint: measured at the reference timepoint and applied to all, or at every '
              'timepoint with -rtp all.')

    _pyramid_section(parser, infer=not legacy)
    _storage_section(parser, **_MOSAIC_STORAGE_HELP)
    _channel_metadata_section(parser, names_help='Names of the output channels. Example: -cn autofluo cfos')
    _performance_section(parser, read_workers=True, temp_dir=True)
    _run_section(parser, legacy_overwrite=legacy)


# ---- fuse-labels ------------------------------------------------------------------------

FUSE_LABELS_HELP = """\
    Stitch the label stores of a segmentation run tile by tile into one OME-Zarr label image.
    Every store in the folder sits at its own NGFF translation. Tiles numbered separately get an
    offset each (--label_offsets), then the objects cut by a tile edge are merged (--label_mapping,
    e.g. from a label solver). Labels are never blended: every voxel is copied from one tile,
    overlaps are cut in their middle, and the pyramid levels sample voxels instead of averaging them.

    Examples:
      zarrsmith fuse-labels -i tiles/ -z merged.ome.zarr -lo offsets.csv -lm mapping.json
      zarrsmith fuse-labels -i tiles/ -z merged.ome.zarr -lo offsets.csv -lm mapping.json \\
          --infer_downscale_factors image.ome.zarr --start_level 1
      zarrsmith fuse-labels -i tiles/ -z merged.ome.zarr        (ids already agree across tiles)

    --label_offsets is a CSV with the columns store (the store name, with or without .ome.zarr) and
    offset, added to every non-zero id of that store; every store needs one. --label_mapping is a JSON
    {"id": new_id} applied after the offsets; ids not listed are kept. Each tile is given only the
    mapping entries between its offset and the next larger one, so the ids of a tile must stay below
    the next tile's offset. Without either, only stores marked as labels are read, as they are.
"""


def _add_fuse_labels_arguments(parser):
    _io_section(parser, batch=False, required=True, input_help='Folder of the tile label stores (OME-Zarr).',
                output_help='Path to the merged label .ome.zarr.')
    labels = _section(parser, 'Labels')
    _add(labels, '-lo', '--label_offsets', metavar='CSV', type=valid_input_path,
         help='CSV store,offset: the offset added to every non-zero id of each store.')
    _add(labels, '-lm', '--label_mapping', metavar='JSON', type=valid_input_path,
         help='JSON {"id": new_id}, applied after the offsets (e.g. objects merged across tile edges).')
    _add(labels, '--tiles_csv', metavar='CSV', type=valid_input_path,
         help='Instead of --label_offsets: a CSV with the columns tile_index and segmented_tile_path (or store) of '
              'every tile; each store gets the offset tile_index * --label_offset_step.')
    _add(labels, '--label_offset_step', type=int, metavar='N',
         help='With --tiles_csv: the step between the offsets of consecutive tiles.')
    _add(labels, '-dt', '--dtype', dest='out_dtype', default='uint32', help='Output dtype (an integer type).')
    _pyramid_section(parser, infer=True)
    _storage_section(parser, **_MOSAIC_STORAGE_HELP,
                     shard_target_help='Target uncompressed shard size in MB, used when --shards auto. Default: 256 '
                                       '(each fusion worker holds one shard).')
    _performance_section(parser, read_workers=True)
    _run_section(parser, dry_run_help='Print what would be written and stop, writing nothing: the output, its '
                                      'chunks and shards, the pyramid levels and the fusion workers.')


# ---- plan-tiles -------------------------------------------------------------------------

PLAN_TILES_HELP = """\
    Cut one level of an OME-Zarr image into overlapping tiles for parallel processing, and write
    them to a CSV: one row per tile with its bounds (voxels of that level, half-open [min, max)),
    its forward neighbours and the planned tile size and overlap.

    Examples:
      zarrsmith plan-tiles -i image.ome.zarr -oc tiles.csv -ts 512 -ov 32 -ha 16
      zarrsmith plan-tiles -i image.ome.zarr -oc tiles.csv -ts "y=1024;x=1024" -ov 32 -lv 1
      zarrsmith plan-tiles -i image.ome.zarr -oc tiles.csv -mtv 300000000 -ov 16 -ha 20
      zarrsmith plan-tiles -i image.ome.zarr -oc tiles.csv -ts 100 -ov 5 -ha 10 -pu --roi "y=50:400;x=0:300"

    An axis given a tile size is cut at that size; an axis without one is left whole, unless a tile
    would read more than --max_tile_voxels voxels (halo included, times --resize ** ndim): those axes
    are then cut, choosing the tile counts with the least cut area that fit. A last tile smaller than
    the halo (at least 2 voxels) is merged into the one before it. Tiles are numbered from 1, x
    fastest, then y, then z.

    CSV columns: dataset_id (<id>__<tile_id>), source_dataset_id, tile_id, tile_index, input_uri,
    x_min ... z_max, resolution_level, tile_overlap_<ax>, tile_size_<ax> (the planned size; an axis
    left whole reports its ROI length and overlap 0) and the forward neighbours right_* (next along x),
    bottom_* (along y), behind_* (along z): their dataset_id and tile_index, blank at the end of an axis.
"""


def parse_axis_values(value):
    """``--tile_size`` / ``--overlap`` / ``--halo``: one number for every spatial axis, or
    ``"z=16;y=512;x=512"``."""
    if value is None:
        return None
    text = str(value).strip()
    if '=' not in text:
        try:
            return float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid value '{value}': a number, or e.g. \"y=512;x=512\".") from exc
    out = {}
    for part in re.split(r'[;,\s]+', text):
        if not part:
            continue
        match = re.fullmatch(r'([zyxZYX])=([0-9.eE+-]+)', part)
        if match is None:
            raise argparse.ArgumentTypeError(f"Invalid '{part}': use axis=value with z, y or x, e.g. \"y=512;x=512\".")
        out[match.group(1).lower()] = float(match.group(2))
    return out


def parse_roi_spec(value):
    """``--roi "z=0:40;y=100:900"`` (voxels, or physical units with --physical); a side may be empty."""
    if value is None:
        return None
    roi = {}
    for part in re.split(r'[;,\s]+', str(value).strip()):
        if not part:
            continue
        match = re.fullmatch(r'([zyxZYX])=([0-9.eE+-]*):([0-9.eE+-]*)', part)
        if match is None:
            raise argparse.ArgumentTypeError(f"Invalid ROI '{part}'. Use axis=start:stop, e.g. \"y=100:900;x=0:500\".")
        axis, start, stop = match.groups()
        roi[axis.lower()] = (float(start) if start else None, float(stop) if stop else None)
    return roi or None


def _add_plan_tiles_arguments(parser):
    io = _section(parser, 'Input and output')
    _add(io, '-i', '--input_path', required=True, type=valid_input_path, help='OME-Zarr image to tile.')
    _add(io, '-oc', '--output_csv', required=True, type=os.path.abspath, help='Path of the tiles CSV.')
    _add(io, '-id', '--source_id',
         help='source_dataset_id of the rows (dataset_id is <id>__<tile_id>). Default: the store name without '
              '.ome.zarr.')
    tiles = _section(parser, 'Tiles')
    _add(tiles, '-lv', '--level', type=int, default=0, help='Pyramid level to tile; the bounds are in its voxels.')
    _add(tiles, '-ts', '--tile_size', type=parse_axis_values,
         help='Tile size: one value for every spatial axis, or per axis, e.g. "y=1024;x=1024". An axis without '
              'one is left whole unless --max_tile_voxels cuts it.')
    _add(tiles, '-ov', '--overlap', type=parse_axis_values, default=0,
         help='Overlap between neighbouring tiles: one value, or per axis.')
    _add(tiles, '-ha', '--halo', type=parse_axis_values, default=0,
         help='Context the processing step reads around each tile: counts towards --max_tile_voxels, and is the '
              'smallest last tile before it is merged into the previous one.')
    _add(tiles, '-mtv', '--max_tile_voxels', type=int,
         help='Most voxels one tile may read, halo included (e.g. what fits in GPU or host memory). Axes without '
              'a tile size are cut to fit it.')
    _add(tiles, '-rs', '--resize', type=float, default=1.0,
         help='Linear factor by which the processing step enlarges each tile (e.g. 30 / diameter for Cellpose); '
              'the budget is checked on voxels * resize ** ndim.')
    _add(tiles, '-roi', '--roi', type=parse_roi_spec,
         help='Part of the level to tile, e.g. "z=0:40;y=100:900". Default: the whole level.')
    _add(tiles, '-pu', '--physical', action='store_true',
         help='--tile_size, --overlap, --halo and --roi are in physical units (the NGFF translation is the centre '
              'of the first voxel). Default: voxels of --level.')
    _run_section(parser, overwrite=False, dry_run_help='Print the plan (tile counts, sizes, largest read) and '
                                                       'write nothing.')


# ---- chromatic-shift --------------------------------------------------------------------

CHROMATIC_SHIFT_HELP = """\
    Measure the axial (Z) chromatic shift between the channels of an OME-Zarr image on the sample
    itself (its lower and upper surfaces) and write a corrected copy with the same chunks, shards,
    compression and metadata as the input. Only a Z translation per channel is corrected.

    Examples:
      zarrsmith chromatic-shift -i brain.ome.zarr -z brain_csc.ome.zarr -r shift.json
      zarrsmith chromatic-shift -i brain.ome.zarr -r shift.json                  (measure only)
      zarrsmith chromatic-shift -i brain.ome.zarr -z brain_csc.ome.zarr -sf 0 104.4
      zarrsmith chromatic-shift -i other.ome.zarr -z other_csc.ome.zarr -sf shift.json
      zarrsmith chromatic-shift -i brain.ome.zarr -z brain_csc.ome.zarr -r shift.json --fail_on_warning
      zarrsmith chromatic-shift --batch images.csv

    Shifts are corrections in the Z unit (usually micrometer): positive values move a channel
    towards higher Z. The reference channel gets 0.

    The --batch CSV file has the columns input | output (empty: measure only), plus any of
    report | shifts | reference_channel | timepoint | max_shift | coarse_level | refine_level | n_regions |
    crop_yx | min_z_normal | min_correlation | threshold | extent | order | include_groups | include_labels |
    overwrite | num_workers | fail_on_warning, with the same meaning as the options below, e.g.
      input                | output                   | report               | reference_channel | max_shift
      /path/to/a.ome.zarr  | /path/to/a_csc.ome.zarr  | /path/to/a_csc.json  | cFos-647          | 150
      /path/to/b.ome.zarr  | /path/to/b_csc.ome.zarr  |                      | 0                 |
    Leave a cell empty for the default; shifts are space separated.
"""


def parse_timepoint_or_all(value):
    """``--reference_timepoint`` value: a timepoint, or ``all``."""
    text = str(value).strip().lower()
    if text == "all":
        return "all"
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid timepoint '{value}'. Use a timepoint or 'all'.") from exc


def parse_refine_level(value):
    """``--refine_level`` value: a level index, or ``none`` to skip refinement."""
    text = str(value).strip().lower()
    if text in {"none", "null"}:
        return "none"
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid refine level '{value}'. Use a level index or 'none'.") from exc


def _add_chromatic_shift_arguments(parser):
    # Options whose default lives in the library are left out of the namespace when not
    # given (SUPPRESS), so the help shows the real default instead of "(default: None)".
    io = _io_section(parser, input_help='Path to the input OME-Zarr image.', batch=False,
                     output_help='Path to the corrected .ome.zarr. Leave out to only measure the shift.')
    io.title = 'Input and output (or --batch)'
    _add(io, '-r', '--report', type=os.path.abspath, help='Save the estimate report (JSON) to this file.')
    _add(io, '--batch', metavar='CSV', type=valid_input_path,
         help='Run every row of this CSV file instead (see the description above).')

    est = _section(parser, 'Estimate')
    _add(est, '-sf', '--shifts', default=argparse.SUPPRESS, nargs='+',
         help='Skip the estimate and apply these shifts: one value per channel, or a JSON report from an earlier run.')
    _add(est, '-ref', '--reference_channel', default='0', help='Reference channel, as an index or a channel name.')
    _deprecated(est, '-rc', dest='reference_channel', new='-ref/--reference_channel')
    _add(est, '-tp', '--timepoints', dest='timepoint', default=argparse.SUPPRESS, type=int, metavar='TIMEPOINT',
         help='Timepoint the shift is measured on, an index along the t axis. Every timepoint is corrected with '
              'the same shift. Default: 0.')
    _deprecated(est, '--timepoint', dest='timepoint', new='-tp/--timepoints', type=int)
    _add(est, '-mxs', '--max_shift', default=argparse.SUPPRESS, type=float,
         help='Largest shift searched, in the Z unit. Default: 10%% of the stack depth.')
    _add(est, '-cl', '--coarse_level', default=argparse.SUPPRESS, type=int, advanced=True,
         help='Pyramid level of the coarse estimate. Default: the coarsest level with at least 16 Z planes and '
              '32 pixels in Y and X.')
    _add(est, '-rl', '--refine_level', default=argparse.SUPPRESS, type=parse_refine_level, advanced=True,
         help='Pyramid level of the refinement crops, or "none" to skip refinement. Default: 0.')
    _add(est, '-nr', '--n_regions', default=argparse.SUPPRESS, type=int, advanced=True,
         help='Number of refinement crops. Default: 32.')
    _add(est, '-cy', '--crop_yx', default=argparse.SUPPRESS, type=int, advanced=True,
         help='Crop width in Y and X, in pixels. Default: 64.')
    _add(est, '-mzn', '--min_z_normal', default=argparse.SUPPRESS, type=float, advanced=True,
         help='How flat a surface must be to count as Z-facing (1 = flat). Default: 0.7.')
    _add(est, '-mc', '--min_correlation', default=argparse.SUPPRESS, type=float, advanced=True,
         help='Crops with a lower correlation peak are left out. Default: 0.2.')
    _add(est, '-th', '--threshold', default=argparse.SUPPRESS, type=float, advanced=True,
         help='Sample mask threshold on the 0-1 rescaled coarse image. Default: Otsu.')
    _add(est, '-fw', '--fail_on_warning', action='store_true',
         help='Exit with an error, without writing the corrected image, when the estimate has warnings (e.g. the '
              'search limit was reached).')

    out = _section(parser, 'Corrected image')
    _add(out, '-e', '--extent', choices=['union', 'same', 'intersection'], default='union',
         help='union grows Z so no channel loses data, same keeps the input size, intersection crops to the planes '
              'every channel covers.')
    _add(out, '-o', '--order', type=int, choices=[0, 1], default=1,
         help='1 interpolates linearly; 0 rounds the shifts to whole planes (voxel values untouched).')
    _add(out, '-ig', '--include_groups', type=parse_bool, default=True,
         help='Also shift image subgroups with the same channels ("true" or "false").')
    _add(out, '-il', '--include_labels', type=parse_bool, default=True,
         help='Also write labels, padded or cropped to the new Z extent ("true" or "false").')
    _performance_section(parser, num_workers_default=argparse.SUPPRESS,
                         num_workers_help='Number of threads used to write the zarr. Default: half of the available '
                                          'cores, capped at 8.')
    _run_section(parser, dry_run=False)


# ---- resample ---------------------------------------------------------------------------

RESAMPLE_HELP = """\
    Change the pixel size of an OME-Zarr image, rescaling the data accordingly, and write a copy with
    the same compression and number of levels as the input, and chunks and shards fitted to each level.
    Image groups and labels are resampled too. Run `zarrsmith convert` first for other formats.

    Examples:
      zarrsmith resample -i brain.ome.zarr -z brain_1um.ome.zarr -ps 1
      zarrsmith resample -i brain.ome.zarr -z brain_iso.ome.zarr -ps 1 0.5 0.5
      zarrsmith resample --batch images.csv

    The pixel size is in the units of the dataset, usually micrometer. Images use linear interpolation,
    with anti-aliasing on the axes that shrink, and labels use nearest neighbour.

    The --batch CSV file has the columns input | output | pixel_size, plus any of fast | include_groups |
    include_labels | overwrite | num_workers, with the same meaning as the options below, e.g.
      input                | output                  | pixel_size  | fast
      /path/to/a.ome.zarr  | /path/to/a_1um.ome.zarr | 1           |
      /path/to/b.ome.zarr  | /path/to/b_iso.ome.zarr | 1 0.5 0.5   | true
    Leave a cell empty for the default; the pixel size is space separated.
"""


def _add_resample_arguments(parser):
    _io_section(parser, input_help='Path to the input OME-Zarr image.', output_help='Path to the resampled .ome.zarr.')
    out = _section(parser, 'Resampling')
    _add(out, '-ps', '--pixel_size', nargs='+', type=float, metavar='SIZE',
         help='New pixel size: one value for every spatial axis, or three values in Z Y X order.')
    _add(out, '-f', '--fast', action='store_true',
         help='Read images from the coarsest pyramid level still fine enough instead of level 0: much less data '
              'when shrinking, but it keeps the aliasing of that level. Labels always do this.')
    _add(out, '-ig', '--include_groups', type=parse_bool, default=True,
         help='Also resample image subgroups ("true" or "false").')
    _add(out, '-il', '--include_labels', type=parse_bool, default=True, help='Also resample labels ("true" or "false").')
    _performance_section(parser, num_workers_default=argparse.SUPPRESS,
                         num_workers_help='Number of threads used to write the zarr. Default: half of the available '
                                          'cores, capped at 8, and fewer if memory requires it.')
    _run_section(parser, dry_run=False)


# ---- allocate ---------------------------------------------------------------------------

ALLOCATE_HELP = """\
    Create an empty OME-Zarr image, to be filled tile by tile (e.g. by parallel jobs) and then
    given its pyramid with `zarrsmith pyramid`. Its geometry comes from a template OME-Zarr image,
    or from --size/--axes/--scale.

    Examples:
      zarrsmith allocate -t raw.ome.zarr -z probabilities.ome.zarr --dtype float32 --num_channels 3
      zarrsmith allocate -t raw.ome.zarr -z seg.ome.zarr --data_type label --dtype uint32 \\
          --crop "t=0:10;y=512:1536" --attrs '{"t_range": [0, 10]}'
      zarrsmith allocate -z new.ome.zarr --axes tczyx --size 5 2 100 2048 2048 --scale 2 0.5 0.5

    From a template, the output keeps its axes, level-0 size, dtype, pixel size, units, origin
    (translation), channels, chunks, shards, pyramid levels and zarr format, unless an option
    below changes them. --start_level N starts from level N of the template instead of level 0
    (e.g. to segment at a coarser resolution); --crop is then in that level's pixels.
    A label output (--data_type label) drops the template's channel axis unless channels are given.
    A crop moves the origin, so the output gets a translation that keeps it where it was.
    Zarr v2 outputs (NGFF 0.4) have no shards: explicit --shards are refused, "auto" is dropped.

    Write tiles that start and end on the shard grid (the chunk grid without shards), e.g. with
    ZarrManager(..., mode="r+").write_image_region(tile, ..., aligned=True), so parallel writers
    never share a file.
"""


def parse_crop_spec(value):
    """Parse --crop "t=0:10;y=512:1536" (or several "axis=start:stop" values) into a dict."""
    if value is None:
        return None
    tokens = value if isinstance(value, (list, tuple)) else [value]
    crop = {}
    for token in tokens:
        for part in re.split(r'[;,\s]+', str(token).strip()):
            if not part:
                continue
            match = re.fullmatch(r'([tczyxTCZYX])=(-?\d*):(-?\d*)', part)
            if match is None:
                raise argparse.ArgumentTypeError(
                    f"Invalid crop '{part}'. Use axis=start:stop, e.g. \"t=0:10;y=512:1536\"."
                )
            axis, start, stop = match.groups()
            crop[axis.lower()] = (int(start) if start else None, int(stop) if stop else None)
    return crop or None


def parse_attrs_spec(value):
    """Parse --attrs: a JSON object, or the path of a JSON file holding one."""
    import json

    if value is None:
        return None
    text = str(value)
    if os.path.isfile(text):
        with open(text, encoding="utf-8") as f:
            text = f.read()
    try:
        attrs = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--attrs is not valid JSON: {exc}") from None
    if not isinstance(attrs, dict):
        raise argparse.ArgumentTypeError("--attrs must be a JSON object, e.g. '{\"t_range\": [0, 10]}'.")
    return attrs


def _add_allocate_arguments(parser):
    src = _section(parser, 'Template (or --size/--axes)')
    _add(src, '-t', '--template', type=valid_input_path,
         help='OME-Zarr image whose root image gives the geometry and metadata.')
    _add(src, '-sl', '--start_level', default='0',
         help="Level of the template (index or path) that becomes the output's level 0; --crop is in its pixels. "
              'Default: 0 (full resolution).')
    _add(src, '-cr', '--crop', nargs='+', type=str,
         help='Part of the template to keep, as axis=start:stop, e.g. "t=0:10;y=512:1536".')
    _add(src, '-sz', '--size', nargs='+', type=int, metavar='N',
         help='Without a template: level-0 size, one value per axis of --axes.')
    _add(src, '-ax', '--axes', type=str.lower, help='Without a template: axes, e.g. tczyx or zyx.')
    _add(src, '-sc', '--scale', nargs='+', type=float, metavar='SIZE',
         help="Pixel size: one value, or one per spatial axis (e.g. Z Y X). Default: the template's, else 1.")
    _add(src, '-u', '--units', nargs='+', metavar='UNIT', help="Spatial unit(s). Default: the template's, else micrometer.")
    _add(src, '-ti', '--time_increment', type=float, help="Time step. Default: the template's, else 1.")

    out = _section(parser, 'Output')
    _add(out, '-z', '--zarr_path', dest='out', type=os.path.abspath, help='Path of the new .ome.zarr.')
    _deprecated(out, '-o', '--out', dest='out', new='-z/--zarr_path', type=os.path.abspath)
    _add(out, '-dt', '--dtype', help="Data type, e.g. uint16, uint32, float32. Default: the template's.")
    _add(out, '-dty', '--data_type', choices=['intensity', 'label'],
         help="intensity (images, probabilities) or label. Default: the template's.")
    _add(out, '-nc', '--num_channels', type=int, help='Number of channels; adds a c axis if needed.')

    _pyramid_section(parser, method=False,
                     num_levels_help="Levels to create; `zarrsmith pyramid` fills them once level 0 is written. "
                                     "Default: the template's levels from --start_level on, else 1.",
                     factor_help="Factor between levels: one value, or one per spatial axis. Default: the "
                                 "template's, else 2.")

    storage = _section(parser, 'Storage layout', advanced=True)
    _add(storage, '-zf', '--zarr_format', type=int, choices=[2, 3], advanced=False,
         help="2 (NGFF 0.4) or 3 (NGFF 0.5). Default: the template's, else 3.")
    _add(storage, '-cs', '--chunk_size', nargs='+', type=int, advanced=False,
         help="Level-0 chunks, one per output axis. Default: the template's.")
    _add(storage, '-sh', '--shards', nargs='+',
         help='"auto", a shard shape (one per output axis), or "none". Default: the template\'s (zarr v3), "auto" '
              'without a template. Not available on zarr v2.')
    _add(storage, '-stm', '--shard_target_mb', type=float, default=DEFAULT_SHARD_TARGET_MB,
         help='Target shard size in MB for --shards auto.')
    _add(storage, '-cp', '--compressor', type=str.lower, choices=['blosc', 'gzip', 'none'], default='blosc',
         help='Codec; "gzip" needs --zarr_format 2.')
    _add(storage, '-cpl', '--compressor_level', type=int, default=1, help='Compression level, 0 to 9.')

    meta = _channel_metadata_section(parser, names_help='Channel names (sets the number of channels).')
    meta.title = 'Metadata'
    _add(meta, '-tr', '--translation', nargs='*', type=float, metavar='ORIGIN',
         help='Without a value: keep the origin of the template, moved by --crop. With one value per spatial axis: '
              'the level-0 origin. An all-zero origin is not written (NGFF reads it the same way). Default: write '
              'one when the template has one, the crop moves it, or the coarser levels are averaged (intensity data '
              'with --num_levels > 1: they sit half a block further than level 0, from an origin of 0 without a '
              'template one).')
    _add(meta, '-at', '--attrs', type=parse_attrs_spec,
         help='Custom group attributes as a JSON object (or a .json file), e.g. \'{"t_range": [0, 10]}\'. Stored '
              'next to the NGFF metadata.')
    _add(meta, '-n', '--name', help="Name of the image in the metadata. Default: the template's.")
    _run_section(parser, dry_run=False)


# ---- pyramid ----------------------------------------------------------------------------

PYRAMID_HELP = """\
    Build the pyramid of an OME-Zarr image in place: the levels after --start_level (default 0)
    are created (or replaced) from it, each level from the one above it, one task per output
    shard (chunk on zarr v2). The start level and the levels above it are kept as they are.
    The format, codecs and chunks come from level 0; the multiscales metadata is updated.
    --num_levels counts every level from level 0.

    Examples:
      zarrsmith pyramid -z probabilities.ome.zarr
      zarrsmith pyramid -z seg.ome.zarr --num_levels 4 --downscale_factor 1 2 2
      zarrsmith pyramid -z image.ome.zarr --start_level 2 --num_levels 6   # keep 0-2, rebuild 3-5
      zarrsmith pyramid -z image.ome.zarr --group labels/nuclei --method mode
      zarrsmith pyramid -z image.ome.zarr --group all

    Methods: "mean" averages each block (default for intensity data), "nearest" keeps one pixel
    of it (default for labels) and "mode" keeps its most frequent value (smoother label pyramids).
"""


def _add_pyramid_arguments(parser):
    image = _section(parser, 'Image')
    _add(image, '-z', '--zarr_path', type=valid_input_path, required=True, help='OME-Zarr store to update in place.')
    _add(image, '-g', '--group', nargs='+', default=None,
         help='Image(s) to process: a group path such as labels/nuclei, or "all" for the root image, every image '
              'group and every label. Default: the root image.')
    pyramid = _section(parser, 'Pyramid')
    _add(pyramid, '-nl', '--num_levels', type=int, default=None,
         help='Total number of levels, level 0 included. Default: the number the image already has (e.g. as '
              'allocated from a template); with a single level, downscale until no spatial axis is above 2048 pixels.')
    _add(pyramid, '-df', '--downscale_factor', nargs='+', type=int, default=None,
         help='Factor between levels: one value, or one per spatial axis (e.g. 1 2 2 keeps Z). Default: the factor '
              'between the existing levels, else 2.')
    _add(pyramid, '-me', '--method', choices=['mean', 'nearest', 'mode'], default=None,
         help='Downsampling method. Default: mean for intensity data, nearest for labels.')
    _add(pyramid, '-sl', '--start_level', default='0',
         help='Level to build from: its index, or its path in the store (e.g. 1). It and the levels above it are '
              'kept; the levels after it are rebuilt. Default: 0.')
    perf = _section(parser, 'Performance')
    _add(perf, '-nw', '--num_workers', type=int, default=None,
         help='Number of threads. Default: half of the available cores, capped at 8.')


# ---- segment and measure: project, threshold, mask, measure, filter-labels ---------------

PROJECT_HELP = """\
    Project an OME-Zarr image along a spatial axis (default z): a maximum (or sum, or mean)
    intensity projection, on every pyramid level. The axis is removed from the output, which
    keeps the channels, the other axes and the pyramid of the input.

    Examples:
      zarrsmith project -i stack.ome.zarr -z mip.ome.zarr
      zarrsmith project -i stack.ome.zarr -z sum.ome.zarr -p sum
"""

THRESHOLD_HELP = """\
    Label the objects of one channel of an OME-Zarr image: a global threshold (Otsu, times
    --factor) at one pyramid level, then connected components, per timepoint. The other levels
    are that labelling resized with nearest neighbour, so an object keeps its id on every level.
    The output is a label image without a channel axis.

    Examples:
      zarrsmith threshold -i image.ome.zarr -z nuclei.ome.zarr -c DAPI
      zarrsmith threshold -i image.ome.zarr -z nuclei.ome.zarr -c 0 -lv 2 -fa 1.2
"""

MASK_HELP = """\
    Zero one channel of an OME-Zarr image outside the objects of a label image (e.g. from
    `zarrsmith threshold` or `filter-labels`), on every pyramid level, and add it to the image
    as a new channel named <channel><suffix> (default _clean), or write it alone.

    Examples:
      zarrsmith mask -i image.ome.zarr -l nuclei.ome.zarr -c DAPI -z image_clean.ome.zarr
      zarrsmith mask -i image.ome.zarr -l nuclei.ome.zarr -c 1 -z dapi_only.ome.zarr --masked_only
"""

MEASURE_HELP = """\
    Measure every object of a label image against the channels of an intensity image
    (skimage.measure.regionprops) and write one CSV row per label (and timepoint): area,
    bounding box, centroid, solidity, extent (2D also perimeter, orientation, eccentricity,
    axis lengths and form factor), plus per channel the mean, min and max intensity, as
    <property>_ch<i>_<name> columns.

    Examples:
      zarrsmith measure -i image.ome.zarr -l nuclei.ome.zarr -oc features.csv
      zarrsmith measure -i image.ome.zarr -l nuclei.ome.zarr -oc features.csv -ch DAPI GFP \\
          --properties euler_number --intensity_properties weighted_centroid
"""

FILTER_LABELS_HELP = """\
    Keep the objects of a label image whose measurements (a CSV with a label column, e.g. from
    `zarrsmith measure`) are within limits, zero every other one on every pyramid level, and
    renumber the kept objects 1..N (per timepoint). Also writes the kept rows, with the new label
    and the old one as original_label.

    Examples:
      zarrsmith filter-labels -l nuclei.ome.zarr -tb features.csv -z kept.ome.zarr -oc kept.csv --min_area 50
      zarrsmith filter-labels -l nuclei.ome.zarr -tb features.csv -z kept.ome.zarr -oc kept.csv \\
          --filters "solidity:0.8:;mean_intensity_ch1_GFP:100:"
"""


def _derived_io(parser, *, input_help, labels=False, labels_help='Label OME-Zarr image.', output=True,
                output_help='Path to the output .ome.zarr.'):
    io = _section(parser, 'Input and output')
    _add(io, '-i', '--input_path', required=True, type=valid_input_path, help=input_help)
    if labels:
        _add(io, '-l', '--labels', required=True, type=valid_input_path, help=labels_help)
    if output:
        _add(io, '-z', '--zarr_path', required=True, type=os.path.abspath, help=output_help)
    return io


def _derived_storage(parser):
    storage = _section(parser, 'Storage layout')
    _add(storage, '-zf', '--zarr_format', choices=[2, 3], type=int,
         help='2 (NGFF 0.4) or 3 (NGFF 0.5). Default: as the input.')


def _add_project_arguments(parser):
    _derived_io(parser, input_help='OME-Zarr image to project.')
    proj = _section(parser, 'Projection')
    _add(proj, '-p', '--projection', choices=['max', 'sum', 'mean'], default='max', help='Projection.')
    _add(proj, '--axis', choices=['z', 'y', 'x'], default='z', help='Spatial axis to project along.')
    _derived_storage(parser)
    _run_section(parser, dry_run=False)


def _add_threshold_arguments(parser):
    _derived_io(parser, input_help='OME-Zarr image with a channel axis.')
    read = _section(parser, 'What to read')
    _add(read, '-c', '--channel', required=True, help='Channel to threshold, by index or name.')
    _add(read, '-lv', '--level', type=int, default=0,
         help='Pyramid level the threshold and the labelling are made at; the other levels are derived from it.')
    th = _section(parser, 'Threshold')
    _add(th, '--threshold_method', choices=['otsu'], default='otsu', help='Global threshold.')
    _add(th, '-fa', '--factor', type=float, default=1.0, help='Multiplier applied to the threshold.')
    _add(th, '-dt', '--dtype', default='uint32', help='Label dtype (an integer type).')
    pyr = _section(parser, 'Pyramid')
    _add(pyr, '--levels', choices=['all', 'from_level'], default='all',
         help='all: every level of the image (the finer ones upsampled), so the labels line up level for level '
              'with it; from_level: --level and the coarser levels only.')
    _derived_storage(parser)
    _run_section(parser, dry_run=False)


def _add_mask_arguments(parser):
    _derived_io(parser, input_help='OME-Zarr intensity image.', labels=True,
                labels_help='Label OME-Zarr image with the same pyramid as the image (no channel axis).')
    read = _section(parser, 'What to read')
    _add(read, '-c', '--channel', required=True, help='Channel to mask, by index or name.')
    mask = _section(parser, 'Mask')
    _add(mask, '--suffix', default='_clean', help='Appended to the channel name to name the masked channel.')
    _add(mask, '--masked_only', action='store_true',
         help='Write only the masked channel, instead of every channel plus the masked one.')
    _derived_storage(parser)
    _run_section(parser, dry_run=False)


def _add_measure_arguments(parser):
    io = _derived_io(parser, input_help='OME-Zarr intensity image.', labels=True, output=False)
    _add(io, '-oc', '--output_csv', required=True, type=os.path.abspath, help='Path of the features CSV.')
    read = _section(parser, 'What to read')
    _add(read, '-ch', '--channels', nargs='+', help='Channels to measure, by index or name. Default: all.')
    _add(read, '-lv', '--level', type=int, default=0, help='Pyramid level of the intensity image.')
    _add(read, '-llv', '--labels_level', type=int, default=0, help='Pyramid level of the labels.')
    props = _section(parser, 'Measurements')
    _add(props, '--properties', nargs='+', default=[], metavar='PROPERTY',
         help='Extra regionprops shape properties, measured once per label (e.g. euler_number).')
    _add(props, '--intensity_properties', nargs='+', default=[], metavar='PROPERTY',
         help='Extra regionprops intensity properties, measured per channel (e.g. weighted_centroid).')
    _add(props, '--dataset_id', help='Value of a first dataset_id column. Default: no such column.')
    _run_section(parser, dry_run=False, overwrite=False)


def _add_filter_labels_arguments(parser):
    io = _section(parser, 'Input and output')
    _add(io, '-l', '--labels', required=True, type=valid_input_path, help='Label OME-Zarr image to filter.')
    _add(io, '-tb', '--table', required=True, type=valid_input_path,
         help='CSV of measurements with a label column (and timepoint for labels with a t axis).')
    _add(io, '-z', '--zarr_path', required=True, type=os.path.abspath, help='Path of the filtered label .ome.zarr.')
    _add(io, '-oc', '--output_csv', required=True, type=os.path.abspath, help='Path of the CSV of the kept rows.')
    lim = _section(parser, 'Limits')
    _add(lim, '--min_area', type=float, help='Smallest area kept (the area column, in voxels).')
    _add(lim, '--max_area', type=float, help='Largest area kept.')
    _add(lim, '--filters',
         help='Limits on any column: "property:min:max" separated by semicolons, either bound may be empty, '
              'e.g. "solidity:0.8:;mean_intensity_ch1:100:".')
    _add(lim, '--keep_ids', action='store_true', help='Keep the original ids instead of renumbering 1..N.')
    _derived_storage(parser)
    _run_section(parser, dry_run=False)


# ---- command line -----------------------------------------------------------------------

#: Old command names, kept working: old name -> (mode, new spelling)
RENAMED = {
    '2zarr': ('convert', 'zarrsmith convert'),
    'batch2zarr': ('convert_batch', 'zarrsmith convert --batch'),
    'mosaic2zarr': ('fuse', 'zarrsmith fuse'),
    'batch-mosaic2zarr': ('fuse_batch', 'zarrsmith fuse --batch'),
    'batchmosaic2zarr': ('fuse_batch', 'zarrsmith fuse --batch'),
}

#: The commands, by stage: (stage title, [(command, mode, one-line help, help text, builder)])
STAGES = [
    ('Make an OME-Zarr', [
        ('convert', 'convert', 'one dataset or microscope file -> OME-Zarr (or a CSV batch with --batch)',
         CONVERT_HELP, _add_convert_arguments),
        ('fuse', 'fuse', 'the raw tiles of a microscope mosaic -> one OME-Zarr (or a CSV batch)',
         FUSE_HELP, _add_fuse_arguments),
    ]),
    ('Correct an OME-Zarr', [
        ('resample', 'resample', 'change the pixel size (or a CSV batch)', RESAMPLE_HELP, _add_resample_arguments),
        ('chromatic-shift', 'chromatic_shift', 'measure and correct the z shift between channels (or a CSV batch)',
         CHROMATIC_SHIFT_HELP, _add_chromatic_shift_arguments),
    ]),
    ('Process an OME-Zarr in tiles', [
        ('plan-tiles', 'plan_tiles', 'cut an image into tiles: a CSV of tile bounds and neighbours',
         PLAN_TILES_HELP, _add_plan_tiles_arguments),
        ('allocate', 'allocate', 'create an empty OME-Zarr (e.g. the output of tiled jobs)',
         ALLOCATE_HELP, _add_allocate_arguments),
        ('fuse-labels', 'fuse_labels', 'stitch per-tile label stores into one label image',
         FUSE_LABELS_HELP, _add_fuse_labels_arguments),
        ('pyramid', 'pyramid', '(re)build the pyramid levels of an OME-Zarr in place',
         PYRAMID_HELP, _add_pyramid_arguments),
    ]),
    ('Segment and measure', [
        ('project', 'project', 'max (or sum, mean) projection along an axis, on every level',
         PROJECT_HELP, _add_project_arguments),
        ('threshold', 'threshold', 'label the objects of one channel (Otsu threshold, connected components)',
         THRESHOLD_HELP, _add_threshold_arguments),
        ('mask', 'mask', 'zero one channel outside the objects of a label image, as a new channel',
         MASK_HELP, _add_mask_arguments),
        ('measure', 'measure', 'regionprops of every label against the channels of an image -> CSV',
         MEASURE_HELP, _add_measure_arguments),
        ('filter-labels', 'filter_labels', 'keep the labels whose measurements are within limits, renumbered',
         FILTER_LABELS_HELP, _add_filter_labels_arguments),
    ]),
]


def _overview() -> str:
    width = max(len(name) for _, commands in STAGES for name, *_ in commands)
    lines = ["Welcome fellow MIF users!", ""]
    for title, commands in STAGES:
        lines.append(f"{title}:")
        lines += [f"  {name:<{width}}  {short}" for name, _, short, *_ in commands]
        lines.append("")
    lines.append("Run `zarrsmith <command> -h` for the options of a command, `--help-all` for every option.")
    return "\n".join(lines)


def _build_parser(show_all: bool = False):
    """The ``zarrsmith`` parser and its subcommand parsers (``--help-all``: every option shown)."""
    parser = argparse.ArgumentParser(prog='zarrsmith', description=_overview(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    # No `help` for the commands: the overview above lists them, grouped by stage.
    subparsers = parser.add_subparsers(dest='command', metavar='<command>', required=True)

    for _, commands in STAGES:
        for name, mode, short, text, builder in commands:
            sub = subparsers.add_parser(name, description=textwrap.dedent(text),
                                        formatter_class=MultilineDefaultsHelpFormatter)
            sub.set_defaults(mode=mode, renamed_to=None)
            builder(sub)
            _finish_help(sub, name, show_all)

    # Old names (not listed in the help), with their original arguments
    for old, (mode, new) in RENAMED.items():
        if old == 'batchmosaic2zarr':
            continue  # alias of batch-mosaic2zarr
        aliases = ['batchmosaic2zarr'] if old == 'batch-mosaic2zarr' else []
        p = subparsers.add_parser(old, aliases=aliases, description=f'Old name of `{new}`.',
                                  formatter_class=MultilineDefaultsHelpFormatter)
        p.set_defaults(mode=mode, renamed_to=new)
        if mode == 'convert':
            _add_convert_arguments(p, legacy=True)
        elif mode == 'fuse':
            _add_fuse_arguments(p, legacy=True)
        else:
            group = p.add_argument_group('Required Named arguments.')
            group.add_argument('-i', '--input_file', required=True, type=valid_input_path,
                               help='Path to the batch .csv file.')
    return parser, subparsers


def _parse_arguments(argv=None):
    """Build and parse the ZarrSmith command line interface arguments."""
    argv = list(sys.argv[1:] if argv is None else argv)
    show_all = '--help-all' in argv
    if show_all:
        argv = ['-h' if a == '--help-all' else a for a in argv]
    parser, subparsers = _build_parser(show_all)
    args = parser.parse_args(argv)
    if hasattr(args, 'help_all'):
        del args.help_all

    # convert / fuse: either -i and -z, or --batch
    if args.mode in ('convert', 'fuse') and args.renamed_to is None:
        command = subparsers.choices[args.mode]
        if args.batch is not None:
            if args.input_path is not None or args.zarr_path is not None:
                command.error('--batch cannot be combined with -i/-z: the CSV file lists inputs and outputs.')
            if args.overwrite:
                command.error('--batch cannot be combined with --overwrite: use the overwrite column of the CSV file.')
            args.mode += '_batch'
            args.input_file = args.batch
        elif args.input_path is None or args.zarr_path is None:
            command.error('the following arguments are required: -i/--input_path and -z/--zarr_path (or --batch CSV)')

    # chromatic-shift: -i (with an optional -z), or --batch
    if args.mode == 'chromatic_shift':
        command = subparsers.choices['chromatic-shift']
        if args.batch is not None:
            if args.input_path is not None or args.zarr_path is not None or args.report is not None:
                command.error('--batch cannot be combined with -i/-z/-r: the CSV file lists them.')
            if args.overwrite:
                command.error('--batch cannot be combined with --overwrite: use the overwrite column of the CSV file.')
            args.mode = 'chromatic_shift_batch'
            args.input_file = args.batch
        elif args.input_path is None:
            command.error('the following arguments are required: -i/--input_path (or --batch CSV)')

    # resample: -i, -z and -ps, or --batch
    if args.mode == 'resample':
        command = subparsers.choices['resample']
        if args.batch is not None:
            if args.input_path is not None or args.zarr_path is not None or args.pixel_size is not None:
                command.error('--batch cannot be combined with -i/-z/-ps: the CSV file lists them.')
            if args.overwrite or args.fast:
                command.error('--batch cannot be combined with --overwrite or --fast: use the columns of the CSV file.')
            args.mode = 'resample_batch'
            args.input_file = args.batch
        elif args.input_path is None or args.zarr_path is None or args.pixel_size is None:
            command.error('the following arguments are required: -i/--input_path, -z/--zarr_path and '
                          '-ps/--pixel_size (or --batch CSV)')
        elif len(args.pixel_size) not in (1, 3):
            command.error('-ps/--pixel_size takes one value, or three values in Z Y X order.')

    # allocate: an output, and a template or --size and --axes
    if args.mode == 'allocate':
        command = subparsers.choices['allocate']
        if getattr(args, 'out', None) is None:
            command.error('the following arguments are required: -z/--zarr_path')
        if args.template is not None and (args.size is not None or args.axes is not None):
            command.error('give either -t/--template or --size/--axes, not both.')
        if args.template is None and (args.size is None or args.axes is None):
            command.error('the following arguments are required: -t/--template, or --size and --axes')
        if args.template is None and len(args.size) != len(args.axes):
            command.error(f'--size has {len(args.size)} values for the {len(args.axes)} axes {args.axes!r}.')
        try:
            args.crop = parse_crop_spec(args.crop)
        except argparse.ArgumentTypeError as error:
            command.error(str(error))

    # fuse-labels: offsets from a CSV or from the tiles CSV, and an integer dtype
    if args.mode == 'fuse_labels':
        import numpy as np

        command = subparsers.choices['fuse-labels']
        if args.tiles_csv is not None and args.label_offsets is not None:
            command.error('give either --label_offsets or --tiles_csv, not both.')
        if (args.tiles_csv is None) != (args.label_offset_step is None):
            command.error('--tiles_csv and --label_offset_step go together.')
        try:
            if np.dtype(args.out_dtype).kind not in 'iu':
                command.error(f'--dtype must be an integer type for labels, got {args.out_dtype}.')
        except TypeError:
            command.error(f'--dtype {args.out_dtype!r} is not a data type.')
    return args
