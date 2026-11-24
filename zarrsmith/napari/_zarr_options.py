"""OME-Zarr output options shared by the napari widgets: chunks, shards, channel names and colours.

No Qt here, so everything is unit-testable; the widgets only read their controls and show
the results.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from zarrsmith.mosaic_manager.fusion import FUSION_SHARD_TARGET_MB
from zarrsmith.utils.chunks import (  # noqa: F401  (re-exported for the widgets)
    DEFAULT_CHUNK_MB, auto_chunk_shape, block_count, chunk_shard_layout, layout_summary, manual_chunks,
)
from zarrsmith.utils.ngff import DEFAULT_SHARD_TARGET_MB

#: OME-Zarr versions offered: label -> zarr format (NGFF 0.5 is zarr v3, 0.4 is zarr v2).
NGFF_FORMATS = {"0.5 (zarr v3)": 3, "0.4 (zarr v2)": 2}
#: Downsampling of the pyramid levels; "auto": mean for intensity images, nearest for labels.
PYRAMID_METHODS = ("auto", "mean", "nearest", "mode")
#: Default (uncompressed) size of a shard, in MB: the defaults of zarrsmith convert and of fusion
#: (zarrsmith fuse, fuse_tiles), so the widgets write what the CLI writes.
DEFAULT_SHARD_MB = DEFAULT_SHARD_TARGET_MB
FUSION_SHARD_MB = FUSION_SHARD_TARGET_MB


# ---------------------------------------------------------------------------
# chunks and shards
# ---------------------------------------------------------------------------

def auto_chunks(axes: str, shape: Sequence[int], dtype, target_mb: float = DEFAULT_CHUNK_MB,
                voxel_size: Optional[Sequence[float]] = None) -> tuple:
    """Output chunk shape of about ``target_mb`` (uncompressed), in ``axes`` order: the same
    chunks as the CLI and Python API (:func:`zarrsmith.utils.chunks.auto_chunk_shape`).

    Chunks are 1 along ``t`` and ``c``, about cubic in micrometers (``voxel_size``, of the
    spatial axes; ``None`` where unknown) and at most 32 planes deep: 4 MB gives
    ``32 x 256 x 256`` for uint16, ``32 x 128 x 256`` for float32 and ``1024 x 2048`` for a
    uint16 2D image.
    """
    return auto_chunk_shape(axes, shape, dtype, target_mb, voxel_size=voxel_size)


# ---------------------------------------------------------------------------
# channel names and colours
# ---------------------------------------------------------------------------

def auto_channel_names(channels: Sequence[str]) -> str:
    """The channel names field filled in from the detected ``channels``: spaces (which separate
    the names in the field and in batch CSV cells) become underscores, ``Alexa 488`` -> ``Alexa_488``."""
    return " ".join("_".join(str(c).split()) for c in channels)


def follow_auto_value(field, new: str, state: dict, key: str) -> None:
    """Set the text ``field`` (a magicgui widget) to ``new`` unless the user typed a value of
    their own: the field follows while it is empty or still holds the value set last time
    (kept in ``state[key]``)."""
    if not str(field.value).strip() or field.value == state.get(key):
        field.value = new
    state[key] = new


def parse_channel_names(text: str) -> list[str]:
    """Output channel names typed in the widget, separated by spaces (empty: the defaults)."""
    return str(text or "").split()


def parse_channel_colors(text: str) -> list[str]:
    """Output channel colours typed in the widget (6-digit hex codes or matplotlib names, separated
    by spaces; empty: the defaults), as ``RRGGBB``. Raises ``ValueError`` on an unknown colour."""
    from zarrsmith.utils.colors import parse_color

    colors = []
    for token in str(text or "").split():
        try:
            colors.append(parse_color(token))
        except TypeError as e:
            raise ValueError(str(e)) from None
    return colors


def channel_option_problems(names_text: str, colors_text: str, channels: Sequence[str]) -> dict:
    """What is wrong with the channel names and colours typed in a widget for the output ``channels``.

    Returns ``{"channel_names": [...], "channel_colors": [...]}``, one message per problem (empty
    lists: both can be used). Names: one per output channel, all different. Colours: one per
    output channel, each a 6-digit hex code or a matplotlib colour name. An empty field (the
    defaults) is always fine.
    """
    from zarrsmith.utils.colors import parse_color

    n = len(channels)
    output = f"{n} output channel{'s' if n != 1 else ''}" + (f" ({', '.join(map(str, channels))})" if n else "")

    def count(values, what):
        if values and len(values) != n:
            return [f"{len(values)} {what if len(values) != 1 else what[:-1]} for {output}: give one per "
                    "channel, or leave the field empty for the defaults."]
        return []

    names = parse_channel_names(names_text)
    name_problems = count(names, "names")
    repeated = sorted({nm for nm in names if names.count(nm) > 1})
    if repeated:
        name_problems.append(f"repeated name{'s' if len(repeated) > 1 else ''} {', '.join(repeated)}: "
                             "every channel needs its own name.")

    tokens = str(colors_text or "").split()
    unknown = []
    for token in tokens:
        try:
            parse_color(token)
        except TypeError:
            unknown.append(token)
    color_problems = count(tokens, "colors")
    if unknown:
        color_problems.append(f"{', '.join(repr(t) for t in unknown)} not recognized: use 6-digit hex codes "
                              "(FF00FF) or matplotlib color names (magenta).")
    return {"channel_names": name_problems, "channel_colors": color_problems}


def channel_warning_text(problems: dict) -> str:
    """The warning shown under the output estimate for :func:`channel_option_problems` (empty: none)."""
    labels = {"channel_names": "Channel names", "channel_colors": "Channel colors"}
    return "\n".join(f"⚠ {labels[key]}: {message}" for key, messages in problems.items() for message in messages)


def channel_error(problems: dict) -> Optional[str]:
    """:func:`channel_option_problems` as one line for the log, or ``None`` when there are none."""
    text = channel_warning_text(problems)
    return " ".join(line.replace("⚠ ", "") for line in text.splitlines()) or None
