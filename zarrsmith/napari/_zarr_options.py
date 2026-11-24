"""OME-Zarr output options shared by the napari widgets: chunks, shards, channel names and colours.

No Qt here, so everything is unit-testable; the widgets only read their controls and show
the results.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

#: OME-Zarr versions offered: label -> zarr format (NGFF 0.5 is zarr v3, 0.4 is zarr v2).
NGFF_FORMATS = {"0.5 (zarr v3)": 3, "0.4 (zarr v2)": 2}
#: Downsampling of the pyramid levels; "auto": mean for intensity images, nearest for labels.
PYRAMID_METHODS = ("auto", "mean", "nearest", "mode")
#: Default (uncompressed) sizes of an output chunk and of a shard, in MB.
DEFAULT_CHUNK_MB = 4
DEFAULT_SHARD_MB = 512


# ---------------------------------------------------------------------------
# chunks and shards
# ---------------------------------------------------------------------------

def auto_chunks(axes: str, shape: Sequence[int], dtype, target_mb: float = DEFAULT_CHUNK_MB) -> tuple:
    """Output chunk shape of about ``target_mb`` (uncompressed), in ``axes`` order.

    Chunks are 1 along ``t`` and ``c``. The spatial sizes are doubled one at a time
    (powers of two, clipped to the image) while the chunk stays within ``target_mb``,
    y and x first and z 8 times less: 4 MB gives ``32 x 256 x 256`` for uint16,
    ``16 x 256 x 256`` for float32 and ``1024 x 2048`` for a uint16 2D image.
    """
    target = max(1, int(target_mb * 1024 * 1024) // np.dtype(dtype).itemsize)
    sizes = {ax: max(1, int(s)) for ax, s in zip(axes, shape)}
    spatial = [ax for ax in "zyx" if ax in sizes]
    weight = {"z": 8, "y": 1, "x": 1}
    chunk = {ax: 1 for ax in spatial}
    while True:
        total = int(np.prod(list(chunk.values())))
        growable = sorted((ax for ax in spatial if chunk[ax] < sizes[ax]),
                          key=lambda ax: (chunk[ax] * weight[ax], "xyz".index(ax)))
        for ax in growable:
            new = min(2 * chunk[ax], sizes[ax])
            if total // chunk[ax] * new <= target:
                chunk[ax] = new
                break
        else:
            break
    return tuple(chunk.get(ax, 1) for ax in axes)


def manual_chunks(axes: str, shape: Sequence[int], zyx: Sequence[int]) -> tuple:
    """Chunk shape in ``axes`` order from the widget's ``(z, y, x)`` chunk: 1 along ``t``
    and ``c``, clipped to the image (z is ignored without a z axis)."""
    wanted = dict(zip("zyx", (int(v) for v in zyx)))
    return tuple(max(1, min(wanted.get(ax, 1), int(s))) for ax, s in zip(axes, shape))


def block_count(shape: Sequence[int], block: Sequence[int]) -> int:
    """Number of ``block``-shaped pieces that tile ``shape``."""
    return int(np.prod([-(-int(s) // int(b)) for s, b in zip(shape, block)]))


def chunk_shard_layout(axes: str, shape: Sequence[int], dtype, *, chunk_auto: bool = True,
                       chunk_mb: float = DEFAULT_CHUNK_MB, chunk_zyx: Sequence[int] = (32, 256, 256),
                       zarr_format: int = 3, shards: str = "auto", shard_mb: float = DEFAULT_SHARD_MB,
                       shard_exclude_axes: Sequence[str] = ("t", "c")) -> dict:
    """Level-0 chunks and shards of an output of ``shape`` (``axes`` order).

    Returns ``chunks`` (auto: :func:`auto_chunks` of ``chunk_mb``; else ``chunk_zyx``),
    ``shards`` (the shard shape that ``shards="auto"`` writes for level 0, ``None`` without
    shards: ``shards="none"`` or zarr v2), ``exclude`` (axes shards never merge), and
    ``block_bytes`` and ``n_blocks``, the size and number of the write blocks (shards, or
    chunks without shards).
    """
    from zarrsmith.utils.ngff import _auto_shard_for_level

    chunks = (auto_chunks(axes, shape, dtype, chunk_mb) if chunk_auto
              else manual_chunks(axes, shape, chunk_zyx))
    exclude = tuple(dict.fromkeys(shard_exclude_axes))
    shard = None
    if int(zarr_format) == 3 and shards == "auto":
        shard = _auto_shard_for_level(shape, chunks, np.dtype(dtype).itemsize, int(float(shard_mb) * 1024 * 1024),
                                      axes=axes, exclude_axes=exclude)
        shard = None if shard is None else tuple(min(int(b), int(s)) for b, s in zip(shard, shape))
    block = shard or chunks
    return {"chunks": chunks, "shards": shard, "exclude": exclude,
            "block_bytes": int(np.prod(block)) * np.dtype(dtype).itemsize, "n_blocks": block_count(shape, block)}


def layout_summary(axes: str, dtype, layout: dict) -> str:
    """``chunks 32x256x256 (4.0 MB), shards 128x1024x1024 (512 MB, 40 shards)``, followed by
    ``, 8 fusion workers (auto)`` when ``layout`` has ``workers``."""
    itemsize = np.dtype(dtype).itemsize

    def shape(block):
        spatial = [int(b) for ax, b in zip(axes, block) if ax in "zyx"]
        mb = int(np.prod(block)) * itemsize / 2**20
        return "x".join(str(b) for b in spatial), f"{mb:.1f} MB" if mb < 10 else f"{mb:.0f} MB"

    text = "chunks {} ({})".format(*shape(layout["chunks"]))
    if layout["shards"] is not None:
        dims, size = shape(layout["shards"])
        n = layout["n_blocks"]
        text += f", shards {dims} ({size}, {n} shard{'s' if n != 1 else ''})"
    else:
        text += ", no shards"
    if "workers" not in layout:
        return text
    workers = layout["workers"]
    return text + f", {workers} fusion worker{'s' if workers != 1 else ''}" + (" (auto)" if layout["auto_workers"] else "")


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
