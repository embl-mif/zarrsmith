"""Physical position (NGFF ``translation``) of a dataset.

ZarrSmith keeps the position of a dataset in ``metadata["translations"]``: one dict
per pyramid level, keyed by spatial axis, giving the physical position (in the units
of ``metadata["units"]``) of the centre of voxel ``[0, ..., 0]``::

    metadata["translations"] = [{"z": -1049.0, "y": 198.5, "x": 8988.5}, ...]

An axis missing from a dict is at ``0.0``. ``None`` (or no key at all) means the
position is unknown: no NGFF ``translation`` is written, exactly as before
translations were supported. An all-zero translation is not written either, which
NGFF reads the same way (no translation is the identity).

Coarser levels follow the NGFF 0.6 convention (see :func:`pyramid_translations`):
a pixel's centre is its position, so a level whose pixels summarize blocks of the
finer level ("mean", "mode", binning) sits half a block further:
``t_k = t_0 + (s_k - s_0) / 2``. A "nearest" level, which keeps the first pixel of
every block, keeps the translation of level 0.
"""
from __future__ import annotations

from numbers import Real
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .axes import SPATIAL_AXIS_SET, normalize_axes, spatial_axes_in_order

Translation = Dict[str, float]


def _as_dict(value: Any, spatial_axes: Sequence[str]) -> Translation:
    """One level's translation as a dict over ``spatial_axes`` (missing axes are 0)."""
    if isinstance(value, Mapping):
        unknown = [k for k in value if k not in spatial_axes]
        if unknown:
            raise ValueError(
                f"Translation axes {unknown} are not spatial axes of the dataset {tuple(spatial_axes)}."
            )
        items = {str(k): value[k] for k in value}
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(spatial_axes):
            raise ValueError(
                f"A translation of {len(value)} values does not match the spatial axes "
                f"{tuple(spatial_axes)}."
            )
        items = dict(zip(spatial_axes, value))
    else:
        raise TypeError("A translation must be a dict such as {'y': 1.0, 'x': 2.0} or a sequence of numbers.")
    out = {}
    for axis in spatial_axes:
        v = items.get(axis, 0.0)
        if not isinstance(v, Real) or isinstance(v, bool):
            raise TypeError(f"Translation of axis {axis!r} must be a number, got {v!r}.")
        out[axis] = float(v)
    return out


def normalize_translations(
    translations: Any,
    axes: str | Sequence[str],
    n_levels: Optional[int] = None,
) -> Optional[List[Translation]]:
    """``translations`` as one dict per level over the spatial axes of ``axes``, or ``None``.

    Accepts ``None``; one translation for every level (a dict or a sequence of numbers
    in spatial axis order); or a list with one translation per level. With
    ``n_levels``, a single translation is repeated for every level and a list of
    another length raises.
    """
    if translations is None:
        return None
    spatial = spatial_axes_in_order(normalize_axes(axes))
    per_level = (
        isinstance(translations, Sequence)
        and not isinstance(translations, (str, bytes))
        and len(translations) > 0
        and all(isinstance(t, Mapping) or (isinstance(t, Sequence) and not isinstance(t, (str, bytes)))
                for t in translations)
    )
    if not per_level:
        single = _as_dict(translations, spatial)
        return [dict(single) for _ in range(n_levels or 1)]
    levels = [_as_dict(t, spatial) for t in translations]
    if n_levels is not None and len(levels) != n_levels:
        if len(levels) == 1:
            return [dict(levels[0]) for _ in range(n_levels)]
        raise ValueError(
            f"'translations' has {len(levels)} entries, expected one per pyramid level ({n_levels})."
        )
    return levels


def has_translation(translations: Optional[Sequence[Mapping[str, float]]]) -> bool:
    """Whether any level is away from the origin."""
    return bool(translations) and any(any(float(v) != 0.0 for v in t.values()) for t in translations)


def translation_vector(translation: Mapping[str, float], axes: str | Sequence[str]) -> List[float]:
    """The NGFF ``translation`` list over all ``axes`` (``0.0`` for ``t``, ``c`` and missing axes)."""
    return [float(translation.get(ax, 0.0)) if ax in SPATIAL_AXIS_SET else 0.0 for ax in normalize_axes(axes)]


def pyramid_translations(
    translation: Optional[Mapping[str, float]],
    source_scale: Sequence[float],
    level_scales: Sequence[Sequence[float]],
    axes: str | Sequence[str],
    method: str = "mean",
) -> Optional[List[Translation]]:
    """The translation of every level of a pyramid, by the NGFF 0.6 convention.

    ``translation`` is the position of the level whose pixel size is
    ``source_scale``; ``level_scales`` are the pixel sizes of the levels to place
    (both in spatial axis order, like ``metadata["scales"]``). NGFF 0.6 puts a
    pixel's position at its centre, so for block-summarizing methods (``"mean"``,
    ``"mode"``, a vendor's binned levels) each level moves by half the growth of its
    pixel: ``t_k = t + (s_k - s) / 2``, as in the 0.6 multiscales note
    ``(pixel-size-at-resolution-N - pixel-size-at-resolution-0) / 2``.
    ``"nearest"`` keeps the first pixel of every block, so every level keeps ``t``.

    ``None`` stays ``None`` for ``"nearest"``; for the other methods an unknown
    position counts as the origin, since the coarser levels still need the offset
    to line up with level 0.
    """
    spatial = spatial_axes_in_order(normalize_axes(axes))
    if translation is None:
        if method == "nearest":
            return None
        translation = {}
    base = _as_dict(translation, spatial)
    if method == "nearest":
        return [dict(base) for _ in level_scales]
    out = []
    for scale in level_scales:
        out.append({
            ax: base[ax] + (float(s) - float(s0)) / 2.0
            for ax, s, s0 in zip(spatial, scale, source_scale)
        })
    return out


def shifted(translation: Mapping[str, float], offsets: Mapping[str, float]) -> Translation:
    """``translation`` moved by ``offsets`` (same units), axis by axis."""
    out = {k: float(v) for k, v in translation.items()}
    for axis, delta in offsets.items():
        out[axis] = out.get(axis, 0.0) + float(delta)
    return out


def drop_axes(translations: Optional[Sequence[Mapping[str, float]]], axes: Sequence[str]) -> Optional[List[Translation]]:
    """``translations`` without the entries of ``axes`` (e.g. a dropped singleton ``z``)."""
    if translations is None:
        return None
    return [{k: v for k, v in t.items() if k not in axes} for t in translations]
