"""Batch CSV export shared by the napari widgets (no Qt dependency, so unit-testable)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd


def _csv_path(csv_path) -> Path:
    csv_path = Path(csv_path)
    return csv_path if csv_path.suffix.lower() == ".csv" else csv_path.with_suffix(".csv")


def append_batch_csv(csv_path, row: Mapping[str, object], columns: Sequence[str]) -> Path:
    """Append ``row`` to the batch CSV at ``csv_path`` and return the path written.

    The file gets a ``.csv`` suffix if it has none. ``columns`` come first, in that
    order; columns that an existing file already has are kept after them. An
    existing file that cannot be read raises instead of being overwritten.
    """
    csv_path = _csv_path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame([{k: "" if v is None else str(v) for k, v in row.items()}])
    if csv_path.exists() and csv_path.stat().st_size > 0:
        existing = pd.read_csv(csv_path, dtype=str).fillna("")
        table = pd.concat([existing, table], ignore_index=True)
    extra = [c for c in table.columns if c not in columns]
    table.reindex(columns=list(columns) + extra, fill_value="").fillna("").to_csv(csv_path, index=False)
    return csv_path


def batch_rows(csv_path) -> list[dict]:
    """The rows of the batch CSV at ``csv_path`` (``.csv`` added as on append), ``[]`` if none."""
    csv_path = _csv_path(csv_path)
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return []
    return pd.read_csv(csv_path, dtype=str).fillna("").to_dict("records")


def same_path(a, b) -> bool:
    """Whether two paths name the same file (case-insensitive on Windows)."""
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def split_zarr_name(path) -> tuple[Path, str, str]:
    """``(parent, stem, suffix)`` of a store path, the suffix being ``.ome.zarr`` or ``.zarr``."""
    path = Path(path)
    name = path.name
    for suffix in (".ome.zarr", ".zarr"):
        if name.lower().endswith(suffix):
            return path.parent, name[: -len(suffix)], name[-len(suffix):]
    return path.parent, name, ""


def numbered_output(output, taken: Sequence = ()) -> Path:
    """``<name>_1.zarr``, ``<name>_2.zarr``...: the first one not in ``taken`` and not on disk."""
    parent, stem, suffix = split_zarr_name(output)
    n = 1
    while True:
        candidate = parent / f"{stem}_{n}{suffix or '.zarr'}"
        if not candidate.exists() and not any(same_path(candidate, t) for t in taken):
            return candidate
        n += 1


def row_differences(old: Mapping, new: Mapping) -> list[str]:
    """``"num_levels 3 -> 5"`` for every field but the output that differs between two rows."""
    return [f"{key} {old.get(key, '') or '(empty)'} -> {new.get(key, '') or '(empty)'}"
            for key in new
            if key != "output" and str(old.get(key, "") or "") != str("" if new[key] is None else new[key])]


def output_reuse(csv_path, row: Mapping) -> Optional[tuple[str, Path]]:
    """``(why, new output)`` when a row of the batch CSV already writes to ``row["output"]``.

    ``why`` names that row and the settings that differ from it; the new output is the
    next free ``_1``, ``_2``... name, used neither by the CSV nor on disk. ``None`` when
    no row uses the output (an output on disk is the overwrite check's business).
    """
    output = row["output"]
    rows = batch_rows(csv_path)
    same = [(i + 1, r) for i, r in enumerate(rows) if r.get("output") and same_path(r["output"], output)]
    if not same:
        return None
    number, previous = same[-1]
    differences = row_differences(previous, row)
    why = (f"{Path(output).name} is already the output of row {number} of {_csv_path(csv_path).name}, "
           + ("with the same settings." if not differences
              else "with other settings:\n  " + "\n  ".join(differences)))
    return why, numbered_output(output, [r["output"] for r in rows if r.get("output")])
