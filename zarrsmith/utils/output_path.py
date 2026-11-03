"""Checks made before writing to an output path, so that existing data is not lost by accident."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

ZARR_METADATA_FILES = ("zarr.json", ".zgroup", ".zarray", ".zattrs")


def has_zarr_name(path: str | Path) -> bool:
    """Whether the name of ``path`` ends in ``.zarr`` (so also ``.ome.zarr``)."""
    return Path(path).name.lower().endswith(".zarr")


def is_zarr_store(path: str | Path) -> bool:
    """Whether the folder ``path`` holds Zarr metadata at its root."""
    return any((Path(path) / name).is_file() for name in ZARR_METADATA_FILES)


def has_content(path: str | Path) -> bool:
    """Whether ``path`` is a file or a non-empty folder."""
    path = Path(path)
    return any(path.iterdir()) if path.is_dir() else path.exists()


def output_refusal(
    path: str | Path,
    *,
    overwrite: bool = False,
    sources: Iterable[Any] = (),
    hint: str = "pass overwrite=True",
) -> str | None:
    """Why ``path`` must not be written to, or ``None`` if it can be.

    A missing path or an empty folder can always be written to. An existing Zarr
    store needs ``overwrite``. Files and non-empty folders that are not Zarr stores
    are never replaced, because writing with ``overwrite`` first deletes everything
    in the output folder.

    ``sources`` are the paths the data is read from (``None`` entries are ignored):
    the output cannot be one of them or contain one, and cannot be inside a source
    that is a Zarr store. Reading is lazy, so replacing a source loses the data.
    ``hint`` says how to allow overwriting, in the terms of the caller.
    """
    out = Path(path).resolve()
    for source in sources:
        if source is None:
            continue
        src = Path(source).resolve()
        if out == src or out in src.parents:
            return f"{path} is or contains the input {source}, which would be lost. Choose another path."
        if src in out.parents and is_zarr_store(src):
            return f"{path} is inside the input store {source}. Choose another path."

    if not has_content(out):
        return None
    if not out.is_dir():
        return f"{path} is an existing file. Choose another path."
    if not overwrite:
        return f"{path} already exists and is not empty; {hint} to replace it, or choose another path."
    if not is_zarr_store(out):
        return f"{path} is not empty and is not a Zarr store, so it is never replaced. Choose another path."
    return None


def check_output_path(
    path: str | Path,
    *,
    overwrite: bool = False,
    sources: Iterable[Any] = (),
    hint: str = "pass overwrite=True",
) -> None:
    """Raise ``FileExistsError`` if ``path`` must not be written to (see :func:`output_refusal`).

    Warn if the output is not named ``.zarr`` or if an existing store will be replaced.
    """
    refusal = output_refusal(path, overwrite=overwrite, sources=sources, hint=hint)
    if refusal is not None:
        raise FileExistsError(refusal)
    if not has_zarr_name(path):
        warnings.warn(f"The output {path} is not named .zarr or .ome.zarr.", UserWarning, stacklevel=3)
    if has_content(path):
        warnings.warn(f"Replacing the existing Zarr store {path}: its content is deleted.",
                      UserWarning, stacklevel=3)
