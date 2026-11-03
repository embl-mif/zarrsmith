"""The checks the napari widgets make before writing, shown in the viewer and in the log of the widget."""
from __future__ import annotations

from typing import Any, Iterable, Optional

from napari.utils.notifications import show_warning

from zarrsmith.utils.output_path import has_content, has_zarr_name, output_refusal


def output_problem(path, *, overwrite: bool = False, sources: Iterable[Any] = ()) -> Optional[str]:
    """Why a widget must not write a dataset to ``path``, or ``None`` if it can.

    The name is checked first: choosing a folder that is not meant to be a dataset is
    how data gets lost. Then what is already there, see
    :func:`zarrsmith.utils.output_path.output_refusal`.
    """
    if not path or not has_zarr_name(path):
        return f"Not writing to {path}: the output name must end in .zarr or .ome.zarr."
    refusal = output_refusal(path, overwrite=overwrite, sources=sources, hint="tick 'Overwrite output'")
    return None if refusal is None else f"Not writing: {refusal}"


def warn_user(message: str) -> None:
    """Show ``message`` in the viewer and keep it in the log."""
    print(f"WARNING: {message}")
    show_warning(message)


def output_allowed(path, *, overwrite: bool = False, sources: Iterable[Any] = ()) -> bool:
    """Whether a widget can write a dataset to ``path``; warns if not, or if a store will be replaced."""
    problem = output_problem(path, overwrite=overwrite, sources=sources)
    if problem is not None:
        warn_user(problem)
        return False
    if has_content(path):
        warn_user(f"{path} already exists: its content will be deleted and replaced.")
    return True
