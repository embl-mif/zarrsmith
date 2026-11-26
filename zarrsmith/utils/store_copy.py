"""Shared steps of the writers that copy an OME-Zarr store with its structure (the chromatic
shift correction and the resampling): the datasets are checked before anything is written,
and the output gets the source's zarr format and group attributes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

import zarr


def datasets_on_disk(datasets: Sequence[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    """The ``(name, dataset)`` pairs that hold data, each checked to still match its arrays on
    disk (their chunks, shards and codecs are copied, so they must be known)."""
    datasets = [(name, dataset) for name, dataset in datasets if dataset.data]
    if not datasets:
        raise ValueError("No datasets to write.")
    for name, dataset in datasets:
        if dataset.zarr_data is None or len(dataset.zarr_data) != len(dataset.data):
            raise ValueError(
                f"Dataset {name!r} has in-memory changes that are not on disk, so its "
                "on-disk structure is unknown. Write it with to_zarr() first."
            )
    return datasets


def open_store_copy(source_root: zarr.Group, path: str | Path, datasets: Sequence[Tuple[str, Any]], *,
                    overwrite: bool = False) -> zarr.Group:
    """Create the output store at ``path`` (refused if it would replace data or the source) in
    the zarr format of ``source_root``, with its root attributes and, when label datasets are
    written, those of its ``labels`` group."""
    from .output_path import check_output_path

    check_output_path(path, overwrite=overwrite, sources=[getattr(source_root.store, "root", None)])
    root = zarr.open_group(str(Path(path)), mode="w" if overwrite else "w-",
                           zarr_format=source_root.metadata.zarr_format)
    root.attrs.update(source_root.attrs.asdict())
    if any(name.startswith("labels/") for name, _ in datasets) and "labels" in source_root:
        root.require_group("labels").attrs.update(source_root["labels"].attrs.asdict())
    return root


def copy_group(root: zarr.Group, source_root: zarr.Group, dataset: Any) -> zarr.Group:
    """The output group of ``dataset`` (the root for the raw image), with its source's attributes."""
    group_path = (dataset.path or "/").strip("/")
    if not group_path:
        return root
    group = root.require_group(group_path)
    group.attrs.update(source_root[group_path].attrs.asdict())
    return group
