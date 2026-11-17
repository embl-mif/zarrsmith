# OME-Zarr preferences

Written by `/setup-ome-zarr-skills` on 2026-11-17. Edit by hand freely.
Skills read this file when present; when a field is `unknown` they infer from context.

## Spec version

- target: 0.6
- reason: aim for the current released spec (adds RFC-5 coordinate systems and transformations). The code reads 0.6 images (axes from `coordinateSystems`, scale/translation from a `sequence`) but does not write 0.6 yet: it writes 0.5 / zarr v3 by default and 0.4 / zarr v2 on request, and refuses to modify a 0.6 store in place.
- ilastik (pipeline consumer) reads only zarr v2, so its outputs are 0.4 / v2 without shards; tile size per ilastik job is not measured yet.
- also_read: 0.5, 0.4 (both are also written, and must keep working)
- spec: https://ngff.openmicroscopy.org/specifications/0.6/index.html

## RFCs in use

None beyond the target version. RFC-2 (zarr v3, adopted in 0.5) and RFC-5 (coordinate systems and transformations, part of 0.6) come with the target.

## Domain

- modality: light-sheet (Luxendo, Viventis, SCAPE), confocal/widefield (Zeiss CZI, Evident), HCS mosaics (Opera Phenix, fused per field)
- axes: any subset of `tczyx`, in that order
- units: micrometer (space), second (time)
- dtypes: uint16 raw (default); uint32 labels (inferred); float32 probability maps (inferred)
- typical_size: unknown
- extras: omero, labels (no plate/well, no bioformats2raw.layout)
- storage: local filesystem; chunks up to about 100 MB with one timepoint and one channel per chunk; `shards="auto"` targets about 1 GB and never groups timepoints or channels; Blosc zstd level 1

## Notes

- Never mix pairs: zarr v2 goes with NGFF 0.4 and zarr v3 with NGFF 0.5. A store is one or the other throughout.
- On v2: `dimension_separator="/"`, numcodecs compressor (Blosc by default, gzip allowed), no shards; NGFF metadata flat on the group attributes, `image-label` carries `"version": "0.4"`. On v3: everything NGFF under `ome`.
- NGFF metadata is written through `zarrsmith.utils.ngff._write_ngff_metadata` (ome-zarr-py), so the format decides the layout. Custom group attributes (`metadata["extra_attributes"]`) sit next to `multiscales` on v2 and outside `ome` on v3.
- Per-level origins are `metadata["translations"]`: one dict per level keyed by spatial axis (`zarrsmith.utils.translation`), written as a `translation` after each `scale`. Coarser levels follow the NGFF 0.6 pixel-centre convention: `mean`/`mode`/binned levels at `t0 + (s_k - s0) / 2`, `nearest` levels at `t0`.
- Validation in tests uses `ome-zarr-models`; napari reading goes through `ome_zarr.reader` / `napari-ome-zarr`.
