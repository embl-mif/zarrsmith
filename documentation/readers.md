# Readers and conventions

The microscope managers in `zarrsmith.microscope_manager` read a dataset and normalize it to one shared API.

The main reader classes currently exposed by `zarrsmith.microscope_manager` are:

- `ArrayManager` — wrap an in-memory NumPy or Dask array using ZarrSmith metadata conventions.
- `LuxendoManager` — Luxendo XML + HDF5 datasets.
- `OperaManager` — Opera Phenix / Opera PE OME-TIFF style datasets.
- `ScapeManager` — Leica SCAPE OME-TIFF + XLIF datasets.
- `ViventisManager` — Viventis LS1 datasets.
- `ZeissManager` — Zeiss CZI datasets.
- `EvidentManager` — Olympus/Evident cellSens VSI datasets: the `.vsi` file plus its `_<name>_/` folder of `.ets` pixel data, read with slideio. Tiled acquisitions are stored stitched and are read as one image; select a scene with `scene_index` or `scene_name`. The lower-resolution levels that cellSens stores for large images (downsampled in Y and X only) are exposed as extra pyramid levels, so the raw file renders quickly in napari 3D; pass `native_levels=False` to read the full resolution only. Single-tile acquisitions usually have no stored levels.
- `ZarrManager` — NGFF v0.4/v0.5 OME-Zarr datasets.
- `ZarrV04Manager` — compatibility reader for older v0.4-style datasets.

All file-based readers share the same conventions:

- The dataset location is the first argument (`path`; for `ScapeManager` the OME-TIFF file) and `chunks` is an optional TCZYX tuple. When omitted, `chunks` defaults to the manager's `DEFAULT_CHUNKS` (`(1, 1, 8, 4096, 4096)` for most readers, `(1, 1, 8, 1024, 1024)` for `ScapeManager`, `(1, 1, 8, 2048, 2048)` for `EvidentManager`, the native chunking for `ZeissManager`).
- The dataset is read in the constructor, so `.data` (one lazy dask array per pyramid level) and `.metadata` are available immediately. `read()` returns `None`.
- Managers can be used as context managers, which closes any open file handle (HDF5 files for `LuxendoManager`) on exit:

  ```python
  with mm.LuxendoManager("path/to/luxendo_folder") as lux:
      lux.to_zarr("output.zarr")
  ```
- The metadata uses one spelling everywhere: channel colors are `RRGGBB` hex strings (no `#`), spatial units are NGFF names such as `"micrometer"`, and time units are e.g. `"second"`.
- Progress messages (opened scenes, subsets, metadata updates, the Zarr tree printed when opening a store) go through the standard `logging` module under the `"zarrsmith"` logger and are silent by default. Show them with:

  ```python
  import logging
  logging.basicConfig(format="%(message)s")
  logging.getLogger("zarrsmith").setLevel(logging.INFO)
  ```

To support a new acquisition system, see [Adding a microscope manager](contributing.md#adding-a-microscope-manager); for acquisitions saved as separate tiles, see [Mosaics](mosaic.md).

## Position (translation)

`metadata["translations"]` holds where a dataset was acquired: one dict per pyramid level, keyed by spatial axis, with the physical position (in the units of `metadata["units"]`, µm for the vendor readers) of the centre of voxel `[0, 0, 0]`, e.g. `[{"z": -1049.0, "y": 198.5, "x": 8988.5}]`. Coarser levels follow the NGFF 0.6 convention, where a pixel's position is its centre: a level that averages or bins blocks of the level above (`mean`, `mode`, a vendor's binned pyramid such as Opera's) sits at `t0 + (s_k - s0) / 2`, `s_k` being its pixel size, and a `nearest` level keeps the position of level 0. It is written as the NGFF `translation` after the `scale` of every level, so that viewers place datasets acquired at different stage positions side by side, and `ZarrManager` reads it back (combined with a `coordinateTransformations` of the multiscale itself, which some writers use). `None` means the position is unknown: no `translation` is written, as before. Subsetting moves the position to the first kept voxel, a dropped singleton axis drops its entry, resampling keeps it, and pyramids place their levels as above.

Where each reader finds it:

| Reader | Source | Without it |
|---|---|---|
| `LuxendoManager` | BigDataViewer `<ViewRegistrations>` of the XML (translation of the composed affine of the first view; a rotation is not stored) | `None` |
| `ZeissManager` | Corner of the scene's bounding box in the pixel frame of the CZI × voxel size, and the lowest `FocusPosition` of its first plane: the frame of `ZeissMosaicManager`, so a scene and the fusion of its tiles overlay. Files stitched by ZEN keep only the offsets between their scenes. | `None` |
| `OperaManager` | The stitched `stitchresult/<well>_F<n>.ome.tiff` has none: the raw `Images/Index.xml` of the export is looked up two folders up, and the corner of the well's fields is placed as `OperaMosaicManager` places them (well on the plate, `y` flipped, relative `PositionZ`). `stitchresult/FieldTables.json`, when present, selects the fields of cluster `F<n>`. Harmony refines the field positions by about 2 µm. | `None` |
| `ViventisManager` | `<acquisition>/Settings/STAGE_PositionSettings.json`, entry named like the folder (`Position 3` for `Position 3_Settings 1`), used as given: the file does not say whether it is the centre or the corner of the field. | `None` |
| `EvidentManager` | `RWC frame origin` of the scene (x, y) and the stage Z of its first frame (multi-tile acquisitions only; z is 0 otherwise) | `None` |
| `ScapeManager` | not read yet | `None` |
| `ArrayManager` | `translations` in the metadata you pass: a dict, a sequence in spatial axis order, or one per level | `None` |
| `ZarrManager` | the NGFF `translation` | `None` |

Set or clear it on any manager with `update_metadata({"translations": {"y": 10.0, "x": -5.0}})` (`None` clears it). `visualize()` places the napari layers at the position; the objects explorer works in the voxel frame of the image and ignores it.
