import os

import numpy as np
import tifffile as tiff
from magicgui import magicgui
from napari import current_viewer
from qtpy.QtWidgets import QWidget, QVBoxLayout
from skimage.color import rgb2gray

from ._layout import compose_dock
from ._output_check import warn_user


# Define functions that will be used during the export process. 

# Function to hide all layers
def hide_layers(viewer):
	# hide all layers
	for l in viewer.layers:
		l.visible = False

def show_layers(viewer):
	# hide all layers
	for l in viewer.layers:
		l.visible = True

# Get current contrast limits
def get_channel_adjustment(viewer):

	ch_names = []
	contrast_limits = []

	for l in viewer.layers:
		# Initiate names
		ch_names.append(l.name)

		# Initiate contrast limits list
		contrast_limits.append(l.contrast_limits)

	settings = dict(zip(ch_names, contrast_limits))

	return settings

def set_settings(viewer, settings: dict):

	# Adjust channels
	for l in viewer.layers:
		l.visible = True
		l.contrast_limits = settings[l.name]

def export_contrast_settings(viewer, output_path="contrast_settings.txt"):
	"""
	Export contrast limits for all layers in a napari viewer to a text file.
	"""

	with open(output_path, "w") as f:
		f.write("Layer\tType\tContrast Limits (min, max)\n")

		for layer in viewer.layers:
			# Most image-like layers have contrast_limits
			if hasattr(layer, "contrast_limits"):
				cl = layer.contrast_limits
				f.write(f"{layer.name}\t{layer.__class__.__name__}\t{cl[0]}, {cl[1]}\n")
			else:
				f.write(f"{layer.name}\t{layer.__class__.__name__}\tN/A\n")


def export_stem(layer, view_id):
    """Start of the name of the files exported for ``layer``, never the name of its source."""
    name = os.path.basename(layer._source.path)
    name = name.replace('.zarr', '') if '.zarr' in name else os.path.splitext(name)[0]
    return name + '-Org-' + view_id


def export_files(layers, view_id):
    """The files of one export: a png per layer, then the merged png, the report and the composite tif."""
    stem = export_stem(layers[-1], view_id)
    channels = [export_stem(layer, view_id) + '_' + str(layer) + '.png' for layer in layers]
    return channels, stem + '_merged.png', stem + '.txt', stem + '_composite.tif'


# Define the main function that will be called when the user clicks the "Export Current View" button in the Napari widget. This function will handle exporting the current view of the Napari viewer, including individual channel views, a merged view, and a composite TIFF file.
def export_widget():
    """Simple Napari widget to export the current view.

    Returns a QWidget containing an input text box and a button. When you click "Export Current View" it will save the current view.
    """

    #global viewer

    @magicgui(
        call_button="Export Current View",
        overwrite={"label": "Overwrite files", "widget_type": "CheckBox", "value": False},
    )
    def _export_widget(View_ID="View-1", overwrite=False):

        viewer = current_viewer()

        # Change to directory where image is located
        os.chdir(os.path.dirname(viewer.layers[0].source.path))

        channel_files, merged_name, report_name, composite_name = export_files(viewer.layers, View_ID)
        existing = [f for f in (*channel_files, merged_name, report_name, composite_name) if os.path.exists(f)]
        if existing and not overwrite:
            warn_user(f"Not exporting: {', '.join(existing)} already exist(s) in {os.getcwd()}. "
                      "Tick 'Overwrite files' or change View_ID.")
            return
        if existing:
            warn_user(f"Replacing {', '.join(existing)} in {os.getcwd()}.")

        input_settings = get_channel_adjustment(viewer)

        # Export current channel views.
        for layer, image_name in zip(viewer.layers, channel_files):

            # hide all layers
            hide_layers(viewer)

            # show only current layer
            layer.visible = True

            # take screenshot of current channel view
            img = viewer.screenshot(image_name, canvas_only=True)

        # Export all channels merged.
        show_layers(viewer)

        # take screenshot of all channels merged
        viewer.screenshot(merged_name, canvas_only=True)

        # Export report
        export_contrast_settings(viewer, report_name)

        # Export not adjusted 3D composite
        # Initiate an array to append channels
        composite_array = []

        # Export current channel views.
        for layer in viewer.layers:

            # hide all layers
            hide_layers(viewer)

            # show only current layer
            layer.visible = True

            # reset contrast
            layer.contrast_limits = (0, 65535) #TODO: to be calculated automatically

            # get channel
            img = viewer.screenshot(canvas_only=True)

            # Convert the rendered colors to grayscale
            # This uses the luminance formula: (0.2125*R + 0.7154*G + 0.0721*B)
            rendered_gray = rgb2gray(img[:, :, :3]) # Ignore alpha channel if present

            # 3. Convert back to uint8 for standard image saving
            rendered_gray_uint8 = (rendered_gray * 255).astype(np.uint8)

            composite_array.append(rendered_gray_uint8)

        # Export not adjusted composite
        stack = np.stack(composite_array, axis=0)  # shape: (n_layers, H, W)
        stack = stack.astype(np.uint8)

        # physical calibration from metadata
        pixel_size_um = layer.scale[-1]   # e.g. µm/pixel in X

        # TIFF resolution is stored in pixels per unit
        # ImageJ expects resolution in pixels per centimeter if resolutionunit='CENTIMETER'

        pixels_per_cm = 10000 / pixel_size_um  # 1 cm = 10,000 µm

        tiff.imwrite(
            composite_name,
            stack,
            imagej=True,
            resolution=(pixels_per_cm, pixels_per_cm),
            resolutionunit='CENTIMETER',
            metadata={
            'unit': 'um'
            }
        )

        # Reset the settings
        set_settings(viewer, input_settings)
    
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.addWidget(_export_widget.native)
    layout.addStretch(1)

    dock = compose_dock(content)
    dock.export_widget = _export_widget  # exposed for tests
    return dock

