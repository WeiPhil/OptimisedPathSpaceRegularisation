import numpy as np

from flip.flip import compute_flip
from flip.utils import *

# Set viewing conditions
monitor_distance = 0.7
monitor_width = 0.7
monitor_resolution_x = 3840

# Compute number of pixels per degree of visual angle
pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)


comparison_images = ["image1","image2","etc"]

for comparison_image in comparison_images:
	# Load sRGB images
	reference = load_image_array('path_to_reference_image.png')
	test = load_image_array('path_to_compared_image/{}.png'.format(comparison_image))

	# Compute FLIP map
	deltaE = compute_flip(reference, test, pixels_per_degree)

	# Save error map
	index_map = np.floor(255.0 * deltaE.squeeze(0))

	use_color_map = True
	if use_color_map:
		result = CHWtoHWC(index2color(index_map, get_magma_map()))
	else:
		result = index_map / 255.0
	save_image("path_to_flip_output/flip_{}.png".format(comparison_image), result)
	print("{} Done".format(comparison_image))
