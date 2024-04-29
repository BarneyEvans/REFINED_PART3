import numpy as np
from scipy.ndimage import gaussian_filter

def smooth_all_boundaries(strips, sigma=1):
    smoothed_strips = {}

    # Loop through each camera_id and its list of tuples (strip identifiers and points)
    for camera_id, strips_list in strips.items():
        # Prepare a list to hold smoothed points for the current camera
        smoothed_strips_list = []

        # Iterate through each tuple in the list for the current camera
        for strip_id, points in strips_list:
            # Assuming points are structured as an array like [x, y, 1]
            smooth_points_x = gaussian_filter(points[:, 0], sigma=sigma)
            smooth_points_y = gaussian_filter(points[:, 1], sigma=sigma)

            # Combine them back into the smoothed strip array
            smoothed_points = np.stack((smooth_points_x, smooth_points_y, np.ones_like(smooth_points_x)), axis=-1)

            # Append the smoothed strip info back into the list
            smoothed_strips_list.append((strip_id, smoothed_points))

        # Store the list of smoothed strips back in the dictionary for the current camera
        smoothed_strips[camera_id] = smoothed_strips_list

    return smoothed_strips

