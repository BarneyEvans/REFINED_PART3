import numpy as np
from scipy.ndimage import gaussian_filter

def smooth_all_boundaries(strips, sigma=1):
    smoothed_strips = {}

    for camera_id, strip_data in strips.items():
        smoothed_strip_data = []

        for strip_id, points in strip_data:
            # Ensure points are a numpy array for the math operations
            points_array = np.array(points, dtype=float)

            # Apply Gaussian filter to smooth the x and y coordinates
            if points_array.ndim == 2:  # Ensure it is a 2D array
                smooth_points_x = gaussian_filter(points_array[:, 0], sigma=sigma)
                smooth_points_y = gaussian_filter(points_array[:, 1], sigma=sigma)
                smooth_points = np.array([smooth_points_x, smooth_points_y, np.ones_like(smooth_points_x)]).T
            else:
                # Fallback in case points are somehow not in expected 2D form
                smooth_points = points_array

            # Add the smoothed points, ensuring they are in the correct format
            smoothed_strip_data.append((strip_id, smooth_points.squeeze()))  # .squeeze() to remove extra dimensions

        smoothed_strips[camera_id] = smoothed_strip_data

    return smoothed_strips

def interpolate_dots_in_strips(strips, y_interval=10):
    interpolated_strips = {}

    for camera_id, strip_data in strips.items():
        if not strip_data:
            print(f"No data for camera {camera_id}")
            continue  # Skip cameras with no data

        interpolated_strip_data = []

        for i in range(len(strip_data)-1):
            current_strip_id, current_point = strip_data[i]
            _, next_point = strip_data[i+1]

            # Ensure current_point and next_point are numpy arrays for the math operations
            current_point = np.array(current_point, dtype=float)
            next_point = np.array(next_point, dtype=float)

            interpolated_strip_data.append((current_strip_id, tuple(current_point)))

            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            num_dots = int(abs(dy) / y_interval)
            x_interval = dx / (num_dots + 1) if num_dots else 0

            for n in range(1, num_dots + 1):
                new_x = current_point[0] + n * x_interval
                new_y = current_point[1] + n * y_interval * np.sign(dy)
                interpolated_strip_data.append((current_strip_id, (new_x, new_y, 1)))

        # Only add the last point if there is data
        if strip_data:
            last_strip_id, last_point = strip_data[-1]
            interpolated_strip_data.append((last_strip_id, tuple(last_point)))

        interpolated_strips[camera_id] = interpolated_strip_data

    return interpolated_strips




