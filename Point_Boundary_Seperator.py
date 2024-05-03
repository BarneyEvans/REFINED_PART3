import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def check_point_in_overlaps(camera_id, query_points, strips, overlap):
    # Retrieve strip data for the specified camera
    strip_data, missing_cameras = get_strip_data_for_camera(camera_id, strips, overlap, query_points)

    # Determine if the input is a single point or a bounding box
    if len(query_points) == 1:
        query_type = 'point'
    elif len(query_points) == 4:
        query_type = 'box'
    else:
        raise ValueError(
            "Invalid number of points provided. Must be either one point or four points for a bounding box.")

    # Find the closest points by Y-coordinate for each query point
    all_closest_points = find_closest_point_by_y(query_points, strip_data)

    # Check relative X positions based on overlap direction
    all_x_position_results = check_relative_x_position(all_closest_points)

    # Validate overlap conditions based on the type of query
    if query_type == 'point':
        valid_box = "zero"  # Strictest setting for a single point
    else:  # Assuming 'box' type
        valid_box = "one"  # Allowing some flexibility for bounding boxes

    all_valid_strips = validate_overlap_conditions(all_x_position_results, valid_box)

    # Compile and return results
    final_results = compile_results(all_valid_strips, query_points, camera_id, query_type, missing_cameras)

    #if camera_id == "cam03":
    #    plot_strips_and_points(strips, query_points, camera_id)
    return final_results


def get_strip_data_for_camera(camera_id, strips, overlap, query_points):
    #if camera_id == "cam03":
    #    plot_strips_and_points(strips, query_points, camera_id)
    # Get the list of cameras that are supposed to overlap with the given camera_id
    valid_cameras = overlap.get(camera_id, [])
    print(strips)
    strip_data = strips.get(camera_id, [])

    filtered_strip_data = [item for item in strip_data if item[0].split('_')[0] in valid_cameras]

    # Check if there is any camera in the valid_cameras that does not appear in any strip
    strip_cameras = set(strip[0].split('_')[0] for strip in strip_data)
    missing_cameras = []
    for cam in valid_cameras:
        if cam not in strip_cameras:

            missing_cameras.append(cam)
    if missing_cameras:
        pass
        #print(f"Assuming all points in Camera {camera_id} overlap with: {', '.join(missing_cameras)}")

    return filtered_strip_data, missing_cameras

def find_closest_point_by_y(query_points, strip_data):
    all_closest_points = []
    for query_point in query_points:
        closest_points = []
        min_y_diffs = {}
        closest_for_each_strip = {}

        for strip_id, point in strip_data:
            # Initialize if strip_id is new
            if strip_id not in min_y_diffs:
                min_y_diffs[strip_id] = float('inf')
                closest_for_each_strip[strip_id] = None

            y_diff = abs(point[1] - query_point[1])
            if y_diff < min_y_diffs[strip_id]:
                min_y_diffs[strip_id] = y_diff
                closest_for_each_strip[strip_id] = (strip_id, point, query_point)

        # Collect closest points for this query point from each strip
        for strip_point in closest_for_each_strip.values():
            if strip_point:
                closest_points.append(strip_point)

        all_closest_points.append(closest_points)
    return all_closest_points


def check_relative_x_position(all_closest_points):
    all_results = []
    for closest_points in all_closest_points:
        results = []
        for strip_id, closest_point, query_point in closest_points:
            if 'strip0' in strip_id:  # Assuming 'strip0' means overlap starts, check query X >= closest X
                results.append((strip_id, query_point[0] >= closest_point[0], query_point))
            elif 'strip1' in strip_id:  # Assuming 'strip1' means overlap ends, check query X <= closest X
                results.append((strip_id, query_point[0] <= closest_point[0], query_point))
        all_results.append(results)
    return all_results


def validate_overlap_conditions(all_x_position_results, valid_box):
    all_valid_strips = []
    tolerance_map = {"zero": 0, "one": 1, "two": 2}
    allowed_failures = tolerance_map.get(valid_box, 0)

    for x_position_results in all_x_position_results:
        valid_strips = []
        strip_groups = {}

        for strip_id, status, point in x_position_results:
            if strip_id not in strip_groups:
                strip_groups[strip_id] = []
            strip_groups[strip_id].append(status)

        for strip_id, statuses in strip_groups.items():
            if statuses.count(False) <= allowed_failures:
                valid_strips.append(strip_id)

        all_valid_strips.append(valid_strips)

    return all_valid_strips


def compile_results(all_valid_strips, query_points, camera_id, query_type, missing_cameras):
    compiled_results = []
    overlap_dict = {}
    for idx, valid_strips in enumerate(all_valid_strips):
        # Get overlapping camera IDs from strip identifiers
        overlap_cams = [strip.split('_')[0] for strip in valid_strips]
        if missing_cameras:
            overlap_cams.extend(missing_cameras)
        # Format results message
        if not valid_strips and len(missing_cameras) == 0:
            result = f"This {query_type} does not lie within an overlapping region within Camera {camera_id}"
            overlap_dict[tuple(query_points[idx])] = (camera_id, [])
        else:
            if len(valid_strips) == 1:
                result = f"Within Camera {camera_id} the {query_type} {query_points[idx]} lies within the following overlap region of {', '.join(overlap_cams)}"
            else:
                result = f"Within Camera {camera_id} the {query_type} {query_points[idx]} lies within the following overlap regions of {', '.join(overlap_cams)}"
            overlap_dict[tuple(query_points[idx])] = (camera_id, overlap_cams)
        compiled_results.append(result)

    return compiled_results, overlap_dict


def plot_strips_and_points(strips, query_points, camera_id):
    fig, ax = plt.subplots()

    # Check and extract data from strips
    strip_data = strips.get(camera_id, [])
    if not strip_data or not all(len(data) == 2 for data in strip_data):
        print("Error: Data format is incorrect or missing")
        return

    # Generate a color map for unique strip_ids
    unique_strip_ids = list(set(strip_id for strip_id, _ in strip_data))
    color_map = get_cmap('tab20')  # Use a colormap with sufficient distinct colors
    colors = {strip_id: color_map(i / len(unique_strip_ids)) for i, strip_id in enumerate(unique_strip_ids)}

    # Plot each strip point
    legend_handled = set()  # To handle legend entries
    for strip_id, point in strip_data:
        if strip_id not in legend_handled:
            ax.plot(point[0], point[1], 'o-', color=colors[strip_id], label=f'Strip {strip_id}')
            legend_handled.add(strip_id)
        else:
            ax.plot(point[0], point[1], 'o-', color=colors[strip_id])

    # Plot query points
    for point in query_points:
        ax.plot(point[0], point[1], 'go')  # Green 'o' for query points
        ax.text(point[0], point[1], 'Query Point', color='green', fontsize=8, verticalalignment='bottom')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Strips and Query Points Visualization for Camera {camera_id}')
    plt.grid(True)
    plt.legend()
    plt.show()