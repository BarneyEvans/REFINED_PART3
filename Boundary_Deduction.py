import numpy as np
import matplotlib.pyplot as plt


def get_strip_data_for_camera(camera_id, strips, min_size=20):
    """
    Retrieves the strip data for a specific camera, filtering out strips with fewer than min_size points.

    Args:
    camera_id (str): The identifier for the camera.
    strips (dict): The dictionary containing all cameras' strips data.
    min_size (int): Minimum number of points a strip must have to be considered.

    Returns:
    list: A list of tuples, each containing the strip identifier and the corresponding point,
          but only for strips meeting the min_size criterion.
    """
    strip_data = strips.get(camera_id, [])
    strip_count = {}

    # Count occurrences of each strip identifier
    for strip_id, _ in strip_data:
        if strip_id in strip_count:
            strip_count[strip_id] += 1
        else:
            strip_count[strip_id] = 1

    for data in strip_count:
        print(strip_count[data])
    # Filter out any strips with fewer than min_size points
    filtered_strip_data = [item for item in strip_data if strip_count[item[0]] >= min_size]

    return filtered_strip_data


def determine_query_points(input_points):
    """
    Determines if the input is a single point or a bounding box based on the number of points provided.

    Args:
    input_points (list): A list of points. Each point is represented as [x, y].

    Returns:
    dict: A dictionary containing the type ('single' or 'box') and the points.
    """
    if len(input_points) == 1:
        return {'type': 'single', 'points': input_points}
    elif len(input_points) == 4:
        return {'type': 'box', 'points': input_points}
    else:
        raise ValueError("Invalid number of points provided. Must be either one point or four points.")


def find_closest_point_by_y(query_points, strip_data):
    """
    Finds the closest strip point by Y-coordinate for each query point from each strip.

    Args:
    query_points (list): A list of points, each represented as [x, y].
    strip_data (list): A list of tuples from the strips data for a camera, each containing a strip identifier and a point.

    Returns:
    list: A list containing lists of tuples for each query point, where each tuple contains the closest strip identifier,
          strip point, and the query point it corresponds to.
    """
    all_closest_points = []
    for query_point in query_points:
        closest_points = []
        # Initialize a dictionary to track the closest point for each strip
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
    print(all_closest_points)
    return all_closest_points


def check_relative_x_position(all_closest_points):
    """
    Determines if each query point's X-coordinate is within the expected range of its closest strip point's X-coordinate,
    handling a nested list of closest points.

    Args:
    all_closest_points (list of lists): Each sublist contains tuples for each query point,
                                        each tuple containing a strip identifier, the closest point,
                                        and the query point.

    Returns:
    list of lists: A nested list of results for each query point, each sublist indicating whether
                   the query point meets the X-coordinate condition for each of its closest points.
    """
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
    """
    Validates the overlap conditions for all query points based on the valid_box criteria.

    Args:
    all_x_position_results (list of lists): Nested list of results, each sublist corresponding to a query point,
                                            containing tuples of (strip_id, boolean status, query_point).
    valid_box (str): Specifies the tolerance ("zero", "one", "two").

    Returns:
    list of lists: Each sublist contains strip identifiers where the query point meets the overlap conditions.
    """
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


def compile_results(all_valid_strips, query_points, camera_id, query_type):
    """
    Compiles and formats the results based on the validation checks.

    Args:
    all_valid_strips (list of lists): Each sublist contains validated strip identifiers for each query point or bounding box.
    query_points (list): List of query points or bounding box corners.
    camera_id (str): Camera ID where the check is performed.
    query_type (str): Type of query ("point" or "box").

    Returns:
    list of str: Formatted descriptions of the results for each query.
    """
    compiled_results = []
    for idx, valid_strips in enumerate(all_valid_strips):
        if not valid_strips:
            result = f"This {query_type} does not lie within an overlapping region within Camera {camera_id}"
        else:
            overlap_cams = ', '.join(strip.split('_')[0] for strip in valid_strips)
            if len(valid_strips) == 1:
                result = f"Within Camera {camera_id} the {query_type} {query_points[idx]} lies within the following overlap region of {overlap_cams}"
            else:
                result = f"Within Camera {camera_id} the {query_type} {query_points[idx]} lies within the following overlap regions of {overlap_cams}"
        compiled_results.append(result)

    return compiled_results


def check_point_in_overlaps(camera_id, query_points, strips):
    """
    Determines if query points or a bounding box lies within overlapping regions of a specified camera.

    Args:
    camera_id (str): The camera ID where the check is performed.
    query_points (list): List of points (either a single point or four points defining a bounding box).
    strips (dict): Dictionary of all strips data indexed by camera IDs.

    Returns:
    str: Formatted description of the overlap results.
    """
    # Retrieve strip data for the specified camera
    strip_data = get_strip_data_for_camera(camera_id, strips)

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
    final_results = compile_results(all_valid_strips, query_points, camera_id, query_type)

    #plot_strips_and_points(strips, query_points, camera_id)
    return final_results

def plot_strips_and_points(strips, query_points, camera_id):
    """
    Plots strips and query points for visual analysis.

    Args:
    strips (dict): Dictionary containing strip data where keys are camera IDs.
    query_points (list): List of points to query, visualized distinctly.
    camera_id (str): Camera ID to filter strips for plotting relevant data.
    """
    fig, ax = plt.subplots()
    # Plot each strip
    for strip_id, point in strips.get(camera_id, []):
        ax.plot(point[0], point[1], 'ro-')  # Red 'o' for points and '-' connects them
        ax.text(point[0], point[1], f'{strip_id}', color='blue')  # Label strip ids next to points

    # Plot query points
    for point in query_points:
        ax.plot(point[0], point[1], 'go')  # Green 'o' for query points
        ax.text(point[0], point[1], 'Query Point', color='green')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Strips and Query Points Visualization for Camera {camera_id}')
    plt.grid(True)
    plt.show()
