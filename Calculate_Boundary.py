from Utility_For_Calculating_Boundary import distance_to_line_along_local_y, is_near_boundary_and_within_edge, distance_to_line_along_local_y_FOR_LATER
from collections import defaultdict


def distances_from_points_to_frustums(points, frustum_edges, max_distance):
    """Calculate distances of point to edges"""
    point_line_distances = {i: [] for i, point in enumerate(points)}

    # Iterate over each point and each line
    for i, point in enumerate(points):
        for cam, edges in frustum_edges.items():
            for edge_idx, edge_points in enumerate(edges):
                p1, p2 = edge_points

                distance = distance_to_line_along_local_y(point, p1, p2)
                if distance < max_distance:
                    point_line_distances[i].append({
                        'camera': cam, 'edge_idx': edge_idx,
                        'distance': distance, 'edge_coordinates': edge_points,
                        'lidar_point': point
                    })

    return point_line_distances


def create_boundary_dict(point_distance, base_threshold, max_threshold):
    # Dictionary to hold points by strip with dynamic thresholding
    strip_points = defaultdict(list)
    for i, distances in point_distance.items():
        if is_near_boundary_and_within_edge(distances, base_threshold, max_threshold):
            for entry in distances:
                distance, dynamic_threshold, local_y_axis = distance_to_line_along_local_y_FOR_LATER(
                    entry['lidar_point'], entry['edge_coordinates'][0],
                    entry['edge_coordinates'][1], base_threshold, max_threshold
                )
                temp_distance = abs(distance)
                if temp_distance < dynamic_threshold:
                    strip_name = f"{entry['camera']}_strip{entry['edge_idx']}"
                    strip_points[strip_name].append(entry['lidar_point'])
    return strip_points