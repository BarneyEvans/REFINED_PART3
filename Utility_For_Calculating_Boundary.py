import numpy as np

def distance_to_line_along_local_y(point, p1, p2):
    """Calculate the perpendicular distance from a point to a line along the local Y-axis."""
    local_y_axis = define_local_y_axis(p1, p2)
    point_vector = np.array(point) - np.array(p1)
    point_vector[2] = 0  # Ignore the Z component
    projection_length = np.dot(point_vector, local_y_axis)
    return abs(projection_length)

def define_local_y_axis(p1, p2):
    """Define a local Y-axis orthogonal to the line segment in the XY plane."""
    global_z = np.array([0, 0, 1])  # Global Z-axis
    edge_vector = np.array(p2) - np.array(p1)
    edge_vector[2] = 0  # Ignore the Z component
    local_y = np.cross(global_z, edge_vector)  # Cross product to find a vector orthogonal in the XY plane
    local_y_normalized = local_y / np.linalg.norm(local_y)  # Normalize the vector
    return local_y_normalized

def is_near_boundary_and_within_edge(point_distances, base_threshold, max_threshold):
    """Determine if a point is near a boundary and within an edge using a dynamic angular threshold."""
    for entry in point_distances:
        distance, dynamic_threshold, _ = distance_to_line_along_local_y_FOR_LATER(entry['lidar_point'], entry['edge_coordinates'][0], entry['edge_coordinates'][1], base_threshold, max_threshold)
        if distance < dynamic_threshold and is_point_within_edge(entry['lidar_point'], entry['edge_coordinates']):
            return True
    return False

def distance_to_line_along_local_y_FOR_LATER(point, p1, p2, base_threshold, max_threshold):
    """Calculate the perpendicular distance from a point to a line along the local Y-axis and apply angular threshold."""
    local_y_axis = define_local_y_axis(p1, p2)
    point_vector = np.array(point) - np.array(p1)
    point_vector[2] = 0  # Ignore the Z component
    projection_length = np.dot(point_vector, local_y_axis)
    distance = projection_length
    dynamic_threshold = calculate_angular_threshold(p1, p2, base_threshold, max_threshold, point)
    return distance, dynamic_threshold, local_y_axis

def is_point_within_edge(point, edge_coordinates):
    """Check if the LiDAR point is within the segment defined by the edge coordinates."""
    p1, p2 = np.array(edge_coordinates[0]), np.array(edge_coordinates[1])
    edge_vector = p2 - p1
    point_vector = np.array(point) - p1
    # Project point_vector onto edge_vector
    proj_length = np.dot(point_vector, edge_vector) / np.linalg.norm(edge_vector)
    # Check if the projection length is between 0 and the length of the edge_vector
    return 0 <= proj_length <= np.linalg.norm(edge_vector)

def calculate_angular_threshold(p1, p2, base_threshold, max_threshold, point):
    """Calculate dynamic threshold based on the position along the line."""
    line_length = np.linalg.norm(np.array(p2) - np.array(p1))
    point_position = np.linalg.norm(np.array(point) - np.array(p1))
    angle_ratio = point_position / line_length  # Ratio of position along the line
    return base_threshold + (max_threshold - base_threshold) * angle_ratio  # Linear interpolation