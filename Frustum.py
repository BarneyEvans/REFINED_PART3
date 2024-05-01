import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
def calculate_frustum_corners(cam_intrinsics, cam_to_velo, near, far):
    """ Calculates the frustum corners of the camera using intrinsics and extrinsic parameters. """
    # Extracing the focal lengths and the coordinates of the optical centre
    fx, fy, cx, cy = cam_intrinsics[0, 0], cam_intrinsics[1, 1], cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    #Inverse the matrix to allow 2d points to be transformed into 3d points within the system
    inv_intrinsics = np.linalg.inv(np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]))

    # Define frustum corners in image space (homogeneous coordinates)
    points_image = np.array([
        [0, 0, 1], [cx * 2, 0, 1], [cx * 2, cy * 2, 1], [0, cy * 2, 1],  # Near plane corners
        [0, 0, 1], [cx * 2, 0, 1], [cx * 2, cy * 2, 1], [0, cy * 2, 1]   # Far plane corners
    ])

    # Scale points to the near and far planes
    points_image[:4, :] *= near
    points_image[4:, :] *= far

    # Transform points to camera coordinates
    points_camera = np.dot(inv_intrinsics, points_image.T).T

    # Transform points to LiDAR coordinates
    points_lidar = np.dot(cam_to_velo, np.hstack([points_camera, np.ones((8, 1))]).T).T[:, :3]

    return points_lidar

def extract_top_edges(frustums):
    """Extract the top edges of the frustums"""
    top_edges = {}
    for camera, points in frustums.items():
        top_edges[camera] = [
            (points[0], points[4]),
            (points[1], points[5])
        ]
    return top_edges

def return_frustums(list):
    """Return all the frustums as well as top edges to said frustums"""
    frustum_dict = {}
    for cam_name in list[3]:
        points = calculate_frustum_corners(list[0][cam_name], list[2][cam_name], list[4], list[5])
        frustum_dict[cam_name] = points
    top_edges = extract_top_edges(frustum_dict)

    return frustum_dict, top_edges


def project_frustum_to_image(cam_intrinsics, cam_to_velo, frustum_dict, image_width, image_height):
    overlap_dict = {}
    point_dictionary = {}  # New dictionary for counting overlapping points
    points_list = []
    points_cam_map = []  # List to keep track of which points belong to which camera

    # Collect all points into a single list while keeping track of their source camera
    for cam_name, points in frustum_dict.items():
        points_list.extend(points)
        points_cam_map.extend([cam_name] * len(points))  # Repeat cam_name for each point

    points_list = np.array(points_list)

    # Iterate through each camera as the source of the frustum points
    for source_cam_id, frustum_corners in frustum_dict.items():
        overlap_dict[source_cam_id] = []
        point_dictionary[source_cam_id] = {}  # Initialize for point counting

        source_intrinsics = cam_intrinsics[source_cam_id]
        source_cam_to_velo = cam_to_velo[source_cam_id]

        cam_intri = np.hstack([source_intrinsics, np.zeros((3, 1))])
        point_xyz = points_list[:, :3]
        points_homo = np.hstack(
            [point_xyz, np.ones(point_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
        points_lidar = np.dot(points_homo, np.linalg.inv(source_cam_to_velo).T)
        mask = points_lidar[:, 2] > 0
        points_lidar = points_lidar[mask]
        points_cam_map_filtered = [points_cam_map[i] for i, m in enumerate(mask) if m]

        points_img = np.dot(points_lidar, cam_intri.T)
        points_img = points_img / points_img[:, [2]]

        # Check if points are within the image bounds and record overlaps
        for point, cam_id in zip(points_img, points_cam_map_filtered):
            if -5000 <= point[1] < image_width + 5000 and 0 <= point[0] < image_height:
                if cam_id != source_cam_id:  # Avoid self-counting
                    # Update the list in overlap_dict
                    if cam_id not in overlap_dict[source_cam_id]:
                        overlap_dict[source_cam_id].append(cam_id)

                    # Count the points in point_dictionary
                    if cam_id in point_dictionary[source_cam_id]:
                        point_dictionary[source_cam_id][cam_id] += 1
                    else:
                        point_dictionary[source_cam_id][cam_id] = 1

    overlap_dict = refine_overlaps(overlap_dict, point_dictionary)
    return overlap_dict



def refine_overlaps(overlap_dict, point_dictionary):
    # Iterate through the point_dictionary to find entries with significant overlap (more than 4 points)
    for cam_id, overlaps in point_dictionary.items():
        for target_cam, point_count in overlaps.items():
            if point_count > 4:  # Check if the point count is greater than 4
                # Check if the reverse relationship exists in overlap_dict
                if cam_id not in overlap_dict.get(target_cam, []):
                    # If not, add cam_id to the target_cam's list in overlap_dict
                    if target_cam in overlap_dict:
                        overlap_dict[target_cam].append(cam_id)
                    else:
                        overlap_dict[target_cam] = [cam_id]

    return overlap_dict


























