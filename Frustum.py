import numpy as np

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



