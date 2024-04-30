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


def project_frustum_to_image(cam_intrinsics, cam_to_velo, frustum_dict):
    plot_frustums(frustum_dict)
    print(frustum_dict)
    overlap_dict = {}
    plot_data = {cam_id: {} for cam_id in frustum_dict}  # Prepare to store points for all cameras
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(plot_data)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    # Iterate through each camera as the source of the frustum points
    for source_cam_id, frustum_corners in frustum_dict.items():
        overlap_dict[source_cam_id] = []
        source_intrinsics = cam_intrinsics[source_cam_id]
        source_cam_to_velo = cam_to_velo[source_cam_id]
        source_points_image = project_points(source_intrinsics, source_cam_to_velo, frustum_corners)

        # Iterate through each camera to project source frustum points onto their image planes
        for target_cam_id, target_intrinsics in cam_intrinsics.items():
            if source_cam_id == target_cam_id:
                continue

            target_cam_to_velo = cam_to_velo[target_cam_id]
            # Project the source frustum corners onto the target camera's image plane
            target_points_image = project_points(target_intrinsics, target_cam_to_velo, frustum_corners)
            # Check overlap and store points
            if check_overlap(target_points_image, target_intrinsics):
                overlap_dict[source_cam_id].append(target_cam_id)
                if source_cam_id not in plot_data[target_cam_id]:
                    plot_data[target_cam_id][source_cam_id] = []
                plot_data[target_cam_id][source_cam_id].append(target_points_image[target_points_image[:, 2] > 0])

    print(
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(plot_data)
    print(
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Plot overlapping points for each camera
    for cam_id, data in plot_data.items():
        if data:  # If there are overlapping points to plot for this camera
            plot_overlapping_points(data, cam_intrinsics[cam_id], cam_id)

    return overlap_dict

def plot_overlapping_points(plot_data, intrinsics, camera_id):
    # Extract the image dimensions from the intrinsics matrices
    width = intrinsics[0, 2] * 2
    height = intrinsics[1, 2] * 2

    # Setup the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink', 'gray', 'olive']
    color_iter = iter(colors)

    for source_cam_id, points_list in plot_data.items():
        color = next(color_iter, 'black')  # Cycle through colors, default to black if exhausted
        for points in points_list:
            ax.scatter(points[:, 0], points[:, 1], color=color, label=f'From {source_cam_id}')

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_title(f'Overlapping Points on {camera_id}')
    ax.set_xlabel('Image Width')
    ax.set_ylabel('Image Height')
    ax.legend(title='Point Sources')
    ax.invert_yaxis()  # Invert the y-axis to match the image coordinate system
    plt.show()

def project_points(intrinsics, cam_to_velo, points):
    # Convert points to homogeneous coordinates
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])

    # Transform points from world to camera coordinates
    points_cam = np.dot(np.linalg.inv(cam_to_velo), points_homo.T).T

    # Filter points in front of the camera
    points_cam = points_cam[points_cam[:, 2] > 0]

    # Apply intrinsic matrix to project points onto the image plane
    # Only take the first three rows (x, y, z)
    points_image = np.dot(intrinsics, points_cam[:, :3].T).T
    points_image = points_image / points_image[:, 2][:, np.newaxis]  # Normalize by the depth

    return points_image


def check_overlap(points_image, intrinsics):
    img_width = intrinsics[0, 2] * 2
    img_height = intrinsics[1, 2] * 2
    in_bounds = (points_image[:, 0] >= 0) & (points_image[:, 0] < img_width) & \
                (points_image[:, 1] >= 0) & (points_image[:, 1] < img_height)
    if np.any(in_bounds):
        print(f"Points within image bounds: {points_image[in_bounds]}")  # Debug: print points within bounds
    return np.any(in_bounds)


def plot_frustums(frustum_dict):
    """
    Plots the frustums in 3D space with edges connecting the corners.

    :param frustum_dict: A dictionary where keys are camera IDs and values are numpy arrays of shape (8, 3) representing frustum corners.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for cam_id, corners in frustum_dict.items():
        # Ensure the corners are in the correct order, if not, reorder them
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom rectangle
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top rectangle
            [0, 4], [1, 5], [2, 6], [3, 7]  # Side edges connecting bottom and top rectangles
        ]
        colors = [[1, 0, 0] for _ in range(len(lines))]  # Red lines

        # Create Open3D line set from corners and lines
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    # Run the visualizer
    vis.run()
    vis.destroy_window()