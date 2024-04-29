import open3d as o3d
import numpy as np
import os
import cv2


def visualize_coloured_frustums_with_point_cloud(lidar_points, point_colours, frustums, output_bool):
    if len(lidar_points) != len(point_colours):
        raise ValueError("The number of LiDAR points must match the number of colour entries.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    for i, colour in enumerate(point_colours):
        if np.array_equal(colour, [120, 120, 120]):
            point_colours[i] = [0, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(point_colours)
    geometries = [pcd]
    frustum_line_equations = []

    # Add each frustum to the visualization and collect line equations
    for points_lidar, colour in frustums:
        # Define the edge pairs for the full frustum visualization
        full_edge_pairs = [[i, i + 4] for i in range(4)] + \
                          [[i, (i + 1) % 4] for i in range(4)] + \
                          [[i + 4, (i + 1) % 4 + 4] for i in range(4)]

        # Define the line set for full frustum visualization
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_lidar),
            lines=o3d.utility.Vector2iVector(full_edge_pairs),
        )
        line_set.colors = o3d.utility.Vector3dVector([colour for _ in full_edge_pairs])
        geometries.append(line_set)

        # Focus only on the top vertical edge pairs for line equation extraction
        top_edge_pairs = [[2, 6], [3, 7]]
        for start_idx, end_idx in top_edge_pairs:
            P0 = points_lidar[start_idx]
            P1 = points_lidar[end_idx]
            direction = np.array(P1) - np.array(P0)
            frustum_line_equations.append((P0, direction))

    if output_bool:
        # Draw all geometries together
        o3d.visualization.draw_geometries(geometries)

    # Return the line equations for the top vertical edges
    return frustum_line_equations


def image_creation(seq, frame, frust_ums, save_location, dataset):
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    if isinstance(frust_ums, dict):
        img_buf_dict = dataset.project_own_lidar_to_image_remove_noise(seq, frame, frust_ums)
    else:
        img_buf_dict, lidar_projected_on_to_camera_dict = dataset.project_own_lidar_to_image(seq, frame, frust_ums)
    for cam_name, img_buf in img_buf_dict.items():
        cv2.imwrite(os.path.join(save_location, f"{cam_name}_{seq}_{frame}.jpg"),
                    cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))
    return img_buf_dict
