import open3d as o3d
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
from PIL import Image


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
    save_location = os.path.join(save_location, "Overlap_Boundaries")
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    else:
        if os.listdir(save_location):
            for filename in os.listdir(save_location):
                file_path = os.path.join(save_location, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    img_buf_dict, info = dataset.project_own_lidar_to_image_remove_noise(seq, frame, frust_ums)
    for cam_name, img_buf in img_buf_dict.items():
        cv2.imwrite(os.path.join(save_location, f"{seq}_{frame}_{cam_name}_Boundaries.jpg"),
                    cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))

    #plot_checker(info["cam03"])
    return info


def check_folder(save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        for filename in os.listdir(save_folder):
            file_path = os.path.join(save_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def plot_checker(data):
    # Organizing the data by strip
    strip_points = {}
    for strip_id, points in data:
        if strip_id not in strip_points:
            strip_points[strip_id] = []
        strip_points[strip_id].append(points[:2])  # Ignore the third element for plotting

    # Plotting
    fig, ax = plt.subplots()
    color_map = get_cmap('tab20')  # Using a colormap to ensure unique colors
    colors = iter(color_map.colors)  # Create an iterator over the color map

    for strip_id, points in strip_points.items():
        points = np.array(points)  # Convert list of points to a NumPy array for easier slicing
        ax.plot(points[:, 0], points[:, 1], marker='o', linestyle='-', color=next(colors), label=strip_id)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Visualization of Strips')
    ax.set_xlim(left=0)  # Set the minimum x-axis limit to 0
    ax.set_ylim(bottom=0)  # Set the minimum y-axis limit to 0
    plt.grid(True)
    plt.legend()
    plt.show()


def generate_custom_colours(n):
    if n == 7:
        # Custom colour arrangement for 7 cameras
        colours = [
            (1, 0, 0),  # Red
            (0, 1, 0),  # Green
            (0, 0, 1),  # Blue
            (1, 1, 0),  # Yellow
            (1, 0, 1),  # Magenta
            (0, 1, 1),  # Cyan
            (0, 0, 0)  # Black
        ]
    else:
        colours = list(mcolors.CSS4_COLORS.values())
        np.random.shuffle(colours)
        colours = [mcolors.to_rgb(colours[i]) for i in range(n)]

    return colours


def visualise_frustums_with_point_cloud(lidar_points, frustums, output_bool, use_black_frustums):
    geometries = []
    frustum_line_equations = []

    if lidar_points is not None:
        if lidar_points.size == 0:
            raise ValueError("LiDAR points array cannot be empty.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points)

        grey_colour = [[0.5, 0.5, 0.5] for _ in lidar_points]
        pcd.colors = o3d.utility.Vector3dVector(grey_colour)

        geometries.append(pcd)

    if frustums is not None:
        num_frustums = len(frustums)
        if use_black_frustums:
            frustum_colours = [(0, 0, 0) for _ in range(num_frustums)]
        else:
            frustum_colours = generate_custom_colours(num_frustums)

        for idx, (key, points_lidar) in enumerate(frustums.items()):
            full_edge_pairs = [[i, i + 4] for i in range(4)] + \
                              [[i, (i + 1) % 4] for i in range(4)] + \
                              [[i + 4, (i + 1) % 4 + 4] for i in range(4)]

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points_lidar),
                lines=o3d.utility.Vector2iVector(full_edge_pairs),
            )
            line_set.colors = o3d.utility.Vector3dVector([frustum_colours[idx] for _ in full_edge_pairs])
            geometries.append(line_set)

            top_edge_pairs = [[2, 6], [3, 7]]
            for start_idx, end_idx in top_edge_pairs:
                P0 = points_lidar[start_idx]
                P1 = points_lidar[end_idx]
                direction = np.array(P1) - np.array(P0)
                frustum_line_equations.append((P0, direction))

    if output_bool and geometries:
        o3d.visualization.draw_geometries(geometries)

    return frustum_line_equations if frustums is not None else None


def generate_custom_colours(n):
    if n == 7:
        # Custom colour arrangement for 7 cameras
        colours = [
            (1, 0, 0),  # Red
            (0, 1, 0),  # Green
            (0, 0, 1),  # Blue
            (1, 1, 0),  # Yellow
            (1, 0, 1),  # Magenta
            (0, 1, 1),  # Cyan
            (0, 0, 0)  # Black
        ]
    elif n == 2:
        colours = [
            (1, 0, 0),  # Red
            (0, 0, 1)  # Blue
        ]
    else:
        colours = list(mcolors.CSS4_COLORS.values())
        np.random.shuffle(colours)
        colours = [mcolors.to_rgb(colours[i]) for i in range(n)]

    return colours


def visualise_top_edges_with_point_cloud(lidar_points, frustum_edges, output_bool, use_black_edges):
    geometries = []

    if lidar_points is not None:
        if lidar_points.size == 0:
            raise ValueError("LiDAR points array cannot be empty.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points)

        grey_colour = [[0.5, 0.5, 0.5] for _ in lidar_points]
        pcd.colors = o3d.utility.Vector3dVector(grey_colour)

        geometries.append(pcd)

    num_frustums = len(frustum_edges)
    if use_black_edges:
        edge_colours = [(0, 0, 0) for _ in range(num_frustums)]
    else:
        edge_colours = generate_custom_colours(num_frustums)

    for idx, (key, edges) in enumerate(frustum_edges.items()):
        points = []
        lines = []
        for i, (start, end) in enumerate(edges):
            points.append(start)
            points.append(end)
            lines.append([2 * i, 2 * i + 1])

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(points)),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector([edge_colours[idx] for _ in lines])
        geometries.append(line_set)

    if output_bool and geometries:
        o3d.visualization.draw_geometries(geometries)


def stitch_images_horizontally(image_path1, image_path2, output_path):
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    width1, height1 = image1.size
    width2, height2 = image2.size

    new_width = width1 + width2
    new_height = max(height1, height2)

    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))

    new_image.save(output_path)