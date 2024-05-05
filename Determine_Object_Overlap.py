from Point_Boundary_Seperator import check_point_in_overlaps
from General_Utility import check_folder
from collections import defaultdict
import numpy as np
import os
import cv2
import re
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def bounding_boxes_in_overlap(projected_points_to_images, overlap, data, image_folder_path):
    #print(projected_points_to_images["cam03"])
    colour_palette = define_colours()

    data = extract_relevant_data(data)
    save_folder = os.path.join(image_folder_path, "Overlapping_Boxes")
    extraction_folder = os.path.join(image_folder_path, "Undistorted_Images")
    check_folder(save_folder)

    coordinate_info = []

    for names, info in data.items():
        centers, classes, confidences, bounding_boxes = info
        cam_name = names.split("_")[2].split(".")[0]
        image_path = os.path.join(extraction_folder, names)
        colour_mapping = {}
        colours = []
        for index, centre in enumerate(centers):
            centre = [[centre[0], centre[1]]]
            new_info = [centers[index], classes[index], confidences[index], bounding_boxes[index]]
            details, overlap_dict = check_point_in_overlaps(cam_name, centre, projected_points_to_images[cam_name], overlap)
            coordinate_info.append((overlap_dict, new_info))
            colour = get_colour_for_overlap(overlap_dict, colour_palette, colour_mapping)
            colours.append(colour)

        annotated_image = draw_bounding_boxes(image_path, bounding_boxes, colours, classes, confidences)
        if annotated_image is not None:
            annotated_image = draw_legend(annotated_image, colour_mapping, colour_palette, cam_name)
            modified_name = names.split('.')[0] + "_BoundingBox_Overlap.jpg"
            save_annotated_image(annotated_image, save_folder, modified_name)

    return coordinate_info


# Define a vibrant colour palette
def define_colours():
    colours = [
        (255, 0, 0),  # Bright Red
        (0, 175, 0),  # Bright Green
        (0, 0, 255),  # Bright Blue
        (255, 255, 0),  # Yellow
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (255, 0, 255)  # Magenta
    ]
    return colours


def get_colour_for_overlap(overlap_dict, colour_palette, colour_mapping):
    if overlap_dict:
        cam_id, overlaps = next(iter(overlap_dict.values()))
        overlaps_str = str(overlaps)
        camera_names = re.findall(r"'(cam\d+)'", overlaps_str)
        key = ', '.join(sorted(set(camera_names)))
        if key not in colour_mapping:
            colour_mapping[key] = colour_palette[len(colour_mapping) % len(colour_palette)]
        return colour_mapping[key]
    else:
        default_key = "No Overlap"
        if default_key not in colour_mapping:
            colour_mapping[default_key] = colour_palette[len(colour_mapping) % len(colour_palette)]
        return colour_mapping[default_key]


def draw_bounding_boxes(image_path, bounding_boxes, colours, classes, confidences):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    for box, colour, class_name, confidence in zip(bounding_boxes, colours, classes, confidences):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        label_background_top_left = (x1, y1 - text_height - baseline - 3)
        label_background_bottom_right = (x1 + text_width, y1)

        cv2.rectangle(image, label_background_top_left, label_background_bottom_right, colour, -1)

        text_position = (x1, y1 - baseline - 2)
        cv2.putText(image, label, text_position, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return image


def draw_legend(image, colour_mapping, colour_palette, camera_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_x, start_y = 30, 30
    box_width = 250
    line_height = 20
    padding = 5

    box_height = padding * 3 + line_height * (len(colour_mapping) + 1)

    top_left = (start_x, start_y)
    bottom_right = (start_x + box_width, start_y + box_height)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)

    title_text = f"{camera_name}"
    title_text = title_text.capitalize()
    cv2.putText(image, title_text, (start_x + padding, start_y + padding + line_height), font, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)

    for idx, (key, colour) in enumerate(colour_mapping.items()):
        text = key if key != "" else "No overlap"
        colour_rect_start = (start_x + padding, start_y + padding * 2 + line_height * (idx + 1))
        colour_rect_end = (start_x + 30, start_y + padding * 2 + line_height * (idx + 1) + 15)

        cv2.rectangle(image, colour_rect_start, colour_rect_end, colour, -1)

        text_position = (start_x + 40, start_y + padding * 2 + 15 + line_height * (idx + 1))
        cv2.putText(image, text, text_position, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def save_annotated_image(image, save_folder, file_name):
    save_path = os.path.join(save_folder, file_name)
    cv2.imwrite(save_path, image)


def extract_relevant_data(detections):
    relevant_data = {}
    for detection in detections:
        summary = detection.get_summary()
        centers = []
        classes = []
        confidences = []
        bounding_boxes = []
        for box in summary['boxes_info']:
            centers.append(box['center'])
            classes.append(box['class'])
            confidences.append(box['confidence'])
            bounding_boxes.append(box['corners'])
        relevant_data[summary['image_name']] = (centers, classes, confidences, bounding_boxes)
    return relevant_data


def plot_data(relevant_data):
    # Extract x and y coordinates
    x_coords = [point[1][0] for point in relevant_data]
    y_coords = [point[1][1] for point in relevant_data]

    # Create scatter plot
    plt.figure(figsize=(10, 8))  # Set the figure size (optional)
    plt.scatter(x_coords, y_coords, colour='red', marker='o')  # Plot points

    # Optionally, you can label axes and set title
    plt.title('Scatter Plot of Points')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)  # Show grid lines

    # Invert the y-axis if needed to match your specific orientation requirement
    # plt.gca().invert_yaxis()

    plt.show()