from Point_Boundary_Seperator import check_point_in_overlaps
from General_Utility import check_folder
from collections import defaultdict
import numpy as np
import os
import cv2



def bounding_boxes_in_overlap(projected_points_to_images, overlap, data, image_folder_path):
    data = extract_relevant_data(data)
    save_folder = os.path.join(image_folder_path, "Overlapping_Boxes")
    extraction_folder = os.path.join(image_folder_path, "Undistorted_Images")
    check_folder(save_folder)

    for names, info in data.items():
        centers, classes, confidences, bounding_boxes = info
        cam_name = names.split("_")[2].split(".")[0]
        for index, centre in enumerate(centers):
            centre = [[centre[0], centre[1]]]
            details, overlap_dict = check_point_in_overlaps(cam_name, centre, projected_points_to_images, overlap)



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