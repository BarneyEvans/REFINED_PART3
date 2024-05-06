import numpy as np


def object_across_image(point_trackers, yolov8_data, distance_threshold):
    data = extract_overlap_points(yolov8_data)
    final_results = {}

    for cameras, detections in data.items():
        results_by_center = process_camera_combinations(point_trackers, cameras, detections)
        for center, camera_results in results_by_center.items():
            if len(camera_results) > 1:  # Ensure there are at least two different cameras
                paired_results = [(camera_results[cam1], camera_results[cam2])
                                  for cam1 in camera_results for cam2 in camera_results if cam1 != cam2]
                if cameras in final_results:
                    final_results[cameras].extend(paired_results)
                else:
                    final_results[cameras] = paired_results

    return final_results


def extract_overlap_points(yolov8_data):
    overlap_points = {}
    for item in yolov8_data:
        camera_dict, detection_info = item
        for (point, (source_camera, overlapping_cameras)) in camera_dict.items():
            if overlapping_cameras:
                overlap_key = tuple(sorted([source_camera] + overlapping_cameras))
                detection_dict = {
                    "source_camid": source_camera,
                    "point": point,
                    "label": detection_info[1],
                    "confidence": detection_info[2],
                    "bbox": detection_info[3]
                }
                if len(set(overlap_key)) > 1:
                    if overlap_key in overlap_points:
                        overlap_points[overlap_key].append(detection_dict)
                    else:
                        overlap_points[overlap_key] = [detection_dict]

    return overlap_points


def process_camera_combinations(point_trackers, cameras, detections):
    results_by_center = {}
    for cam in cameras:
        camera_data = point_trackers.get(cam, [])
        closest_points = find_closest_points(camera_data, detections)
        for center, points in closest_points.items():
            if center not in results_by_center:
                results_by_center[center] = {}
            results_by_center[center][cam] = points[1]  # Store only lidar points

    return results_by_center



def find_closest_points(camera_data, bounding_boxes):
    results = {}
    threshold = 8  # pixels

    for bbox in bounding_boxes:
        center = np.array(bbox[0])  # bbox[0] is the central point of the bounding box
        image_coords = np.array([data[0] for data in camera_data])
        lidar_coords = np.array([data[1] for data in camera_data])

        within_threshold = np.abs(image_coords - center) <= threshold
        filtered_image_coords = image_coords[np.all(within_threshold, axis=1)]
        filtered_lidar_coords = lidar_coords[np.all(within_threshold, axis=1)]

        if filtered_image_coords.size > 0:
            distances = np.linalg.norm(filtered_image_coords - center, axis=1)
            min_index = np.argmin(distances)

            results[tuple(center)] = (tuple(filtered_image_coords[min_index]), tuple(filtered_lidar_coords[min_index]))

    return results


def match_lidar_points(overlap_data, distance_threshold):
    matched_pairs = {}
    for overlap_key, detections in overlap_data.items():
        # Initialize lidar_points for each camera in the current overlap
        lidar_points = {cam: [] for cam in overlap_key}

        # Collect lidar points for each camera
        for cam in overlap_key:
            details = detections.get(cam, [])
            for det in details:
                lidar_point = det.get('closest_lidar_coord')
                if lidar_point:
                    lidar_points[cam].append((lidar_point, det))

        for cam1 in overlap_key:
            for cam2 in overlap_key:
                if cam1 != cam2:
                    for point1, det1 in lidar_points[cam1]:
                        for point2, det2 in lidar_points[cam2]:
                            distance = np.linalg.norm(np.array(point1) - np.array(point2))
                            if distance < distance_threshold:  # Use the provided threshold
                                print(f"Comparing {cam1} {point1} to {cam2} {point2} -> Distance: {distance}")
                                matched_pair_key = (
                                cam1, cam2, det1['closest_image_coord'], det2['closest_image_coord'])
                                matched_pairs.setdefault(matched_pair_key, []).append({
                                    'distance': distance,
                                    'details': {
                                        cam1: det1,
                                        cam2: det2
                                    }
                                })

    return matched_pairs


def filter_close_points(closest_points, distance_threshold):

    filtered_results = []

    for point in closest_points:
        image_coord, lidar_coord, label, confidence, bbox_coords = point
        center = np.array(bbox_coords[:2])  # Assuming bbox_coords are in the form (x1, y1, x2, y2)
        bbox_center = (center[0] + bbox_coords[2]) / 2, (center[1] + bbox_coords[3]) / 2

        # Calculate Euclidean distance from image coordinate to the center of the bounding box
        distance = np.linalg.norm(np.array(image_coord) - np.array(bbox_center))

        if distance <= distance_threshold:
            filtered_results.append(point)

    return filtered_results
