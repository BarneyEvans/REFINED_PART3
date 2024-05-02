from Boundary_Deduction import check_point_in_overlaps

def bounding_boxes_in_overlap(cams, projected_points_to_images, overlap, image_height, data):
    data = extract_relevant_data(data)
    for names, info in data.items():
        centers, classes, confidences, bounding_boxes = info
        cam_name = names.split("_")[2].split(".")[0]
        for index, centre in enumerate(centers):
            centre = [[centre[0], centre[1]]]
            details = check_point_in_overlaps(cam_name, centre, projected_points_to_images, overlap, image_height)
            print(details)
            print(classes[index], confidences[index], centers[index])

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
