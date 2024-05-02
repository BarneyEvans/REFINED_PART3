from ultralytics import YOLO
import os
import cv2
from YOLOV8_Boxes_Class import DetectionInfo


# Load the trained model

def predict_on_images(model_path, image_folder_path, cams):
    model = YOLO(model_path)
    save_folder = os.path.join(image_folder_path, "Images_With_Predictions")
    save_folder_boundaries = os.path.join(image_folder_path, "Overlap_Boundaries_With_Predictions")
    images_folder = os.path.join(image_folder_path, "Undistorted_Images")
    boundary_images_folder = os.path.join(image_folder_path, "Overlap_Boundaries")
    check_folder(save_folder)
    check_folder(save_folder_boundaries)

    detections = []  # List to store detection info for all images
    for i,image_name in enumerate(os.listdir(images_folder)):
        img_path = os.path.join(images_folder, image_name)

        boundary_image_name = image_name.split(".")[0] + "_Boundaries.jpg"
        boundary_img_path = os.path.join(boundary_images_folder, boundary_image_name)
        img = cv2.imread(img_path)
        boundary_img = cv2.imread(boundary_img_path)


        results = model(img, verbose=False)
        if results:
            img = results[0].plot()

            detection_info = DetectionInfo(image_name, results[0], img.shape)
            detections.append(detection_info)  # Append the summary info

            # Draw bounding boxes on boundary_img
            for box_info in detection_info.get_summary()['boxes_info']:
                x1, y1, x2, y2 = box_info['corners']
                cv2.rectangle(boundary_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box

        base_name, ext = os.path.splitext(image_name)
        save_image_name = f"{base_name}_Predictions{ext}"
        save_boundary_image_name = f"{base_name}_Predictions_With_Boundary{ext}"
        save_path = os.path.join(save_folder, save_image_name)
        save_boundary_image_path = os.path.join(save_folder_boundaries, save_boundary_image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert back to RGB to save correctly
        boundary_img = cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_boundary_image_path, boundary_img)
        cv2.imwrite(save_path, img)

    return detections

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


def draw_boxes(image, boxes, labels, confidences, colors=None):
    for (box, label, confidence) in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        label_text = f'{label} ({confidence:.2f})'
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image


