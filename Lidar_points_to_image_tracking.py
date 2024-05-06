import cv2
from General_Utility import check_folder
import os

def extract_points(dataset, seq_id, frame_id, save_folder):
    img_buf_dict, point_trackers = dataset.tracking_lidar_to_image(seq_id, frame_id)
    plot_images(img_buf_dict, save_folder, seq_id, frame_id)
    return point_trackers

def plot_images(image_dict, save_folder, seq_id, frame_id):
    folder = os.path.join(save_folder, 'Lidar_Projected_To_Images')
    check_folder(folder)
    for cam_name, img_buf in image_dict.items():
        cv2.imwrite(f"{folder}/{seq_id}_{frame_id}_{cam_name}_lidar_image.png",
                    cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))