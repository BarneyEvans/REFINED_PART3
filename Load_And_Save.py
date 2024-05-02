import cv2
import os
from General_Utility import check_folder



def load_and_save_images(dataset, seq_id, frame_id, image_folder, cams):
    image_folder = os.path.join(image_folder, "Undistorted_Images")
    check_folder(image_folder)
    undistorted_images, _ = dataset.undistort_image_v2(seq_id, frame_id)
    img_dict = {}
    for i, img in enumerate(undistorted_images):
        cv2.imwrite(os.path.join(image_folder, f"{seq_id}_{frame_id}_{cams[i]}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_dict[f"{seq_id}_{frame_id}_{i}"] = img
