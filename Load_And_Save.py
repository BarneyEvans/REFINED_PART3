import cv2
import os
from once import ONCE



def load_and_save_images(dataset, seq_id, frame_id, image_folder, cams):
    image_folder = os.path.join(image_folder, "Undistorted_Images")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    else:
        if os.listdir(image_folder):
            for filename in os.listdir(image_folder):
                file_path = os.path.join(image_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    undistorted_images, _ = dataset.undistort_image_v2(seq_id, frame_id)

    img_dict = {}
    for i, img in enumerate(undistorted_images):
        cv2.imwrite(os.path.join(image_folder, f"{seq_id}_{frame_id}_{cams[i]}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_dict[f"{seq_id}_{frame_id}_{i}"] = img
