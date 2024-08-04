import cv2
import os
from General_Utility import check_folder


def load_and_save_images(dataset, seq_id, frame_id, image_folder, cams):
    image_folder = os.path.join(image_folder, "Undistorted_Images")
    check_folder(image_folder)

    try:
        undistorted_images, _ = dataset.undistort_image_v2(seq_id, frame_id)
    except Exception as e:
        print(f"Error loading undistorted images: {e}")
        return {}

    img_dict = {}
    num_images = len(undistorted_images)
    num_cams = len(cams)

    # Logging the lengths of undistorted_images and cams for debugging
    print(f"Number of undistorted images: {num_images}")
    print(f"Number of cameras: {num_cams}")

    if num_images > num_cams:
        #print("Error: Number of undistorted images exceeds number of cameras.")
        return {}

    for i, img in enumerate(undistorted_images):
        try:
            if i >= num_cams:
                print(f"Skipping image {i} as there are not enough camera names.")
                continue

            img_path = os.path.join(image_folder, f"{seq_id}_{frame_id}_{cams[i]}.jpg")
            print(f"Saving image: {img_path}")
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_dict[f"{seq_id}_{frame_id}_{i}"] = img
        except Exception as e:
            print(f"Error saving image {seq_id}_{frame_id}_{cams[i]}.jpg: {e}")

    return img_dict