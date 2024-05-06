from once import ONCE
from Frustum import return_frustums, project_frustum_to_image
from Calculate_Boundary import distances_from_points_to_frustums, create_boundary_dict
from General_Utility import image_creation
from Logging import logger
from YoloV8_On_Dataset import predict_on_images
from Load_And_Save import load_and_save_images
from Determine_Object_Overlap import bounding_boxes_in_overlap
from Lidar_points_to_image_tracking import extract_points
from Calculate_Objects_Across_Image import object_across_image
import time
import datetime



"""

MAIN FUNCTION FOR CALCULATING THE BOUNDARY

"""

"""
INFORMATION FOR ALL RELEVANT FUNCTIONS
"""

dataset = ONCE(r'C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Full_DataSet')
#seq_id = "000076"
#frame_id = "1616343528200"
seq_id = "000028"
frame_id = "1616102478800"
cam_names = ["cam01", "cam03", "cam05", "cam06", "cam07", "cam08", "cam09"]

near_plane = 0.1
far_plane = 150

max_threshold = 0.27
base_threshold = 0.03

query_points_single = [[100,800]]

image_height = 1020
image_width = 1920

yolo_model_path = r"C:\Users\evans\OneDrive - University of Southampton\Desktop\FINAL\REFINED_PART3\YoloPt\best.pt"
image_save = r"C:\Users\evans\OneDrive - University of Southampton\Desktop\FINAL\REFINED_PART3\Output"


#FOR IMAGE TEST
save_location = rf"C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Semester 2\NLP_FINAL_COURSEWORK\pythonProject1\Images\Tests\Test1\Base_{base_threshold}_Max_{max_threshold}"


"""
Load_And_Save_Images
"""
logger.info("Loading and undistorting dataset images")
load_and_save_images(dataset, seq_id, frame_id, image_save, cam_names)


"""
Extract lidar points and camera parameters
"""

logger.info("Extracting camera parameters")
new_cam_intrinsics_dict, old_intrinsic_dict, extrinsic_dict = dataset.get_vital_info(seq_id, frame_id)
logger.info("Extracting Pointcloud")
img_buf_dict, unique_points, colours = dataset.project_lidar_to_image_with_colour(seq_id, frame_id)

"""
Calculate Frustum Corners
"""

logger.info("Extracting frustum edges")
package_info = [new_cam_intrinsics_dict, old_intrinsic_dict, extrinsic_dict, cam_names, near_plane, far_plane]
frustums, top_edges = return_frustums(package_info)

"""
Generate general overlapping regions
"""
time.sleep(0.1)
overlap = project_frustum_to_image(new_cam_intrinsics_dict, extrinsic_dict, frustums, image_width, image_height)
#print(overlap)


"""
Calculate Boundary
"""

logger.info("Calculating lidar distances from frustum edges")
distances = distances_from_points_to_frustums(unique_points, top_edges, max_threshold)


"""
Refine Boundary
"""
logger.info("Refining Boundary")
lidar_boundary_strips = create_boundary_dict(distances, base_threshold, max_threshold)


"""
Display and retrieve 2D strip coordinates
"""
logger.info("Retrieving 2D coordinates")
projected_points_to_images = image_creation(seq_id, frame_id, lidar_boundary_strips, image_save, dataset)


"""
Extract Yolo_Data
"""
logger.info("Predicting_Images_Using_Yolo_Model")
yolo_data = predict_on_images(yolo_model_path, image_save, cam_names)


"""
Determine whether an YOLO bounding box is in the overlap
"""

logger.info("Determining whether an objecting is in the YOLO bounding box overlap")
coordinate_info = bounding_boxes_in_overlap(projected_points_to_images, overlap, yolo_data, image_save)

"""
Extract LIDAR and respective 2d image coordinates for each image
"""

logger.info("Extracting lidar and 2d image coordinates for each image")
point_trackers = extract_points(dataset, seq_id, frame_id, image_save)

"""
Object Tracking
"""
logger.info("Calculating position of object over images in a single frame")
#data = object_across_image(point_trackers, coordinate_info, 0.5)


#print(data)


now = datetime.datetime.now()
print(now.time())


#print(point_trackers["cam05"][0])
#print(coordinate_info[0:1])


#images_dict = dataset.project_2D_points_to_image(seq_id, frame_id, projected_points_to_images)
#for cam_name, img_buf in images_dict.items():
#            cv2.imwrite(f'images/Smoothing/Test_1/{base_threshold}_{max_threshold}_{cam_name}.jpg'.format(cam_name, frame_id),
#                        cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))
