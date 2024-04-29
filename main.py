from once import ONCE
from Frustum import return_frustums
from Calculate_Boundary import distances_from_points_to_frustums, create_boundary_dict
from General_Utility import image_creation
from Logging import logger
from Boundary_Smoothing import smooth_all_boundaries
import cv2



"""

MAIN FUNCTION FOR CALCULATING THE BOUNDARY

"""

"""
INFORMATION FOR ALL RELEVANT FUNCTIONS
"""

dataset = ONCE(r'C:\Users\be1g21\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Full_DataSet')
seq_id = "000027"
frame_id = "1616100875399"
cam_names = ["cam07", "cam08"]

near_plane = 0.1
far_plane = 150

max_threshold = 0.27
base_threshold = 0.03

save_location = rf"C:\Users\be1g21\OneDrive - University of Southampton\Desktop\Year 3\Semester 2\NLP_FINAL_COURSEWORK\pythonProject1\Images\Tests\Test1\Base_{base_threshold}_Max_{max_threshold}"



"""
Extract lidar points and camera parameters
"""

logger.info("Extracting camera parameters")
new_cam_intrinsics_dict, old_intrinsic_dict, extrinsic_dict = dataset.get_vital_info(seq_id, frame_id)
logger.info("Extracting Pointclud")
img_buf_dict, unique_points, colours = dataset.project_lidar_to_image_with_colour(seq_id, frame_id)

"""
Calculate Frustum Corners
"""
logger.info("Extracting frustum edges")
package_info = [new_cam_intrinsics_dict, old_intrinsic_dict, extrinsic_dict, cam_names, near_plane, far_plane]
frustums, top_edges = return_frustums(package_info)

"""
Calculate Boundary
"""

logger.info("Calculating Distances to edge")
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
projected_points_to_images = image_creation(seq_id, frame_id, lidar_boundary_strips, save_location, dataset)

"""
Smooth Boundary
"""
logger.info("Smooth Boundary")
smooth_points = smooth_all_boundaries(projected_points_to_images)


images_dict = dataset.project_2D_points_to_image(seq_id, frame_id, projected_points_to_images)
for cam_name, img_buf in images_dict.items():
            cv2.imwrite(f'images/Smoothing/Test_1/{base_threshold}_{max_threshold}.jpg'.format(cam_name, frame_id),
                        cv2.cvtColor(img_buf, cv2.COLOR_BGR2RGB))
