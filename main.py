from once import ONCE
from Frustum import return_frustums
from Calculate_Boundary import distances_from_points_to_frustums, create_boundary_dict
from General_Utility import image_creation


"""

MAIN FUNCTION FOR CALCULATING THE BOUNDARY

"""

"""
INFORMATION FOR ALL RELEVANT FUNCTIONS
"""

dataset = ONCE(r'C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Year 3 Project\Full_DataSet')
seq_id = "000076"
frame_id = "1616343528200"
cam_names = ["cam07", "cam08"]

near_plane = 0.1
far_plane = 150

max_threshold = 0.27
base_threshold = 0.03

save_location = rf"C:\Users\evans\OneDrive - University of Southampton\Desktop\Year 3\Semester 2\NLP_FINAL_COURSEWORK\pythonProject1\Images\Tests\Test1\Base_{base_threshold}_Max_{max_threshold}"



"""
Extract lidar points and camera parameters
"""

new_cam_intrinsics_dict, old_intrinsic_dict, extrinsic_dict = dataset.get_vital_info(seq_id, frame_id)
img_buf_dict, unique_points, colours, image_points = dataset.project_lidar_to_image_with_colour(seq_id, frame_id)

"""
Calculate Frustum Corners
"""

package_info = [new_cam_intrinsics_dict, old_intrinsic_dict, extrinsic_dict, cam_names, near_plane, far_plane]
frustums, top_edges = return_frustums(package_info)

"""
Calculate Boundary
"""

distances = distances_from_points_to_frustums(unique_points, top_edges, max_threshold)

"""
Refine Boundary
"""

lidar_boundary_strips = create_boundary_dict(distances, base_threshold, max_threshold)

"""
Display and retrieve 2D strip coordinates
"""

projected_points_to_images = image_creation(seq_id, frame_id, lidar_boundary_strips, save_location, dataset)
