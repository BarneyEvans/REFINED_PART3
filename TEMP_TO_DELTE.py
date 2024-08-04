from General_Utility import stitch_images_horizontally

path1 = r"C:\Users\evans\OneDrive\Desktop\NEW_PROJECT_THIRD_YEAR\Third_Year_Project\REFINED_PART3\Output\Overlap_Boundaries\000076_1616343528200_cam07_Boundaries.jpg"
path2 = r"C:\Users\evans\OneDrive\Desktop\NEW_PROJECT_THIRD_YEAR\Third_Year_Project\REFINED_PART3\Output\Overlap_Boundaries\000076_1616343528200_cam08_Boundaries.jpg"
output = r"C:\Users\evans\OneDrive\Desktop\NEW_PROJECT_THIRD_YEAR\Third_Year_Project\REFINED_PART3\IMAGE_FOLEDR_IGNORE\Test_3.jpg"

stitch_images_horizontally(path1, path2, output)