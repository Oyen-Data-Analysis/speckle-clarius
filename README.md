Image Segmentation Pipeline

(dotted green outline, i.e., clarius, start here)
1. use outline.py to connect the green dots and save as grayscale connected outline in Outlined_Images
OR
(solid white outline, i.e., e22, start here)
1. use mha_to_jpeg.py to change the format of segmentation outline files to .jpeg and move them to Outlined_Images

2. use fill_and_smooth.py to turn outlines into binary masks saved in Output_Masks (Glowing_Placentas is now deprecated)

3. Run segment_placenta_with_mask.py use masks to segment placenta images and save carved-out placentas in Output_Segmented_Images

4. Run match_file_with_patient.py to aggregate information about all masks and segmentation files in Output_Masks and Output_Segmented_Images for ease of use during analysis

Analysis Options:
1. GLCM (tba)
2. Boxed GLCM (tba)