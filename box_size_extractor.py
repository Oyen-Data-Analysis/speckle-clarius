import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re
from config import *
import pandas as pd

PATIENT_ID_REGEX = re.compile(r"FGR\d{3}-1")

def read_in_mask(path):
    mask_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask_array

def find_boundaries(grayscale_mask):
    height, width = grayscale_mask.shape

    # calculate bounds for height
    height_arr = np.zeros((width, 2))

    for pix_width in range (width):
        max_height = 0
        min_height = height - 1
        for pix_height in range (height):
            if grayscale_mask[pix_height, pix_width] == 255:
                if pix_height > max_height:
                    max_height = pix_height
                if pix_height < min_height:
                    min_height = pix_height
            height_arr[pix_width][0] = max_height
            height_arr[pix_width][1] = min_height
    return height_arr

def largest_rectangle(height_arr):
    max_area = 0
    top_left_x = 0
    top_left_y = 0
    # Find the left-top of placenta
    for x_coord, max_min_height in enumerate(height_arr):
        if max_min_height[0] - max_min_height[1] > 0:
            top_left_x = x_coord
            top_left_y = max_min_height[0]
            break
    max_height = 0
    max_width = 0
    # Find largest rectangle starting from current column
    for starting_x in range(top_left_x, len(height_arr)):
        max_area_from_x = 0
        max_width_from_x = 0
        max_height_from_x = 0
        height = height_arr[starting_x][0] - height_arr[starting_x][1] + 1
        if height <= 0:
            continue
        starting_y = height_arr[starting_x][1]
        # Shrink height if necessary and update largest rectangle from x column
        for ending_x in range(starting_x, len(height_arr)):
            height = min(height_arr[ending_x][0], height_arr[starting_x][0]) - max(height_arr[ending_x][1], top_left_y) + 1
            width = ending_x - starting_x + 1
            area = width * height
            if area > max_area_from_x:
                if height_arr[ending_x][1] > starting_y:
                    starting_y = height_arr[ending_x][1]
                max_area_from_x = area
                max_width_from_x = width
                max_height_from_x = height
        # If the max area of the rectangle starting from x is greater than the current max area, update the max area
        if max_area_from_x > max_area:
            max_area = max_area_from_x
            top_left_x = starting_x
            top_left_y = starting_y
            max_width = max_width_from_x
            max_height = max_height_from_x
    return top_left_x, top_left_y, max_width, max_height

def tally_largest_sizes(group, mode="per patient"):
    FILE_PATHS = "File_Patient_Info.csv"
    # Read the CSV file
    df = pd.read_csv(FILE_PATHS)

    # Get all mask file paths associated with current group
    rows = df[(df['Machine'] == group.lower()) & (df['File Type'] == 'mask')]
    patient_ids = rows['ID'].unique()
    # For Debugging purposes, use only one patient
    patient_ids = patient_ids[:1]
    mask_paths = {patient_id: rows[rows['ID'] == patient_id]['Path'] for patient_id in patient_ids}
    # Find largest rectangle coordinates for each mask
    largest_rectangles = []
    for id in mask_paths.keys():
        patient_max_area = 0
        patient_mask_paths = mask_paths[id]
        for mask_path in patient_mask_paths:
            mask_array = read_in_mask(mask_path)
            height_arr = find_boundaries(mask_array)
            temp_top_left_x, temp_top_left_y, temp_width, temp_height = largest_rectangle(height_arr)
            if temp_width * temp_height > patient_max_area:
                patient_max_area = temp_width * temp_height
                patient_largest_rectangle = (id, mask_path, temp_top_left_x, temp_top_left_y, temp_width, temp_height, patient_max_area)    
        print(patient_largest_rectangle)
        largest_rectangles.append(patient_largest_rectangle)

    # Return the result
    return largest_rectangles

# Example usage:
# Replace '/path/to/your/directory' with the actual directory path containing the DICOM and MHA files.
# graph_largest_rectangles('/path/to/your/directory')
e22_largest_size_list = tally_largest_sizes(group="E-22", mode="per patient")
e22_df = pd.DataFrame(e22_largest_size_list, columns=["ID", "Mask Path", "Top Left X", "Top Left Y", "Width", "Height", "Area"])
e22_df.to_csv("e22_largest_size.csv", index=False)
# e22_largest_size_list = tally_largest_sizes(group="E-22", mode="per image")
clarius_largest_size_list = tally_largest_sizes(group="clarius", mode="per patient")
clarius_df = pd.DataFrame(clarius_largest_size_list, columns=["ID", "Mask Path", "Top Left X", "Top Left Y", "Width", "Height", "Area"])
clarius_df.to_csv("clarius_largest_size.csv", index=False)
# clarius_largest_size_list = tally_largest_sizes(group="clarius", mode="per image")

 # Plotting the histogram of areas with 20 bins
# plt.figure(figsize=(10, 5))
# plt.hist(control[0], bins=50, color='#D7D7D9', )
# plt.hist(fgr[0], bins=50, color='#953017', alpha=.85)
# plt.tight_layout()
# plt.show()
