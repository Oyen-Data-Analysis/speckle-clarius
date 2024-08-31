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
    height_arr = np.zeros((width, 3))

    for pix_width in range (width):
        max_height = 0
        min_height = height - 1
        concave = 0
        for pix_height in range (height):
            if grayscale_mask[pix_height, pix_width] == 255:
                if pix_height > max_height:
                    max_height = pix_height
                if pix_height < min_height:
                    min_height = pix_height
        # For a none-empty column, check if it is concave
        if max_height > min_height:
            for pix_height in range (min_height, max_height + 1):
                if grayscale_mask[pix_height, pix_width] == 0:
                    concave = 1
        # For an empty column, set concave to 1 to exclude from the rectangle
        else:
            concave = 1
        height_arr[pix_width][0] = max_height
        height_arr[pix_width][1] = min_height
        height_arr[pix_width][2] = concave
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
        if height_arr[starting_x][2] == 1:
            continue
        max_area_from_x = 0
        max_width_from_x = 0
        max_height_from_x = 0
        height = height_arr[starting_x][0] - height_arr[starting_x][1] + 1
        if height <= 0:
            continue
        starting_y = height_arr[starting_x][1]
        # Shrink height if necessary and update largest rectangle from x column
        for ending_x in range(starting_x, len(height_arr)):
            if height_arr[ending_x][2] == 1:
                break
            height = min(min(height_arr[ending_x][0], height_arr[starting_x][0]) - max(height_arr[ending_x][1], top_left_y) + 1, height)
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
    # For Debugging purposes, use a small number of patients
    # patient_ids = patient_ids[3:4]
    # Or not
    mask_paths = {patient_id: rows[rows['ID'] == patient_id]['Path'] for patient_id in patient_ids}
    # Find largest rectangle coordinates for each mask
    largest_rectangles = []
    for id in mask_paths.keys():
        patient_max_area = 0
        patient_mask_paths = mask_paths[id]
        for mask_path in patient_mask_paths:
            mask_array = read_in_mask(mask_path)
            # Find the largest horizontal rectangle in the mask
            height_arr = find_boundaries(mask_array)
            horizontal_top_left_x, horizontal_top_left_y, horizontal_width, horizontal_height = largest_rectangle(height_arr)
            # Find the largest vertical rectangle in the mask by flipping the image (transpose)
            width_arr = find_boundaries(mask_array.T)
            vertical_top_left_y, vertical_top_left_x, vertical_height, vertical_width = largest_rectangle(width_arr)
            # Compare the horizontal and vertical rectangles to find the largest rectangle
            if horizontal_width * horizontal_height > vertical_width * vertical_height:
                temp_top_left_x = horizontal_top_left_x
                temp_top_left_y = horizontal_top_left_y
                temp_width = horizontal_width
                temp_height = horizontal_height
                orientation = "horizontal"
            else:
                temp_top_left_x = vertical_top_left_x
                temp_top_left_y = vertical_top_left_y
                temp_width = vertical_width
                temp_height = vertical_height
                orientation = "vertical"
            # Update the largest rectangle if the current rectangle is larger
            if temp_width * temp_height > patient_max_area:
                patient_max_area = temp_width * temp_height
                patient_largest_rectangle = (id, mask_path, temp_top_left_x, temp_top_left_y, temp_width, temp_height, patient_max_area, orientation)    
        print(patient_largest_rectangle)
        largest_rectangles.append(patient_largest_rectangle)

    # Return the result
    return largest_rectangles

# Example usage:
e22_largest_size_list = tally_largest_sizes(group="E-22", mode="per patient")
e22_df = pd.DataFrame(e22_largest_size_list, columns=["ID", "Mask Path", "Top Left X", "Top Left Y", "Width", "Height", "Area", "Orientation"])
e22_df.to_csv("e22_largest_size.csv", index=False)
# e22_largest_size_list = tally_largest_sizes(group="E-22", mode="per image")
clarius_largest_size_list = tally_largest_sizes(group="clarius", mode="per patient")
clarius_df = pd.DataFrame(clarius_largest_size_list, columns=["ID", "Mask Path", "Top Left X", "Top Left Y", "Width", "Height", "Area", "Orientation"])
clarius_df.to_csv("clarius_largest_size.csv", index=False)
# clarius_largest_size_list = tally_largest_sizes(group="clarius", mode="per image")