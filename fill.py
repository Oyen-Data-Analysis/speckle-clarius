from PIL import Image
import pydicom
import cv2
import numpy as np
import SimpleITK as sitk
import os


def fill(original_image_array):
    image_array = original_image_array.copy()
    height, width = image_array.shape[:2]

    # calculate bounds for height
    height_arr = np.zeros((width, 2))

    for pix_width in range (width):
        max_height = 0 
        min_height = height - 1
        for pix_height in range (height):
            if image_array[pix_height, pix_width] == 255:
                if pix_height > max_height:
                    max_height = pix_height
                if pix_height < min_height:
                    min_height = pix_height
            height_arr[pix_width][0] = max_height
            height_arr[pix_width][1] = min_height


    width_arr = np.zeros((height, 2))
    for pix_height in range (height):
        max_width = 0
        min_width = width - 1
        for pix_width in range (width):
            if image_array[pix_height, pix_width] == 255:
                if pix_width > max_width:
                    max_width = pix_width
                if pix_width < min_width:
                    min_width = pix_width
            width_arr[pix_height][0] = max_width
            width_arr[pix_height][1] = min_width

    for pix_width in range(width):
        for pix_height in range(height):
            if (pix_height < height_arr[pix_width][0]) and (pix_height > height_arr[pix_width][1]) and (pix_width < width_arr[pix_height][0]) and (pix_width > width_arr[pix_height][1]):
                image_array[pix_height, pix_width] = 255
            else:
                image_array[pix_height, pix_width] = 0
    return image_array

def fill_and_save_dicom(dicom_path, output_dir):
    dicom_data = pydicom.dcmread(dicom_path).pixel_array
    dicom_image_array = np.where(dicom_data > 0, 255, 0)
    filled_image = fill(dicom_image_array)
    output_path = output_dir + "/"+os.path.basename(dicom_path).replace(".dcm", ".filled_image.mha")
    sitk.WriteImage(sitk.GetImageFromArray(filled_image), output_path)
    return output_path

def fill_and_save_jpg(image_path, output_dir):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.where(image > 0, 255, 0)
    filled_image = fill(image)
    output_path = output_dir + "/"+os.path.basename(image_path).replace(".jpg", ".filled_image.mha")
    sitk.WriteImage(sitk.GetImageFromArray(filled_image), output_path)
    return output_path

paths = ['Output_Segmented_Images/IMG_20240214_1_30.0_segmented.jpg',
         'Output_Segmented_Images/IMG_20240214_1_31.0_segmented.jpg',
         'Output_Segmented_Images/IMG_20240214_1_32.0_segmented.jpg']
for path in paths:
    fill_and_save_jpg(path, "Output_Segmented_Images")