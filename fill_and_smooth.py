from PIL import Image
import pydicom
import cv2
import numpy as np
import SimpleITK as sitk
import os
from PIL import ImageFilter

def smooth(image, radius):
    smoothed_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    # Convert smoothed_image to grayscale
    smoothed_gray_array = np.array(smoothed_image)
    # Normalize the image
    smoothed_gray_array = cv2.normalize(smoothed_gray_array, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return smoothed_gray_array

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

def fill_and_save_jpg(image_path, output_dir, smooth_image="large"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.where(image > 0, 255, 0).astype(np.uint8)
    filled_image_array = fill(image)
    filled_image = Image.fromarray(filled_image_array, mode='L')
    if smooth_image == "large":
        smoothed_image_array = smooth(filled_image, radius=20)
    elif smooth_image == "small":
        smoothed_image_array = smooth(filled_image, radius=10)
    if image_path.endswith(".jpg"):
        output_path = output_dir + "/"+os.path.basename(image_path).replace("_outlined.jpg", "_filled.jpg")
    else:
        output_path = output_dir + "/"+os.path.basename(image_path).replace("_outlined.jpeg", "_filled.jpg")
    sitk.WriteImage(sitk.GetImageFromArray(smoothed_image_array), output_path)
    return output_path

paths = []
for path in os.listdir("Outlined_Images"):
    if path.endswith(".jpg"):
        print(fill_and_save_jpg(os.path.join("Outlined_Images", path), "Glowing_Placentas", smooth_image="small"))
    else:
        print(fill_and_save_jpg(os.path.join("Outlined_Images", path), "Glowing_Placentas"))