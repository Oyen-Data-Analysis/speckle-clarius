import cv2
import numpy as np
import os
from config import *
import pydicom

for root, _, files in os.walk(MAKERERE_PATH):
    for file in files:
        if file.endswith(".jpeg") and not file.endswith("(2).jpeg"):
            # read in the image
            image = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            # read in the mask
            projected_mask_path = os.path.join(MASK_PATH, file.replace(".jpeg", " _mask.jpg"))
            save_file_name = file.replace(".jpeg", "_clarius_segmented.jpg")
        elif file.endswith(".dcm"):
            # read in the image
            image = pydicom.dcmread(os.path.join(root, file)).pixel_array
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            # read in the mask
            projected_mask_path = os.path.join(MASK_PATH, file.replace(".dcm", "_mask.jpg"))
            save_file_name = file.replace(".dcm", "_e10_segmented.jpg")
        else:
            continue
        if not os.path.exists(projected_mask_path):
            print(f"Mask for {file} does not exist")
            continue
        mask = cv2.imread(projected_mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # resize the mask to the same size as the image
        if binary_mask.shape != image.shape:
            binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # ensure mask is 2D
        if len(binary_mask.shape) == 3 and binary_mask.shape[2] == 3:
            binary_mask = binary_mask[:, :, 0]
        # apply the mask to the image
        segmented_image = cv2.bitwise_and(image, image, mask=binary_mask)
        # save the segmented image
        cv2.imwrite(os.path.join(SEGMENTED_PATH, save_file_name), segmented_image)