from PIL import Image
import numpy as np
import SimpleITK as sitk
import os
import cv2

def mha_to_jpeg(mha_path, jpeg_path):    
    # Read the mha file
    image = sitk.ReadImage(mha_path)
    image = sitk.GetArrayFromImage(image)
    image = image[0] if image.ndim > 2 else image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    pil_image = Image.fromarray(image.astype(np.uint8)) 
    pil_image.save(jpeg_path)

def dcm_to_jpeg(dcm_path, jpeg_path):
    # Read the dcm file
    image = sitk.ReadImage(dcm_path)
    image = sitk.GetArrayFromImage(image)
    image = image[0] if image.ndim > 2 else image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image.save(jpeg_path)

id = ["FGR173-1", "FGR187-1"]

path = "Analysis"
if not os.path.exists(path):
    print("Path does not exist: ", path)
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.mha'):
            mask_path = os.path.join(root, file)
            jpeg_path = os.path.join("Output_Masks", file.replace('.mha', '_mask.jpg'))
            mha_to_jpeg(mask_path, jpeg_path)

# for i in id:
#     path = "C:\\Users\\DRACula\\Documents\\Research\\Oyen Lab\\speckle-clarius\\Export for Globius\\FGR\\{i}\\3rd Trim\\{i} Annotated Clarius Images\\Labelled Clarius Images".format(i=i)
#     if not os.path.exists(path):
#         print("Path does not exist: ", path)
#         continue
#     for f in os.listdir(path):``
#         if f.endswith('.mha'):
#             mha_path = os.path.join(path, f)
#             jpeg_path = os.path.join(path, f.replace('.mha', '.jpeg'))
#             mha_to_jpeg(mha_path, jpeg_path)