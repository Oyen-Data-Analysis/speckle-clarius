from PIL import Image
import numpy as np
import SimpleITK as sitk
import os

def mha_to_jpeg(mha_path, jpeg_path):    
    # Read the mha file
    image = sitk.ReadImage(mha_path)
    image = sitk.GetArrayFromImage(image)
    image = image[0] if image.ndim > 2 else image
    pil_image = Image.fromarray(np.uint8(image * 255.0 / image.max())) 
    pil_image.save(jpeg_path)

def dcm_to_jpeg(dcm_path, jpeg_path):
    # Read the dcm file
    image = sitk.ReadImage(dcm_path)
    image = sitk.GetArrayFromImage(image)
    image = image[0] if image.ndim > 2 else image
    pil_image = Image.fromarray(np.uint8(image * 255.0 / image.max())) 
    pil_image.save(jpeg_path)

id = ["FGR173-1", "FGR187-1"]

path = "C:\\Users\\DRACula\\Documents\\Research\\Oyen Lab\\speckle-clarius\\Analysis\\FGR_Patients_Segmented/131-1/Visit 2"
if not os.path.exists(path):
    print("Path does not exist: ", path)
for f in os.listdir(path):
    if f.endswith('.dcm'):
        mha_path = os.path.join(path, f)
        jpeg_path = os.path.join(path, f.replace('.dcm', '.jpeg'))
        dcm_to_jpeg(mha_path, jpeg_path)

# for i in id:
#     path = "C:\\Users\\DRACula\\Documents\\Research\\Oyen Lab\\speckle-clarius\\Export for Globius\\FGR\\{i}\\3rd Trim\\{i} Annotated Clarius Images\\Labelled Clarius Images".format(i=i)
#     if not os.path.exists(path):
#         print("Path does not exist: ", path)
#         continue
#     for f in os.listdir(path):
#         if f.endswith('.mha'):
#             mha_path = os.path.join(path, f)
#             jpeg_path = os.path.join(path, f.replace('.mha', '.jpeg'))
#             mha_to_jpeg(mha_path, jpeg_path)