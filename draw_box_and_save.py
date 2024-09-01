# This program draws the actual boxes used for GLCM and saves them
# Due to the imprtance of this step, all images and checked

import cv2
import numpy as np
import pandas as pd
import os
from config import *

# Load master data file
boxed_df = pd.read_csv("boxed_glcm_features.csv")
boxed_df = boxed_df[1:]

# Load Patient-File data file
e22_df = pd.read_csv("e22_largest_size.csv")
clarius_df = pd.read_csv("clarius_largest_size.csv")

for index, row in boxed_df.iterrows():
    # Get the row data
    id = row["Patient ID"]
    machine = row["Machine"]
    if machine == "e-22":
        image_path = os.path.join(SEGMENTED_PATH, os.path.basename(e22_df[e22_df["ID"] == id]["Mask Path"].values[0]).replace("mask", "segmented"))
    elif machine == "clarius":
        mask_path = clarius_df[clarius_df["ID"] == id]["Mask Path"].values[0]
        image_path = os.path.join(SEGMENTED_PATH, os.path.basename(mask_path).replace("mask", "segmented"))
        if not os.path.exists(image_path):
            image_path = os.path.join(SEGMENTED_PATH, os.path.basename(mask_path).replace(" _mask", "_clarius_segmented"))
    img = cv2.imread(image_path)
    x = int(row['Top Left X'])
    y = int(row['Top Left Y'])
    w = int(row['Width'])
    h = int(row['Height'])
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(GLCM_BOX_DIR, f"{id}.jpg"), img)
    print("Saved: ", row['Patient ID'])