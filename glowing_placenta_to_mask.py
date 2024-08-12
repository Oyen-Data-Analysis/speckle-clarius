import cv2
import numpy as np
import os

for glowing_placenta_path in os.listdir("Glowing_Placentas"):
    # Read in glowing placentas
    glowing_placenta_image = cv2.imread(os.path.join("Glowing_Placentas", glowing_placenta_path), cv2.IMREAD_GRAYSCALE)
    # Threshold the image
    glowing_placenta_image = np.where(glowing_placenta_image > 120, 255, 0).astype(np.uint8)
    # Save as mask
    cv2.imwrite(os.path.join("Output_Masks", glowing_placenta_path), glowing_placenta_image)
    # pring name of glowing placenta
    print(glowing_placenta_path)