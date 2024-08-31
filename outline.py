import cv2
import numpy as np
import os
from config import *

def connect_dotted_outline_and_save(dotted_outline_path, output_dir="Outlined_Images"):
    dotted_outlined_image = cv2.imread(dotted_outline_path)

    # Convert to HSV color space
    hsv = cv2.cvtColor(dotted_outlined_image, cv2.COLOR_BGR2HSV)
    # Define the range for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # binary image
    th = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # morphological operations
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate = cv2.morphologyEx(th, cv2.MORPH_DILATE, k1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erode = cv2.morphologyEx(dilate, cv2.MORPH_ERODE, k2)

    # find contours
    cnts1 = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if len(cnts1) == 2 else cnts[1]
    cnts = cnts1[0]

    # For each contour, find the closest distance between their extreme points and join them
    for i in range(len(cnts)):
        min_dist = max(mask.shape[0], mask.shape[1])
        cl = []
        
        ci = cnts[i]
        ci_left = tuple(ci[ci[:, :, 0].argmin()][0])
        ci_right = tuple(ci[ci[:, :, 0].argmax()][0])
        ci_top = tuple(ci[ci[:, :, 1].argmin()][0])
        ci_bottom = tuple(ci[ci[:, :, 1].argmax()][0])
        ci_list = [ci_bottom, ci_left, ci_right, ci_top]
        
        for j in range(i + 1, len(cnts)):
            cj = cnts[j]
            cj_left = tuple(cj[cj[:, :, 0].argmin()][0])
            cj_right = tuple(cj[cj[:, :, 0].argmax()][0])
            cj_top = tuple(cj[cj[:, :, 1].argmin()][0])
            cj_bottom = tuple(cj[cj[:, :, 1].argmax()][0])
            cj_list = [cj_bottom, cj_left, cj_right, cj_top]
            
            for pt1 in ci_list:
                for pt2 in cj_list:
                    dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))     #dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
                    if dist < min_dist:
                        min_dist = dist             
                        cl = []
                        cl.append([pt1, pt2, min_dist])
        if len(cl) > 0:
            cv2.line(erode, cl[0][0], cl[0][1], 255, thickness = 2)
    save_path = os.path.join(output_dir, os.path.basename(dotted_outline_path).replace('(2).jpeg', '_outlined.jpg'))
    # Save the image
    cv2.imwrite(save_path, erode)
    return save_path

path = MAKERERE_PATH
if not os.path.exists(path):
    print("Path does not exist: ", path)
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('(2).jpeg'):
            dotted_outline_path = os.path.join(root, file)
            print(connect_dotted_outline_and_save(dotted_outline_path))