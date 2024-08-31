# Given an image, and a bounding box, draw the bounding box on the image.
# Example use: Sanity check for largest rectangle algorithm

import cv2
import numpy as np
import pandas as pd

def draw_a_box(image, top_left_x, top_left_y, width, height):
    # Cast arguments to integers
    top_left_x = int(top_left_x)
    top_left_y = int(top_left_y)
    width = int(width)
    height = int(height)
    # Draw top line
    for x in range(top_left_x, top_left_x + width):
        image[top_left_y, x] = [0, 255, 0]
    # Draw bottom line
    for x in range(top_left_x, top_left_x + width):
        image[top_left_y + height, x] = [0, 255, 0]
    # Draw left line
    for y in range(top_left_y, top_left_y + height):
        image[y, top_left_x] = [0, 255, 0]
    # Draw right line
    for y in range(top_left_y, top_left_y + height):
        image[y, top_left_x + width] = [0, 255, 0]
    return image

# Example use for e22
df = pd.read_csv("e22_largest_size.csv")
# # Select every tenth row in df
# df = df[::10]
# For testing purposes, only use a small number of rows
df = df[:1]
# Read in image
for _, row in df.iterrows():
    image = cv2.imread(r'' + row["Mask Path"])
    image = draw_a_box(image, row["Top Left X"], row["Top Left Y"], row["Width"], row["Height"])
    print(row['ID'])
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example use for clarius
df = pd.read_csv("clarius_largest_size.csv")
# # Select every tenth row in df
# df = df[::10]
# For testing purposes, only use a small number of rows
df = df[:1]
# Read in image
for _, row in df.iterrows():
    image = cv2.imread(r'' + row["Mask Path"])
    image = draw_a_box(image, row["Top Left X"], row["Top Left Y"], row["Width"], row["Height"])
    print(row['ID'])
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()