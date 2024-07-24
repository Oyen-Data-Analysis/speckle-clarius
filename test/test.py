import cv2
import numpy as np
from PIL import Image, ImageFilter 

segmented_image = cv2.imread('segmented_image.jpeg')

# Convert to HSV color space
hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
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

flood_filled_image = erode.copy()
height, width = flood_filled_image.shape[:2]

# calculate bounds for height
height_arr = np.zeros((width, 2))

for pix_width in range (width):
    max_height = 0 
    min_height = height - 1
    for pix_height in range (height):
        if erode[pix_height, pix_width] == 255:
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
        if erode[pix_height, pix_width] == 255:
            if pix_width > max_width:
                max_width = pix_width
            if pix_width < min_width:
                min_width = pix_width
        width_arr[pix_height][0] = max_width
        width_arr[pix_height][1] = min_width

for pix_width in range(width):
    for pix_height in range(height):
        if (pix_height < height_arr[pix_width][0]) and (pix_height > height_arr[pix_width][1]) and (pix_width < width_arr[pix_height][0]) and (pix_width > width_arr[pix_height][1]):
            flood_filled_image[pix_height, pix_width] = 255
        else:
            flood_filled_image[pix_height, pix_width] = 0

cv2.imwrite('flood_filled_image.jpg', flood_filled_image)

smoothed_image = Image.open('flood_filled_image.jpg')
smoothed_image = smoothed_image.filter(ImageFilter.GaussianBlur(radius=20))
smoothed_image.save('smoothed_image.jpg')
smoothed_image.show()

smoothed_image = cv2.imread('smoothed_image.jpg')
# Convert smoothed_image to grayscale
smoothed_gray = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)
# Normalize the image
smoothed_gray = cv2.normalize(smoothed_gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
# Save the smoothed_gray image
cv2.imwrite('smoothed_gray.jpg', smoothed_gray)
# Display the smoothed_gray image
cv2.imshow('Smoothed Gray Image', smoothed_gray)

# Convert smoothed_gray to binary image
_, binary_image = cv2.threshold(smoothed_gray, 127, 255, cv2.THRESH_BINARY)

# Display the binary image
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()