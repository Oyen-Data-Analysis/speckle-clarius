import pydicom
import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re
import csv

PATIENT_ID_REGEX = re.compile(r"FGR\d{3}-1")

# def read_scan_image(path):
#     """Reads and returns a DICOM image as a numpy array."""
#     image_array = cv2.imread(path)
#     image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0  # Normalize
#     image_array = np.uint8(image_array)
#     return image_array

# def read_mask_image(path):
#     """Reads and returns an MHA image as a numpy array."""
#     image_array = cv2.imread(path)
#     image_array = np.squeeze(image_array)  # Remove singleton dimensions if any
#     return image_array


def largest_rectangle(binary_mask):
    nrows, ncols = binary_mask.shape
    max_area = (0, None)  # (area, top-left corner, dimensions)
    h = np.zeros(ncols, dtype=int)
    
    for row in range(nrows):
        for col in range(ncols):
            h[col] = h[col] + 1 if binary_mask[row, col] else 0
            
        for start_col in range(ncols):
            if h[start_col]:
                width = 1
                for k in range(start_col + 1, ncols):
                    if h[k] >= h[start_col]:
                        width += 1
                    else:
                        break
                area = width * h[start_col]
                if area > max_area[0]:
                    max_area = (area, (row - h[start_col] + 1, start_col), (h[start_col], width))
    
    _, (top_left_y, top_left_x), (height, width) = max_area
    return top_left_x, top_left_y, width, height

def find_largest_rectangle_in_segmentation(scan_path, segmentation_path):
    # Load scan and segmentation images
    scan_array = cv2.imread(scan_path)
    scan_array = cv2.normalize(scan_array, None, 0, 255, cv2.NORM_MINMAX)

    mask_array = cv2.imread(segmentation_path)

    # Ensure segmentation is binary
    mask_array = mask_array.astype(np.uint8)
    if len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    _, mask_array = cv2.threshold(mask_array, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(scan_array, contours, -1, (0,255,0), 2)  # Green outline

    # Find the largest rectangle inside the segmentation
    top_left_x, top_left_y, width, height = largest_rectangle(mask_array)

    # Draw the largest rectangle on the DICOM image
    cv2.rectangle(scan_array, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height), (0,0,255), 3)
    area = width * height  # Calculate the area based on the dimensions found
    return scan_array, area, width, height


def graph_largest_rectangles(group, root_directory=""):
    areas = []
    widths = []
    heights = []
    scans = []
    masks = []

    for subdir, dirs, files in os.walk(os.path.join(root_directory, "Export for Globius", group)):
        if subdir.endswith("Unlabelled Clarius Images"):
            base_names = [os.path.basename(f) for f in files if f.endswith(".jpeg")]
            print(base_names)
            scans = [os.path.join(subdir, f) for f in base_names]
            masks = [os.path.join("Output_Masks", f.replace(".jpeg", "_mask.jpg")) for f in base_names]
            for base_name, scan_file, mask_file in zip(base_names, scans, masks):
                if os.path.exists(mask_file):
                    outlined_image, area, width, height = find_largest_rectangle_in_segmentation(scan_file, mask_file)
                    areas.append(area)
                    widths.append(width)
                    heights.append(height)
                            
                    # Convert outlined_image to RGB if needed (OpenCV uses BGR by default)
                    if len(outlined_image.shape) == 2:  # If the image is grayscale
                        outlined_image = cv2.cvtColor(outlined_image, cv2.COLOR_GRAY2BGR)
                            
                    # Save the outlined image as PNG
                    outlined_path = os.path.join(root_directory, "Outlined_Images", base_name.replace(".jpeg", "_outlined.png"))
                    cv2.imwrite(outlined_path, outlined_image)
        continue
    
    return areas, widths, heights


# Example usage:
# Replace '/path/to/your/directory' with the actual directory path containing the DICOM and MHA files.
# graph_largest_rectangles('/path/to/your/directory')
control = graph_largest_rectangles("Controlled")
fgr = graph_largest_rectangles("FGR")
with open("boxes.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Area', 'Width', 'Height'])
    for item in zip(control[0], control[1], control[2]):
        writer.writerow(item)
    for item in zip(fgr[0], fgr[1], fgr[2]):
        writer.writerow(item)
print("Controlled:", control)
print("FGR:", fgr)

 # Plotting the histogram of areas with 20 bins
plt.figure(figsize=(10, 5))
plt.hist(control[0], bins=50, color='#D7D7D9', )
plt.hist(fgr[0], bins=50, color='#953017', alpha=.85)
plt.tight_layout()
plt.show()
