import os
import numpy as np
import SimpleITK as sitk
import cv2
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter 
import csv

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

def dotted_outline_to_mask(segmented_image_path):
    segmented_image = cv2.imread(segmented_image_path)

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

    flood_filled_image = fill(erode)

    img = cv2.cvtColor(flood_filled_image, cv2.COLOR_BGR2RGB)
    smoothed_image = Image.fromarray(img)

    smoothed_image = smoothed_image.filter(ImageFilter.GaussianBlur(radius=20))
    smoothed_image = np.array(smoothed_image)
    # Convert RGB to BGR
    smoothed_image = smoothed_image[:, :, ::-1].copy()
    # smoothed_image.save('smoothed_image.jpg')
    # smoothed_image.show()

    # smoothed_image = cv2.imread('smoothed_image.jpg')
    # Convert smoothed_image to grayscale
    smoothed_gray = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)
    # Normalize the image
    smoothed_gray = cv2.normalize(smoothed_gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # Save the smoothed_gray image
    # cv2.imwrite('smoothed_gray.jpg', smoothed_gray)
    
    # Display the smoothed_gray image
    # cv2.imshow('Smoothed Gray Image', smoothed_gray)

    # Convert smoothed_gray to binary image
    _, binary_image = cv2.threshold(smoothed_gray, 127, 255, cv2.THRESH_BINARY)

    # Delete flood_filled_image.jpg
    # os.remove('flood_filled_image.jpg')

    return binary_image

def compute_glcm_features(image):
    image = image.astype('uint8')
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0],
    }
    return features

def apply_mask_and_compute_features(unlabelled_path, labelled_path, output_dir, masks_dir):
    unlabelled_image_array = cv2.imread(unlabelled_path)

    if len(unlabelled_image_array.shape) == 3 and unlabelled_image_array.shape[-1] in [3, 4]:
        unlabelled_image_2d = cv2.cvtColor(unlabelled_image_array, cv2.COLOR_BGR2GRAY)
    else:
        unlabelled_image_2d = unlabelled_image_array

    unlabelled_image_2d = cv2.normalize(unlabelled_image_2d, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    if labelled_path.endswith('.jpeg'):
        binary_mask = dotted_outline_to_mask(labelled_path)
    else:
        image = sitk.ReadImage(labelled_path)
        image = sitk.GetArrayFromImage(image)
        image = image[0] if image.ndim > 2 else image
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        binary_mask = fill(image)

    if binary_mask.shape != unlabelled_image_2d.shape:
        binary_mask = cv2.resize(binary_mask, (unlabelled_image_2d.shape[1], unlabelled_image_2d.shape[0]), interpolation=cv2.INTER_NEAREST)

    if len(binary_mask.shape) == 3 and binary_mask.shape[2] == 3: # Basically, since the mask is grayscale, we can just take the first channel?
        binary_mask = binary_mask[:, :, 0]
    
    # Save mask
    mask_filename = os.path.basename(unlabelled_path).replace('.jpeg', '_mask.jpg')
    cv2.imwrite(os.path.join(masks_dir, mask_filename), binary_mask)

    # Apply mask to the unlabelled image
    segmented_image = cv2.bitwise_and(unlabelled_image_2d, unlabelled_image_2d, mask=binary_mask)
    average_pixel_intensity = cv2.mean(segmented_image, mask=binary_mask)[0]
    features = compute_glcm_features(segmented_image)

    # Saving the segmented image
    image_filename = os.path.basename(unlabelled_path).replace('.jpeg', '_segmented.jpg')
    cv2.imwrite(os.path.join(output_dir, image_filename), segmented_image)

    return features, unlabelled_path, image_filename, average_pixel_intensity

def process_directory_for_glcm(scan_directory, group_name, output_dir="Output_Segmented_Images", masks_dir="Output_Masks"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)
    features_list = []
    for patient_id in os.listdir(scan_directory):
        print(f"Processing patient {patient_id}...")
        unlabelled_dir = os.path.join(scan_directory, patient_id, "3rd Trim\\"+patient_id+" Annotated Clarius Images\\Unlabelled Clarius Images")
        labelled_dir = os.path.join(scan_directory, patient_id, "3rd Trim\\"+patient_id+" Annotated Clarius Images\\Labelled Clarius Images")
        unlabelled_files = [f for f in os.listdir(unlabelled_dir) if f.lower().endswith('.jpeg')]
        labelled_files = [f for f in os.listdir(labelled_dir) if (f.lower().endswith('.jpeg') or f.lower().endswith('.mha'))]
        if len(unlabelled_files) != len(labelled_files):
            print(f"Number of unlabelled and labelled images do not match for patient {patient_id}. Skipping...")
            continue
        
        for unlabelled_file, labelled_file in zip(unlabelled_files, labelled_files):
            unlabelled_path = os.path.join(unlabelled_dir, unlabelled_file)
            labelled_path = os.path.join(labelled_dir, labelled_file)
            features, path, image_filename, avg_pixel_int = apply_mask_and_compute_features(unlabelled_path, labelled_path, output_dir, masks_dir)
            study_id = patient_id
            features_list.append((features, path, image_filename, study_id, avg_pixel_int))
    if not features_list:
        print("No images processed or no features extracted.")
        return None
    
    csv_filename = os.path.join(output_dir, f'{group_name}_features.csv')
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Study ID', 'Group', 'Image', 'Average Pixel Int', 'Homogeneity', 'Dissimilarity', 'Contrast', 'Correlation', 'Energy'])
        for features, path, image_filename, study_id, avg_pixel_int in features_list:
            row = [
                study_id,
                group_name,
                image_filename,
                avg_pixel_int,
                features['homogeneity'],
                features['dissimilarity'],
                features['contrast'],
                features['correlation'],
                features['energy']
            ]
            writer.writerow(row)

    return features_list

def print_feature_statistics(features_list, group_name):
    aggregated_features = {feature_name: {'values': [], 'files': []} for feature_name, _ in features_list[0][0].items()}
    
    for features, path, _, _, _ in features_list:
        for feature_name, value in features.items():
            aggregated_features[feature_name]['values'].append(value)
            aggregated_features[feature_name]['files'].append(path)
    
    print(f"\n{group_name} Group GLCM Feature Statistics:")
    for feature_name, data in aggregated_features.items():
        values = data['values']
        files = data['files']
        max_index = np.argmax(values)
        min_index = np.argmin(values)
        print(f"\n{feature_name}:")
        print(f"  Max value: {values[max_index]:.2f} (File: {files[max_index]})")
        print(f"  Min value: {values[min_index]:.2f} (File: {files[min_index]})")
# Assuming the other functions are as you posted them earlier and have not been altered

# Example usage
control = "Export for Globius\\Controlled"
fgr = "Export for Globius\\FGR"

control_features_list = process_directory_for_glcm(control, "Control")
fgr_features_list = process_directory_for_glcm(fgr, "FGR")

if control_features_list is not None and fgr_features_list is not None:
    print_feature_statistics(control_features_list, "Control")
    print_feature_statistics(fgr_features_list, "FGR")

    # Extracting the features dictionaries for plotting.
    control_feature_dicts = [features for features, _, _, _, _ in control_features_list]
    fgr_feature_dicts = [features for features, _, _, _, _ in fgr_features_list]
    feature_names = control_feature_dicts[0].keys()  # Getting keys from the first dictionary

    # Plotting each feature in a separate figure
    for feature_name in feature_names:
        control_values = [feat[feature_name] for feat in control_feature_dicts]
        fgr_values = [feat[feature_name] for feat in fgr_feature_dicts]

        # Creating a new figure for each feature
        plt.figure(figsize=(8, 6))
        bp = plt.boxplot([control_values, fgr_values], patch_artist=True, labels=['Control', 'FGR'])
        
        # Set colors for the box plots
        bp['boxes'][0].set_facecolor('#D7D7D9')
        bp['boxes'][1].set_facecolor('#953017')
        
        plt.title(feature_name)
        plt.ylabel('Value')
        plt.show()
else:
    print("No features to display.")