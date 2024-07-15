# Assumptions:
# 1. Under pwd is a folder named "Output_Segmented_Images" that contains the segmented images. File names must include patient IDs. Recommend running glcm.py first to generate these images.
# 2. Under pwd is a folder named "Analysis" that contains the Control_Patients_Segmented and FGR_Patients_Segmented folders. These folders contain sub-folders with both patients' DICOM and MHA files.

import os
import numpy as np
import pydicom
import SimpleITK as sitk
import cv2
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import re

PATIENT_ID_REGEX = re.compile(r"\d{3}-1")

def get_list_of_patients(directory):
    patients = []
    for file in os.listdir(directory):
        patient_id = PATIENT_ID_REGEX.search(file)
        if patient_id != None:
            patients.append(patient_id.group())
    return patients

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

def apply_mask_and_compute_features(dicom_path, segmentation_path, output_dir):
    dicom_data = pydicom.dcmread(dicom_path)
    dicom_image_array = dicom_data.pixel_array

    if len(dicom_image_array.shape) == 3 and dicom_image_array.shape[-1] in [3, 4]:
        dicom_image_2d = cv2.cvtColor(dicom_image_array, cv2.COLOR_BGR2GRAY)
    else:
        dicom_image_2d = dicom_image_array

    dicom_image_2d = cv2.normalize(dicom_image_2d, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    segmentation_image = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path))
    segmentation_2d = segmentation_image[0] if segmentation_image.ndim > 2 else segmentation_image
    segmentation_2d = segmentation_2d.astype(np.uint8)
    _, binary_mask = cv2.threshold(segmentation_2d, 0, 255, cv2.THRESH_BINARY)

    if binary_mask.shape != dicom_image_2d.shape:
        binary_mask = cv2.resize(binary_mask, (dicom_image_2d.shape[1], dicom_image_2d.shape[0]), interpolation=cv2.INTER_NEAREST)

    if len(binary_mask.shape) == 3 and binary_mask.shape[2] == 3: # Basically, since the mask is grayscale, we can just take the first channel?
        binary_mask = binary_mask[:, :, 0]

    segmented_image = cv2.bitwise_and(dicom_image_2d, dicom_image_2d, mask=binary_mask)
    features = compute_glcm_features(segmented_image)

    # Saving the segmented image
    image_filename = os.path.basename(dicom_path).replace('.dcm', '_segmented.jpg')
    cv2.imwrite(os.path.join(output_dir, image_filename), segmented_image)

    return features, dicom_path

def process_directory_for_glcm(dicom_directory, list_of_patients, output_dir="Output_Segmented_Images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    features_list = []
    for subdir, dirs, files in os.walk(dicom_directory):
        patient_id = PATIENT_ID_REGEX.search(subdir)
        if patient_id is None:
            continue
        patient_id = patient_id.group()
        if patient_id not in list_of_patients:
            continue
        elif not files:
            print(f"No files found in {subdir}. Skipping this directory.")
            continue
        mha_files = [f for f in files if f.lower().endswith('.mha')]
        dcm_files = []
        for f in mha_files:
            dcm_path = f.replace('.mha', '.dcm') if os.path.exists(os.path.join(subdir, f.replace('.mha', '.dcm'))) else f.replace('.mha', '.0.dcm')
            dcm_files.append(dcm_path)
        
        for dicom_file, mha_file in zip(dcm_files, mha_files):
            dicom_path = os.path.join(subdir, dicom_file)
            mha_path = os.path.join(subdir, mha_file)
            features, path = apply_mask_and_compute_features(dicom_path, mha_path, output_dir)
            features_list.append((features, path))
    
    if not features_list:
        print("No images processed or no features extracted.")
        return None

    return features_list

def print_feature_statistics(features_list, group_name):
    aggregated_features = {feature_name: {'values': [], 'files': []} for feature_name, _ in features_list[0][0].items()}
    
    for features, path in features_list:
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
control = 'Analysis\\Control_Patients_Segmented'
fgr = 'Analysis\\FGR_Patients_Segmented'

list_of_patients = get_list_of_patients("Output_Segmented_Images")

control_features_list = process_directory_for_glcm(control, list_of_patients)
fgr_features_list = process_directory_for_glcm(fgr, list_of_patients)

if control_features_list is not None and fgr_features_list is not None:
    print_feature_statistics(control_features_list, "Control")
    print_feature_statistics(fgr_features_list, "FGR")

    # Extracting the features dictionaries for plotting.
    control_feature_dicts = [features for features, _ in control_features_list]
    fgr_feature_dicts = [features for features, _ in fgr_features_list]
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