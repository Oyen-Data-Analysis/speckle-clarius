import csv
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import re
import pydicom
from skimage.feature import graycomatrix, graycoprops
import SimpleITK as sitk

PATIENT_ID_REGEX = re.compile(r"\d{3}-\d")

def find_file_pairs(folder_path, group, ids=[]):
    scan_file_list = []
    mask_file_list = []
    patient_id_list = []
    # Walk through all directories and subdirectories in the given folder path
    if group == 'clarius':
        for root, _, files in os.walk(folder_path):
            ID_MATCH = PATIENT_ID_REGEX.search(root)
            if not ID_MATCH:
                continue
            if root.endswith("Unlabelled Clarius Images"):
                patient_id = PATIENT_ID_REGEX.search(root).group(0)
                if ids and patient_id not in ids:
                    continue
                patient_id = ID_MATCH.group(0)
                if ids and patient_id not in ids:
                    continue
                for file in files:
                    if file.endswith(".jpeg"):
                        # Extract the base name without the extension
                        file_name= file.rstrip('.jpeg')
                        predicted_mask_path = os.path.join("Output_Masks", f"{file_name}_mask.jpg")
                        if os.path.exists(predicted_mask_path):
                            scan_file_list.append(os.path.join(root, file))
                            mask_file_list.append(predicted_mask_path)
                            patient_id_list.append(patient_id)
                        else:
                            print(f"Mask not found for {file} from patient {patient_id}")
    elif group == 'E22':
        for root, _, files in os.walk(folder_path):
            ID_MATCH = PATIENT_ID_REGEX.search(root)
            if not ID_MATCH:
                continue
            patient_id = ID_MATCH.group(0)
            if ids and patient_id not in ids:
                continue
            dir_mask_file_list = [f for f in files if f.endswith('.mha')]
            dir_scan_file_list = [f.replace('.mha', '.dcm') for f in dir_mask_file_list]

            # Filter out non-existing scan files and their corresponding mask files
            existing_scan_indices = [index for index, scan_file in enumerate(dir_scan_file_list) if os.path.exists(os.path.join(root, scan_file))]
            dir_scan_file_list = [dir_scan_file_list[index] for index in existing_scan_indices]
            dir_mask_file_list = [dir_mask_file_list[index] for index in existing_scan_indices]

            # Extend the main lists with the filtered lists
            mask_file_list.extend([os.path.join(root, f) for f in dir_mask_file_list])
            scan_file_list.extend([os.path.join(root, f) for f in dir_scan_file_list])
            patient_id_list.extend([patient_id] * len(dir_scan_file_list))

    return scan_file_list, mask_file_list, patient_id_list

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

def fill(original_image_array):
    image_array = original_image_array.copy()
    height, width = image_array.shape[:2]

    # calculate bounds for height
    height_arr = np.zeros((width, 2))

    for pix_width in range (width):
        max_height = 0 
        min_height = height - 1
        for pix_height in range (height):
            if image_array[pix_height][pix_width] == 255:
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

def process_data(folder_path, group, ids=[]):
    features = []
    file_pairs = find_file_pairs(folder_path, group, ids=ids)
    for scan_file, mask_file, _ in zip(*file_pairs):
        print(f"Processing {mask_file}...")
        possible_segmented_path = ""
        if group == 'E22':
            possible_segmented_path = os.path.join("Output_Segmented_Images", os.path.basename(scan_file).replace('.dcm', '_segmented.jpg'))
        if group == 'clarius':
            possible_segmented_path = os.path.join("Output_Segmented_Images", os.path.basename(scan_file).replace('.jpeg', '_segmented.jpg'))
        if os.path.exists(possible_segmented_path):
            segmented_image = cv2.imread(possible_segmented_path, cv2.IMREAD_GRAYSCALE)
        else:
            if group == 'clarius':
                plain_data = cv2.imread(scan_file, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                mask_data = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            if group == 'E22':
                # Your existing code to read DICOM and binary mask here
                plain_data = pydicom.dcmread(scan_file).pixel_array.astype(np.uint8)
                plain_data = cv2.cvtColor(plain_data, cv2.COLOR_BGR2GRAY)
                if os.path.exists(os.path.join("Output_Segmented_Images", os.path.basename(mask_file).replace('.mha', '_filled.mha'))):
                    mask_data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join("Output_Segmented_Images", os.path.basename(mask_file).replace('.mha', '_filled.mha')))).astype(np.uint8)
                else:
                    mask_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_file)).astype(np.uint8).squeeze()
                    if mask_data.ndim > 2:
                        mask_data = mask_data[:, :, 0]
                    mask_data[mask_data > 0] = 255
                    mask_data = fill(mask_data)
                    sitk.WriteImage(sitk.GetImageFromArray(mask_data), os.path.join("Output_Segmented_Images", os.path.basename(mask_file).replace('.mha', '_filled.mha'))) 
            _, binary_mask = cv2.threshold(mask_data, 0, 255, cv2.THRESH_BINARY)

            # Ensure binary_mask is 2D before passing it
            if binary_mask.ndim > 2:
                binary_mask = binary_mask[:, :, 0]

            if binary_mask.shape != plain_data.shape:
                binary_mask = cv2.resize(binary_mask, (plain_data.shape[1], plain_data.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            segmented_image = cv2.bitwise_and(plain_data, plain_data, mask=binary_mask)
            cv2.imwrite(possible_segmented_path, segmented_image)
        glcm_features = compute_glcm_features(segmented_image)

        # Saving the segmented image
        image_filename = os.path.basename(scan_file).replace('.dcm', '_segmented.jpg')
        cv2.imwrite(os.path.join("Output_Segmented_Images", image_filename), segmented_image)

        if not glcm_features:
            print(f"No suitable rectangle found for {mask_file}")
            continue
        features.append(glcm_features)
    return features, file_pairs[2]

E22_Control = 'Analysis\\Control_Patients_Segmented'
E22_FGR = 'Analysis\\FGR_Patients_Segmented'
Clarius_Control = 'Export for Globius\\Controlled'
Clarius_FGR = 'Export for Globius\\FGR'

ids = [
       '175-1', 
       '176-1', 
       '177-1',
       '178-1',
       '179-1',
       '180-1',
       '182-1',
       '185-1',
       '186-1',
       '189-1',
       '192-1',
       '194-1',
       '173-1',
       '183-1', 
       '187-1', 
       '190-1'
]

with open("glcm_for_clarius_patients_on_e22_full.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Patient ID', 'Machine', 'Group', 'Homogeneity', 'Dissimilarity', 'Contrast', 'Correlation', 'Energy'])
    e22_control_features, e22_control_ids = process_data(E22_Control, 'E22', ids=ids)
    e22_fgr_features, e22_fgr_ids = process_data(E22_FGR, 'E22', ids=ids)

    for feature_dict, id in zip(e22_control_features, e22_control_ids):
        writer.writerow([
            id,
            "E-22",
            "Control",
            feature_dict['homogeneity'],
            feature_dict['dissimilarity'],
            feature_dict['contrast'],
            feature_dict['correlation'],
            feature_dict['energy']
        ])

    for feature_dict, id in zip(e22_fgr_features, e22_fgr_ids):
        writer.writerow([
            id,
            "E-22",
            "FGR",
            feature_dict['homogeneity'],
            feature_dict['dissimilarity'],
            feature_dict['contrast'],
            feature_dict['correlation'],
            feature_dict['energy']
        ])

    feature_names = e22_control_features[0].keys()  # Getting keys from the first dictionary

    for feature_name in feature_names:
            control_values = [feat[feature_name] for feat in e22_control_features]
            fgr_values = [feat[feature_name] for feat in e22_fgr_features]

            # Creating a new figure for each feature
            plt.figure(figsize=(8, 6))
            bp = plt.boxplot([control_values, fgr_values], patch_artist=True, labels=['Control', 'FGR'])
            
            # Set colors for the box plots
            bp['boxes'][0].set_facecolor('#D7D7D9')
            bp['boxes'][1].set_facecolor('#953017')
            
            plt.title(f"E22_{feature_name}")
            plt.ylabel('Value')
            plt.show()

with open("glcm_for_clarius_patients_on_clarius_full.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Patient ID', 'Machine', 'Group', 'Homogeneity', 'Dissimilarity', 'Contrast', 'Correlation', 'Energy'])

    clarius_control_features, clarius_control_ids = process_data(Clarius_Control, 'clarius', ids=ids)
    clarius_fgr_features, clarius_fgr_ids = process_data(Clarius_FGR, 'clarius', ids=ids)

    for feature_dict, id in zip(clarius_control_features, clarius_control_ids):
        writer.writerow([
            id,
            "Clarius",
            "Control",
            feature_dict['homogeneity'],
            feature_dict['dissimilarity'],
            feature_dict['contrast'],
            feature_dict['correlation'],
            feature_dict['energy']
        ])
    for feature_dict, id in zip(clarius_fgr_features, clarius_fgr_ids):
        writer.writerow([
            id,
            "Clarius",
            "FGR",
            feature_dict['homogeneity'],
            feature_dict['dissimilarity'],
            feature_dict['contrast'],
            feature_dict['correlation'],
            feature_dict['energy']
        ])
    feature_names = clarius_control_features[0].keys()  # Getting keys from the first dictionary
    #   code will break if there are no big enough placentas in control, at which point it's pointless to run the program anyways

    for feature_name in feature_names:
            control_values = [feat[feature_name] for feat in clarius_control_features]
            fgr_values = [feat[feature_name] for feat in clarius_fgr_features]

            # Creating a new figure for each feature
            plt.figure(figsize=(8, 6))
            bp = plt.boxplot([control_values, fgr_values], patch_artist=True, labels=['Control', 'FGR'])
            
            # Set colors for the box plots
            bp['boxes'][0].set_facecolor('#D7D7D9')
            bp['boxes'][1].set_facecolor('#953017')
            
            plt.title(f"Clarius_{feature_name}")
            plt.ylabel('Value')
            plt.show()