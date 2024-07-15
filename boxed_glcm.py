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
        for root, dirs, files in os.walk(folder_path):
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
    elif group == 'E22':
        for root, _, files in os.walk(folder_path):
            ID_MATCH = PATIENT_ID_REGEX.search(root)
            if not ID_MATCH:
                continue
            patient_id = ID_MATCH.group(0)
            if ids and patient_id not in ids:
                continue
            file_list = [f.rstrip('.mha') for f in files if f.endswith('.mha')]
            for file in file_list:
                scan_file = os.path.join(root, f"{file}.dcm")
                if not os.path.exists(scan_file):
                    print(f"Scan file not found for {root}/{file}.mha")
                    continue
                mask_file = os.path.join('Output_Segmented_Images', f"{file}_filled.mha")
                if os.path.exists(mask_file):
                    scan_file_list.append(scan_file) 
                    mask_file_list.append(mask_file)
                    patient_id_list.append(patient_id)
                else:
                    print(f"Filled mask not found for {root}/{file}.dcm")
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

def glcm_for_fixed_size_rectangle(dicom_array, binary_mask):
    """
    Extract normalized intensities from a fixed-size rectangle (50x139 or 139x50) within the segmented region.

    Parameters:
    - dicom_array: The original DICOM image array.
    - binary_mask: The binary mask array used to find the largest rectangle.

    Returns:
    - A flattened array of normalized intensities if a suitable rectangle is found; otherwise, an empty array.
    - A dictionary of GLCM features if a suitable rectangle is found; otherwise, an empty dictionary.
    """
    nrows, ncols = binary_mask.shape
    dimensions = []

    # Define dimensions
    dimension_options = {
        "0": [(80, 43), (43, 80)],
        # "0":[(50,100), (100,50),(40, 125), (125, 40)],
        # "0": [(100, 100), (50, 200), (200, 50)],
        "1": [(50, 139), (139, 50)],
        "2": [(150, 100), (100, 150)],
        "3": [(200, 125), (125, 200)],
        "4": [(240, 125), (125, 240)],
        "5": [(50, 139), (139, 50)],
        "6": [(150, 100), (100, 150), (75, 200), (200, 75), (120, 125), (125, 120)],
        "7": [(200, 125), (125, 200),(40, 625), (50, 500), (100, 250), (500, 50), (250, 100)],
        "8": [(240, 125), (125, 240), (75, 400), (400, 75), (100, 300), (300, 100), (200, 150), (150, 200), (60, 500), (500, 60)]
    }

    fixed_sizes = dimension_options.get("0")  # Default to the first option if invalid selection

    for size in fixed_sizes:
        print("Trying size:", size)
        for row in range(nrows - size[0] + 1):
            for col in range(ncols - size[1] + 1):
                if np.all(binary_mask[row:row + size[0], col:col + size[1]]):
                    dimensions = size
                    extracted_pixels = dicom_array[row:row + size[0], col:col + size[1]]
                    glcm_features = compute_glcm_features(extracted_pixels)
                    return glcm_features, dimensions
    return {}, (0, 0)  # Return empty dictionary and tuple if no suitable rectangle is found

def process_data(folder_path, group, ids=[]):
    dimensions = []
    features = []
    file_pairs = find_file_pairs(folder_path, group, ids=ids)
    for scan_file, mask_file, _ in zip(*file_pairs):
        print(f"Processing {mask_file}...")
        if group == 'clarius':
            plain_data = cv2.imread(scan_file, cv2.IMREAD_GRAYSCALE)
            mask_data = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if group == 'E22':
            # Your existing code to read DICOM and binary mask here
            plain_data = pydicom.dcmread(scan_file).pixel_array
            plain_data = cv2.cvtColor(plain_data, cv2.COLOR_BGR2GRAY)
            mask_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
        _, binary_mask = cv2.threshold(mask_data, 0, 255, cv2.THRESH_BINARY)

        # Ensure binary_mask is 2D before passing it
        if binary_mask.ndim > 2:
            binary_mask = binary_mask[:, :, 0]

        glcm_features, dimension = glcm_for_fixed_size_rectangle(plain_data, binary_mask)
        if not glcm_features:
            print(f"No suitable rectangle found for {mask_file}")
            continue
        features.append(glcm_features)
        dimensions.append(dimension)
    return features, dimensions, file_pairs[2]

E22_Control = 'Analysis\\Control_Patients_Segmented'
E22_FGR = 'Analysis\\FGR_Patients_Segmented'
Clarius_Control = 'Export for Globius\\Controlled'
Clarius_FGR = 'Export for Globius\\FGR'

ids = ['175-1', 
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

with open("e22_glcm_features.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Patient ID', 'Group', 'Homogeneity', 'Dissimilarity', 'Contrast', 'Correlation', 'Energy', 'Width', 'Height'])
    e22_control_features, e22_control_dimensions, e22_control_ids = process_data(E22_Control, 'E22')
    e22_fgr_features, e22_fgr_dimensions, e22_fgr_ids = process_data(E22_FGR, 'E22')

    for feature_dict, dimensions, id in zip(e22_control_features, e22_control_dimensions, e22_control_ids):
        writer.writerow([
            id,
            "Control",
            feature_dict['homogeneity'],
            feature_dict['dissimilarity'],
            feature_dict['contrast'],
            feature_dict['correlation'],
            feature_dict['energy'],
            dimensions[0],
            dimensions[1]
        ])

    for feature_dict, dimensions, id in zip(e22_fgr_features, e22_fgr_dimensions, e22_fgr_ids):
        writer.writerow([
            id,
            "FGR",
            feature_dict['homogeneity'],
            feature_dict['dissimilarity'],
            feature_dict['contrast'],
            feature_dict['correlation'],
            feature_dict['energy'],
            dimensions[0],
            dimensions[1]
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

# with open("clarius_glcm_features.csv", "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Patient ID', 'Group', 'Homogeneity', 'Dissimilarity', 'Contrast', 'Correlation', 'Energy', 'Width', 'Height'])

#     clarius_control_features, clarius_control_dimensions, clarius_control_ids = process_data(Clarius_Control, 'clarius', writer, ids=ids)
#     clarius_fgr_features, clarius_fgr_dimensions, clarius_fgr_ids = process_data(Clarius_FGR, 'clarius', writer, ids=ids)

#     for feature_dict, dimensions, id in zip(clarius_control_features, clarius_control_dimensions, clarius_control_ids):
#         writer.writerow([
#             id,
#             "Control",
#             feature_dict['homogeneity'],
#             feature_dict['dissimilarity'],
#             feature_dict['contrast'],
#             feature_dict['correlation'],
#             feature_dict['energy'],
#             dimensions[0],
#             dimensions[1]
#         ])
#     for feature_dict, dimensions, id in zip(clarius_fgr_features, clarius_fgr_dimensions, clarius_fgr_ids):
#         writer.writerow([
#             id,
#             "FGR",
#             feature_dict['homogeneity'],
#             feature_dict['dissimilarity'],
#             feature_dict['contrast'],
#             feature_dict['correlation'],
#             feature_dict['energy'],
#             dimensions[0],
#             dimensions[1]
#         ])
#     feature_names = clarius_control_features[0].keys()  # Getting keys from the first dictionary
#     #   code will break if there are no big enough placentas in control, at which point it's pointless to run the program anyways

#     for feature_name in feature_names:
#             control_values = [feat[feature_name] for feat in clarius_control_features]
#             fgr_values = [feat[feature_name] for feat in clarius_fgr_features]

#             # Creating a new figure for each feature
#             plt.figure(figsize=(8, 6))
#             bp = plt.boxplot([control_values, fgr_values], patch_artist=True, labels=['Control', 'FGR'])
            
#             # Set colors for the box plots
#             bp['boxes'][0].set_facecolor('#D7D7D9')
#             bp['boxes'][1].set_facecolor('#953017')
            
#             plt.title(f"Clarius_{feature_name}")
#             plt.ylabel('Value')
#             plt.show()