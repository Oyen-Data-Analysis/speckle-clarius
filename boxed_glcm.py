from matplotlib import pyplot as plt
import pandas as pd

import cv2
import os
from skimage.feature import graycomatrix, graycoprops

from config import *

BOX_DIMENSIONS = {
    # E-22, median - stddev, 34000px
    "0": [(170, 200), (200, 170), (100, 340), (340, 100), (136, 250), (250, 136)],
    # Clarius, median - stddev, 5000px
    "1": [(50, 100), (100, 50), (20, 250), (250, 20), (25, 200), (200, 25), (40, 125), (125, 40)],
}

def extract_size(segmentation, largest_rectangle_x, largest_rectangle_y, largest_w, largest_h, size_option):
    # Extract the largest rectangle
    for size in BOX_DIMENSIONS[size_option]:
        if size[0] < largest_w and size[1] < largest_h:
            return (segmentation[largest_rectangle_y:largest_rectangle_y + size[1], largest_rectangle_x:largest_rectangle_x + size[0]], size)
    return None

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

def process_data(meta_df, group_df, group_ids, machine):
    group_data = []
    for id in group_ids:
        # Find patient's group
        patient_row = meta_df[meta_df['ID'] == id].values[0]
        group = patient_row[2]
        # Open patient's segmentation
        patient_row = group_df[group_df['ID'] == id]
        segmentation_path = os.path.join(SEGMENTED_PATH, os.path.basename(patient_row['Mask Path'].values[0]).replace('_mask.jpg', '_segmented.jpg'))
        if not os.path.exists(segmentation_path):
            segmentation_path = os.path.join(SEGMENTED_PATH, os.path.basename(patient_row['Mask Path'].values[0]).replace(' _mask.jpg', '_clarius_segmented.jpg'))
        if not os.path.exists(segmentation_path):
            print(f"Could not find segmentation for patient {id}")
            continue
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
        # Extract fixed-size box
        if machine == 'e-22':
            size_option = '0'
        elif machine == 'clarius':
            size_option = '1'
        box_output = extract_size(segmentation, int(patient_row['Top Left X'].values[0]), int(patient_row['Top Left Y'].values[0]), int(patient_row['Width'].values[0]), int(patient_row['Height'].values[0]), size_option=size_option)
        if box_output is None:
            print(f"Could not extract box for patient {id}")
            continue
        boxed_array, size = box_output
        # Run GLCM on the fixed-size box
        glcm_features = compute_glcm_features(boxed_array)
        group_data.append((id, machine, group, glcm_features['homogeneity'], glcm_features['dissimilarity'], glcm_features['contrast'], glcm_features['correlation'], glcm_features['energy'], patient_row['Top Left X'].values[0], patient_row['Top Left Y'].values[0], size[0], size[1]))
        print(f"Processed patient {id}")
    return group_data

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

# Read in Patient Group Data
meta_df = pd.read_csv('File_Patient_Info.csv')

# process data for E22
e22_df = pd.read_csv('e22_largest_size.csv')
e22_ids = e22_df['ID'].unique()
e22_data = process_data(meta_df, e22_df, e22_ids, 'e-22')

# process data for Clarius
clarius_df = pd.read_csv('clarius_largest_size.csv')
clarius_ids = clarius_df['ID'].unique()
clarius_data = process_data(meta_df, clarius_df, clarius_ids, 'clarius')

# Write to CSV
output_df = pd.DataFrame(e22_data + clarius_data, columns=['Patient ID', 'Machine', 'Group', 'Homogeneity', 'Dissimilarity', 'Contrast', 'Correlation', 'Energy', 'Top Left X', 'Top Left Y', 'Width', 'Height'])
output_df.to_csv('boxed_glcm_features.csv', index=False)

# Plot the data
# output_df = pd.read_csv('boxed_glcm_features.csv')
for machine in ['e-22', 'clarius']:
    machine_data = output_df[output_df['Machine'] == machine]
    for feature in ['Homogeneity', 'Dissimilarity', 'Contrast', 'Correlation', 'Energy']:
        control_feature_list = machine_data[machine_data['Group'] == 'control'][feature].values
        fgr_feature_list = machine_data[machine_data['Group'] == 'fgr'][feature].values
        plt.figure(figsize=(8, 6))
        bp = plt.boxplot([control_feature_list, fgr_feature_list], patch_artist=True, labels=['Control', 'FGR'])
        
        # Set colors for the box plots
        bp['boxes'][0].set_facecolor('#D7D7D9')
        bp['boxes'][1].set_facecolor('#953017')
        
        plt.title(f"{machine} {feature} Boxed GLCM")
        plt.ylabel('Value')

        # Save the plot
        plt.savefig(os.path.join(GLCM_GRAPH_DIR, f"{machine}_{feature}_boxed_glcm.png"))

        # Show the plot
        plt.show()
        plt.close()
        