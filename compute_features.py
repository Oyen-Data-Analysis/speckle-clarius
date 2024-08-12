import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2
from config import *
import os
import csv
import re
import matplotlib.pyplot as plt
import pandas as pd

# Define Patient ID format
MARKERE_ID = re.compile(r"\d{4}_\d{1}")
OYEN_ID = re.compile(r"FGR\d{3}-\d{1}")

def compute_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0],
    }
    return features

# Read in segmentation file data
df = pd.read_csv("File_Patient_Info.csv")
segmentation_files = df[df['File Type'] == 'segmentation']
# processed_df = pd.DataFrame(columns=['ID', 'Trimester', 'Machine', 'Path', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'])
processed_data = []
for index, row in segmentation_files.iterrows():
    segmentation_path = row['Path']
    # Read in the segmented placenta image
    segmented_placenta_array = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    # Compute and print the GLCM features
    features = compute_glcm_features(segmented_placenta_array)
    print(features)
    # Write data to row in processed_df
    processed_data.append({
        'ID': row['ID'],
        'Trimester': row['Trimester'],
        'Group': row['Group'],
        'Machine': row['Machine'],
        'Path': row['Path'],
        'Contrast': features['contrast'],
        'Dissimilarity': features['dissimilarity'],
        'Homogeneity': features['homogeneity'],
        'Energy': features['energy'],
        'Correlation': features['correlation']
    })

processed_df = pd.DataFrame(processed_data)

# Save the processed data to a CSV file
processed_df.to_csv("features.csv", index=False)

# Plot the features on box plots
# Creating a new figure for each feature
for machine in ['clarius', 'e-22']:
    processed_df = pd.read_csv("features.csv")
    machine_df = processed_df[processed_df['Machine'] == machine]
    control_rows = machine_df[machine_df['Group'] == 'control']
    fgr_rows = machine_df[machine_df['Group'] == 'fgr']
    for feature_name in ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']:
            control_values = control_rows[feature_name].values
            fgr_values = fgr_rows[feature_name].values
            plt.figure(figsize=(8, 6))
            bp = plt.boxplot([control_values, fgr_values], patch_artist=True, labels=['Control', 'FGR'])
            
            # Set colors for the box plots
            bp['boxes'][0].set_facecolor('#D7D7D9')
            bp['boxes'][1].set_facecolor('#953017')
            
            plt.title(f"{machine}_{feature_name}")
            plt.ylabel('Value')
            plt.show()