import pandas as pd
import os

df = pd.read_csv("File_Patient_Info.csv")
csv_segmentation_files = df[df['File Type'] == 'mask']
actual_segmentation_files = os.listdir("Output_Masks")
mismatched_files = csv_segmentation_files[~csv_segmentation_files['Path'].apply(os.path.basename).isin(actual_segmentation_files)]
print(mismatched_files)