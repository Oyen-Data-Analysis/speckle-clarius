import os
import numpy as np
import cv2
import pandas as pd
from collections import defaultdict
import math

def load_patient_info(csv_path):
    return pd.read_csv(csv_path)

def calculate_area(mask):
    return np.sum(mask > 0)

def factorize_area(area):
    factors = []
    for i in range(1, int(math.sqrt(area)) + 1):
        if area % i == 0:
            factors.append((i, area // i))
    return factors

def find_largest_rectangle(mask, target_area):
    height, width = mask.shape
    factors = factorize_area(target_area)
    
    for h, w in factors:
        if h <= height and w <= width:
            for y in range(height - h + 1):
                for x in range(width - w + 1):
                    if np.all(mask[y:y+h, x:x+w]):
                        return True, (h, w)
    return False, None

def analyze_areas(df, base_path):
    areas = defaultdict(list)
    rectangles = defaultdict(lambda: defaultdict(list))

    total_masks = df[df['File Type'] == 'mask'].shape[0]
    processed_masks = 0

    for _, row in df.iterrows():
        if row['File Type'] == 'mask':
            patient_id = row['ID']
            machine = row['Machine']
            mask_path = os.path.join(base_path, row['Path'])
            
            print(f"Processing mask for patient {patient_id} ({machine})...")
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Failed to load mask for patient {patient_id}")
                continue
            
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            area = calculate_area(binary_mask)
            areas[machine].append((patient_id, area))
            
            processed_masks += 1
            print(f"Processed {processed_masks}/{total_masks} masks")

    for machine, machine_areas in areas.items():
        mean_area = np.mean([area for _, area in machine_areas])
        print(f"\n{machine} - Mean Area: {mean_area:.2f}")
        
        for percentage in range(10, 70, 10):
            target_area = int(mean_area * percentage / 100)
            count = 0
            print(f"\nChecking {percentage}% of mean area ({target_area} pixels):")
            for patient_id, area in machine_areas:
                found, dimensions = find_largest_rectangle(binary_mask, target_area)
                if found:
                    count += 1
                    rectangles[machine][percentage].append((patient_id, dimensions))
                    print(f"  Patient {patient_id}: Found rectangle {dimensions[0]}x{dimensions[1]}")
                else:
                    print(f"  Patient {patient_id}: No suitable rectangle found")
            print(f"{machine} - {percentage}% of mean area: {count} placentas")

    return areas, rectangles

def main():
    csv_path = r"C:\Users\Jared\Desktop\speckle-clarius\File_Patient_Info.csv"
    base_path = r"C:\Users\Jared\Desktop\speckle-clarius"
    
    print("Loading patient information...")
    df = load_patient_info(csv_path)
    print("Patient information loaded.")
    
    print("\nStarting area analysis...")
    areas, rectangles = analyze_areas(df, base_path)

    print("\n--- Summary ---")
    for machine, machine_areas in areas.items():
        print(f"\n{machine}:")
        areas_only = [area for _, area in machine_areas]
        print(f"Total placentas: {len(areas_only)}")
        print(f"Mean area: {np.mean(areas_only):.2f}")
        print(f"Median area: {np.median(areas_only):.2f}")
        print(f"Min area: {np.min(areas_only):.2f}")
        print(f"Max area: {np.max(areas_only):.2f}")
        
        for percentage, rectangles_found in rectangles[machine].items():
            print(f"\n{percentage}% of mean area:")
            print(f"Total placentas with suitable rectangle: {len(rectangles_found)}")
            for patient_id, dimensions in rectangles_found:
                print(f"  Patient {patient_id}: {dimensions[0]}x{dimensions[1]}")

if __name__ == "__main__":
    main()