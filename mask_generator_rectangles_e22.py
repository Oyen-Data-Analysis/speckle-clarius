# Assumptions:
# 1. Under pwd is a folder named "Output_Segmented_Images" that contains the segmented images. File names must include patient IDs. Recommend running glcm.py first to generate these images.

from collections import Counter
import os
import time
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageDraw
import seaborn as sns
import scipy.stats as stats
import pydicom
import csv
import sys
from tkinter import ttk
import ast
import re

PATIENT_ID_REGEX = re.compile(r"\d{3}-1")

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

def get_list_of_patients(directory=os.path.join(os.getcwd(), "Output_Segmented_Images")):
    patients = []
    for file in os.listdir(directory):
        patient_id = PATIENT_ID_REGEX.search(file)
        if patient_id != None:
            patients.append(patient_id.group())
    return patients

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
    print(f'Largest rectangle: top-left corner: ({top_left_x}, {top_left_y}), dimensions: {width}x{height}')
    return top_left_x, top_left_y, width, height

def highlight_and_save_largest_rectangle(dicom_array, binary_mask, output_folder, file_name, x, y, width, height):
    # Convert DICOM array to an image
    if dicom_array.ndim == 3 and dicom_array.shape[2] == 3:  # If the DICOM image is RGB
        image = Image.fromarray(dicom_array)
    else:  # Assuming the DICOM image is grayscale
        image = Image.fromarray(dicom_array).convert("L")
    
    draw = ImageDraw.Draw(image)

    rect_start = (x, y)
    rect_end = (x + width, y + height)
    
    # Draw the rectangle in red
    draw.rectangle([rect_start, rect_end], outline="red", width=2)
    
    # Save the image with the rectangle as JPG
    output_path = os.path.join(output_folder, f"{file_name}_largest_rectangle.jpg")
    image.save(output_path)
    print(f"Saved highlighted image to {output_path}")

def calculate_area_of_segmented_region(dicom_array, mha_array, dicom_data):
    pixel_spacing = dicom_data.PixelSpacing
    pixel_area = pixel_spacing[0] * pixel_spacing[1]
    
    binary_mask = mha_array > 0
    num_segmented_pixels = np.sum(binary_mask)
    area_in_physical_units = num_segmented_pixels * pixel_area
    
    x, y, w, h = largest_rectangle(binary_mask)
    
    largest_rect_area = w * h * pixel_area
    
    return area_in_physical_units, (x, y, w, h, largest_rect_area)

def save_and_plot_segmented_pixels(dicom_array, mha_array, output_folder, file_name, intensity_list, dicom_data):
    dicom_image = dicom_array
    mha_mask = mha_array
    
    binary_mask = mha_mask > 0
    segmented_pixels = dicom_image * binary_mask
    
    Image.fromarray(dicom_image).save(os.path.join(output_folder, f'{file_name}_original_dicom_image.png'))
    Image.fromarray((binary_mask * 255).astype(np.uint8)).save(os.path.join(output_folder, f'{file_name}_binary_mask.png'))
    Image.fromarray(segmented_pixels).save(os.path.join(output_folder, f'{file_name}_segmented_pixels.png'))
    
    normalized_intensities = segmented_pixels[binary_mask].astype(np.float32) / np.max(dicom_image)
    intensity_list.append(normalized_intensities)
    
    area, (x, y, w, h, largest_rect_area) = calculate_area_of_segmented_region(dicom_image, mha_mask, dicom_data)
    print(f'Area of segmented region for {file_name}: {area:.2f} square millimeters')
    print(f'Largest rectangle area: {largest_rect_area:.2f} square millimeters')
    
    # Plot histogram for the largest rectangle
    plt.figure(figsize=[10, 8])
    plt.hist(normalized_intensities, bins=50, color='grey', alpha=0.7)
    plt.title(f'Histogram of Normalized Intensities for {file_name}\nArea: {area:.2f} sq. mm\nRectangle Area: {largest_rect_area:.2f} sq. mm')
    plt.xlabel('Intensity', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.grid(False)
    plt.savefig(os.path.join(output_folder, f'{file_name}_intensity_histogram.png'))
    plt.close()
    
    # Additional code for smaller rectangles will be included in the process_folder function
    return segmented_pixels, area

def plot_histograms_for_smaller_rectangles(segmented_pixels, binary_mask, dicom_image, output_folder, file_name, common_width, common_height):
    rectangle_factors = [1, 1/2, 1/4, 1/8]
    x, y, w, h = largest_rectangle(binary_mask)
    
    for factor in rectangle_factors:
        rect_w, rect_h = int(common_width * factor), int(common_height * factor)
        # Ensure the rectangle is extracted within the boundaries of the image
        end_x = min(x + rect_w, segmented_pixels.shape[1])
        end_y = min(y + rect_h, segmented_pixels.shape[0])
        extracted_pixels = segmented_pixels[y:end_y, x:end_x]
        normalized_intensities = extracted_pixels[extracted_pixels > 0].astype(np.float32) / np.max(dicom_image)
        
        plt.figure(figsize=[10, 8])
        plt.hist(normalized_intensities, bins=50, color='grey', alpha=0.7)
        plt.title(f'Histogram for {file_name} - Rectangle Size: {factor} of Largest')
        plt.xlabel('Intensity', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.grid(False)
        plt.savefig(os.path.join(output_folder, f'{file_name}_intensity_histogram_{factor}_of_largest.png'))
        plt.close()

def plot_largest_rectangle_sizes(largest_rectangles, output_folder):
    """
    Plot a histogram of the largest rectangle areas across images with 50 evenly spaced bins.

    Parameters:
    - largest_rectangles: A list of tuples, where each tuple contains the width and height of the largest rectangle for an image.
    - output_folder: The path to the directory where the plot will be saved.
    """
    # Calculate areas of the largest rectangles
    areas = [w * h for w, h in largest_rectangles if w > 0 and h > 0]  # Filter out any invalid areas

    # Check if areas list is empty
    if not areas:
        print("No areas to plot.")
        return

    # Find the min and max values for the area
    min_area = min(areas)
    max_area = max(areas)

    # Create the histogram with 50 bins between the min and max area
    plt.figure(figsize=[10, 8])
    plt.hist(areas, bins=50, range=(min_area, max_area), color='#D7D7D9', alpha=0.7)
    plt.title('Distribution of Largest Rectangle Areas Across Images')
    plt.xlabel('Area (pixels²)', fontsize=15)
    plt.ylabel('Number of Instances', fontsize=15)
    plt.grid(axis='y')

    # Save the plot
    plot_path = os.path.join(output_folder, 'largest_rectangle_areas_histogram.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Histogram saved to {plot_path}")
 
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        print("No folder selected, exiting program.")
        return None
    return folder_path

def find_file_pairs(folder_path, list_of_patient_ids = []):
    file_pairs = {}
    # Walk through all directories and subdirectories in the given folder path
    for root, dirs, files in os.walk(folder_path):
        if list_of_patient_ids != [] and PATIENT_ID_REGEX.search(root) is None:
            continue
        if list_of_patient_ids != [] and PATIENT_ID_REGEX.search(root).group() not in list_of_patient_ids:
            continue
        for file in files:
            if file.endswith('.dcm') or file.endswith('.mha'):
                # Extract the base name without the extension
                file_name, ext = os.path.splitext(file)
                # Construct the relative path from the folder path to the file
                relative_path = os.path.relpath(root, folder_path)
                # Use the relative path as part of the key to avoid name clashes in different subdirectories
                file_key = os.path.join(relative_path, file_name)
                if file_key not in file_pairs:
                    file_pairs[file_key] = {}
                # Store the full path to the file
                file_pairs[file_key][ext] = os.path.join(root, file)
    return file_pairs

def process_files(file_pairs, folder_path):
    largest_rectangles = []
    intensities_dict = {"1": [], "1/2": [], "1/4": [], "1/8": []}

    for file_name, file_types in file_pairs.items():
        try:
            if '.dcm' in file_types and '.mha' in file_types:
                process_file_pair(folder_path, file_name, file_types, largest_rectangles, intensities_dict, output_folder=folder_path)
        except Exception as e:
            print(f"Error handling file pair {file_name}: {e}")
            continue  # Continue to the next file pair

    return largest_rectangles, intensities_dict

def extract_fixed_size_rectangle_intensities(dicom_array, binary_mask):
    """
    Extract normalized intensities from a fixed-size rectangle (50x139 or 139x50) within the segmented region.

    Parameters:
    - dicom_array: The original DICOM image array.
    - binary_mask: The binary mask array used to find the largest rectangle.

    Returns:
    - A flattened array of normalized intensities if a suitable rectangle is found; otherwise, None.
    """
    nrows, ncols = binary_mask.shape
    dimensions = []

    # Define dimensions
    dimension_options = {
        "0": [(80, 43), (43, 80)],
        "1": [(50, 139), (139, 50)],
        "2": [(150, 100), (100, 150)],
        "3": [(200, 125), (125, 200)],
        "4": [(240, 125), (125, 240)],
        "5": [(50, 139), (139, 50)],
        "6": [(150, 100), (100, 150), (75, 200), (200, 75), (120, 125), (125, 120)],
        "7": [(200, 125), (125, 200),(40, 625), (50, 500), (100, 250), (500, 50), (250, 100), ],
        "8": [(240, 125), (125, 240), (75, 400), (400, 75), (100, 300), (300, 100), (200, 150), (150, 200), (60, 500), (500, 60)]
    }

    fixed_sizes = dimension_options.get("0")  # Default to the first option if invalid selection

    for size in fixed_sizes:
        print(f"Trying size: {size}...")
        start_time = time.time()
        for row in range(nrows - size[0] + 1):
            for col in range(ncols - size[1] + 1):
                if time.time() - start_time > 90:  # If it takes more than 5 seconds, skip to the next size
                    print("Taking too long. Skipping to next file...")
                    return [], []
                if np.all(binary_mask[row:row + size[0], col:col + size[1]]):
                    dimensions.append(size)
                    extracted_pixels = dicom_array[row:row + size[0], col:col + size[1]]
                    normalized_intensities = extracted_pixels.astype(np.float32) / np.max(dicom_array)
                    return normalized_intensities.ravel(), dimensions
    print("Oop... Bummer")
    return [],[]

def extract_fixed_size_rectangle_intensities_test(dicom_array, binary_mask, fixed_sizes):
    print("IN EXTRACTION")
    nrows, ncols = binary_mask.shape
    dimensions = []
    for size in fixed_sizes:
        for row in range(nrows - size[0] + 1):
            for col in range(ncols - size[1] + 1):
                if np.all(binary_mask[row:row + size[0], col:col + size[1]]):
                    print("FOUND RECTANGLE")
                    dimensions.append(size)
                    extracted_pixels = dicom_array[row:row + size[0], col:col + size[1]]
                    normalized_intensities = extracted_pixels.astype(np.float32) / np.max(dicom_array)
                    return normalized_intensities.ravel(), dimensions
    return [],[]

def launch_ui_to_extract_rectangle(dicom_array, binary_mask, results_container):
    # Define the mapping of UI selections to fixed_sizes values just like in the Tkinter version
    selection_map = {
        "6950px - SAME": [(50, 139), (139, 50)],
        "15000px - SAME": [(150, 100), (100, 150)],
        "25000px - SAME": [(200, 125), (125, 200)],
        "30000px - SAME": [(240, 125), (125, 240)],
        "6950px - ALL": [(50, 139), (139, 50)],
        "15000px - ALL": [(150, 100), (100, 150), (75, 200), (200, 75), (120, 125), (125, 120)],
        "25000px - ALL": [(200, 125), (125, 200),(40, 625), (50, 500), (100, 250), (500, 50), (250, 100), ],
        "30000px - ALL": [(240, 125), (125, 240), (75, 400), (400, 75), (100, 300), (300, 100), (200, 150), (150, 200), (60, 500), (500, 60)]
    }

    # Print the available selections to the console
    print("Select Dimension Type:")
    for idx, (text, value) in enumerate(selection_map.items(), start=1):
        print(f"{idx}. {text}")

    # User makes a selection via console input
    try:
        selection_index = int(input("Enter your choice (number): ")) - 1
        if selection_index >= 0 and selection_index < len(selection_map):
            selected_key = list(selection_map.keys())[selection_index]
            fixed_sizes = selection_map[selected_key]
            # Extract intensities and dimensions based on the selection
            intensities, dimensions = extract_fixed_size_rectangle_intensities(dicom_array, binary_mask, fixed_sizes)
            print(f"Selected: {selected_key}, Intensities: {len(intensities)}, Dimensions: {dimensions}")
            results_container['intensities'] = intensities
            results_container['dimensions'] = dimensions
        else:
            print("Invalid selection. Please run the program again with a valid choice.")
    except ValueError:
        print("Invalid input. Please enter a number.")

def get_fixed_size_rectangle_intensities(dicom_array, binary_mask):
    """
    Get normalized intensities within a fixed-size rectangle (50x139 or 139x50) from the binary mask.

    Parameters:
    - dicom_array: The original DICOM image array.
    - binary_mask: The binary mask array.
    
    Returns:
    - normalized_intensities: Normalized intensities within the rectangle, or None if no suitable rectangle is found.
    """
    nrows, ncols = binary_mask.shape
    found_rectangle = False
    #fixed_sizes = [(120,250), (250,120), (100,300), (300,100), (125,240), (240,125), (250, 200), (200,150)] #10,000 pixels
    #fixed_sizes = [(50, 139), (139, 50)] #6950 pixels
    fixed_sizes = [(200,125),(40, 625), (50,500),(100,250), (500,50), (250, 100), (125,200)] # 25000
    for size in fixed_sizes:
        for row in range(nrows - size[0] + 1):
            for col in range(ncols - size[1] + 1):
                if np.all(binary_mask[row:row + size[0], col:col + size[1]]):
                    extracted_pixels = dicom_array[row:row + size[0], col:col + size[1]]
                    found_rectangle = True
                    break
            if found_rectangle:
                break
        if found_rectangle:
            break

    if found_rectangle:
        normalized_intensities = extracted_pixels.astype(np.float32) / np.max(dicom_array)
        return normalized_intensities.ravel()
    else:
        return None

def process_file_pair(folder_path, file_name, file_types, largest_rectangles, intensities_dict, output_folder):
    dicom_file_path = os.path.join(folder_path, file_types['.dcm'])
    mha_file_path = os.path.join(folder_path, file_types['.mha'])

    dicom_data = pydicom.dcmread(dicom_file_path)
    mha_image = sitk.GetArrayFromImage(sitk.ReadImage(mha_file_path))
    mha_array = mha_image[0] if mha_image.ndim > 2 else mha_image

    binary_mask = mha_array > 0
    x, y, width, height = largest_rectangle(binary_mask)
    largest_rectangles.append((width, height))

    # Assuming dicom_data.pixel_array is the array you use for the DICOM image
    if dicom_data.pixel_array.ndim == 3 and dicom_data.pixel_array.shape[2] == 3:  # RGB
        dicom_image_array = dicom_data.pixel_array
    else:  # Grayscale
        dicom_image_array = np.stack((dicom_data.pixel_array,)*3, axis=-1)  # Convert grayscale to RGB for consistency

    # Highlight and save the largest rectangle on the image
    highlight_and_save_largest_rectangle(dicom_image_array, binary_mask, output_folder, file_name, x, y, width, height)

def save_data(name, folder_path, combined_intensities, dimensions):
    with open('data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, folder_path, combined_intensities, dimensions])

def load_data(name):
    maxInt = 2147483647
    csv.field_size_limit(maxInt)
    with open('data.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['NAME'] == name:
                # Assuming row["DATA"] is a string representation of a list
                try:
                    data_list = ast.literal_eval(row["DATA"])
                    if isinstance(data_list, list):
                        return data_list
                except (ValueError, SyntaxError):
                    # Handle the case where row["DATA"] is not a valid list string
                    print(f"Error converting DATA to list for NAME={name}")
                    return None
    return None 

def select_directory():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

def process_data(folder_path, tempMode=False):
    # if tempMode:
    #     with open ("temp.csv", "w") as file:
    #         file.write("NAME,DATA,WIDTH,HEIGHT,TOP_LEFT, TOP_RIGHT\n")
    combined_intensities = []
    dimensions = []
    list_of_patient_ids = get_list_of_patients()
    print(list_of_patient_ids)
    file_pairs = find_file_pairs(folder_path, list_of_patient_ids)
    for file_name, file_types in file_pairs.items():
        if '.dcm' in file_types and '.mha' in file_types:
            # Your existing code to read DICOM and binary mask here
            # print(folder_path)
            # print(os.path.join(folder_path,file_types['.dcm']))
            dicom_data = pydicom.dcmread(os.path.join(file_types['.dcm']))
            print("Processing {}...".format(file_name))
            mha_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(file_types['.mha']))).squeeze()
            mha_array = mha_image[:, :, 0] if mha_image.ndim > 2 else mha_image
            mha_array[mha_array == 1] = 255
            #make into binary before filling
            fill_path = os.path.join("Output_Segmented_Images", file_name[file_name.rfind('\\')+1:] + "_filled.mha")
            mha_array = fill(mha_array) if not os.path.exists(fill_path) else sitk.GetArrayFromImage(sitk.ReadImage(fill_path))
            sitk.WriteImage(sitk.GetImageFromArray(mha_array), fill_path)
            binary_mask = mha_array > 0
            if dicom_data.pixel_array.ndim == 3 and dicom_data.pixel_array.shape[2] == 3:  # RGB
                dicom_image_array = dicom_data.pixel_array
            else:  # Grayscale
                dicom_image_array = np.stack((dicom_data.pixel_array,)*3, axis=-1)

            # Ensure binary_mask is 2D before passing it
            if binary_mask.ndim > 2:
                binary_mask = binary_mask[:, :, 0]

            intensities, dimension = extract_fixed_size_rectangle_intensities(dicom_image_array[:, :, 0], binary_mask)
            combined_intensities.extend(intensities)
            dimensions.extend(dimension)
    return combined_intensities, dimensions

def process_data_test(folder_path):
    combined_intensities = []
    dimensions = []
    file_pairs = find_file_pairs(folder_path)
    for file_name, file_types in file_pairs.items():
        if '.dcm' in file_types and '.mha' in file_types:
            # Your existing code to read DICOM and binary mask here
            dicom_data = pydicom.dcmread(os.path.join(folder_path, file_types['.dcm']))
            mha_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder_path, file_types['.mha'])))
            mha_array = mha_image[0] if mha_image.ndim > 2 else mha_image
            binary_mask = mha_array > 0
            if dicom_data.pixel_array.ndim == 3 and dicom_data.pixel_array.shape[2] == 3:  # RGB
                dicom_image_array = dicom_data.pixel_array
            else:  # Grayscale
                dicom_image_array = np.stack((dicom_data.pixel_array,)*3, axis=-1)

            # Ensure binary_mask is 2D before passing it
            if binary_mask.ndim > 2:
                binary_mask = binary_mask[:, :, 0]

            results_container = {}  # Container to hold results
            launch_ui_to_extract_rectangle(dicom_image_array[:, :, 0], binary_mask, results_container)
            print("RESULTS:", results_container)
            if results_container:  # Check if results_container was populated
                intensities = results_container.get('intensities', [])
                dimension = results_container.get('dimensions', [])
                print(len(intensities), dimension)
                combined_intensities.extend(intensities)
                dimensions.extend(dimension)
    return combined_intensities, dimensions

def plot_dimensions(dimensions):
    dimensions_as_tuples = [tuple(dimension) for dimension in dimensions]

# Now you can use Counter
    dimension_counts = Counter(dimensions_as_tuples)

    # Separate the keys and values from the counter object
    dimension_types = list(dimension_counts.keys())
    occurrences = list(dimension_counts.values())
    # Convert dimension types to strings for easier handling by matplotlib
    dimension_labels = [f'{dim[0]}x{dim[1]}' for dim in dimension_types]

    plt.figure(figsize=(10, 6))
    plt.bar(dimension_labels, occurrences, color='skyblue')
    plt.xlabel('Types of Dimensions')
    plt.ylabel('Number of Occurrences')
    plt.title('Occurrence of Each Dimension Type')
    plt.xticks(rotation=45)  # Adjust rotation as needed
    plt.show()

def user_interface():
    root = tk.Tk()
    root.withdraw()
    choice = simpledialog.askinteger("Input", "Do you want to:\nEnter 1 to process new data\nEnter 2 to create graphs on preprocessed data\nEnter 3 to compare 2 Datasets", parent=root)
    name, name2, combined_intensities, combined_intensities2 = None, None, None, None
    if choice == 1:
        folder_path = select_directory()
        if folder_path:
            # tempMode = simpledialog.askyesno("Keep Temp Copy", "Do you want to save data on processing every file?", parent=root)
            combined_intensities, dimensions = process_data(folder_path)
            if messagebox.askyesno("Save Data", "Do you want to save this data for future use?", parent=root):
                name = simpledialog.askstring("Input", "What do you want to save this dataset as?", parent=root)
                if name:
                    save_data(name, folder_path, combined_intensities, dimensions)
                    plot_dimensions(dimensions)
            return name, name2, combined_intensities, combined_intensities2

            #generate_data(folder_path, combined_intensities)
    elif choice == 2:
        name = simpledialog.askstring("Input", "Enter the name of your dataset:", parent=root)
        if name:
            combined_intensities = load_data(name)

            if combined_intensities:
                return name1, name2, combined_intensities, combined_intensities
            else:
                messagebox.showerror("Error", "Dataset not found.", parent=root)
    elif choice == 3:
        name1 = simpledialog.askstring("Input", "Enter the name of dataset #1:", parent=root)
        name2 = simpledialog.askstring("Input", "Enter the name of dataset #2:", parent=root)
        if name1 and name2:
            combined_intensities1 = load_data(name1)
            combined_intensities2 = load_data(name2)
            if combined_intensities1 and combined_intensities2:
                return name1, name2, combined_intensities1, combined_intensities2
                #generate_data(folder_path2, combined_intensities2, name2)
            else:
                messagebox.showerror("Error", "Dataset not found.", parent=root)

def plot_cdf(combined_intensities1, combined_intensities2, name1,name2):
    combined_intensities1 = np.array(combined_intensities1)
    plt.figure(figsize=[10, 8])
    
    # Plot for combined_intensities1
    sorted_intensities1 = np.sort(combined_intensities1)
    yvals1 = np.arange(len(sorted_intensities1)) / float(len(sorted_intensities1) - 1)
    plt.plot(sorted_intensities1, yvals1, label=name1, color='#D7D7D9')  # You can change 'blue' to any color you prefer

    if combined_intensities2:
        combined_intensities2 = np.array(combined_intensities2)
        # Plot for combined_intensities2
        sorted_intensities2 = np.sort(combined_intensities2)
        yvals2 = np.arange(len(sorted_intensities2)) / float(len(sorted_intensities2) - 1)
        plt.plot(sorted_intensities2, yvals2, label=name2, color='#953017')  # Change 'red' to a different color if desired
    
    # plt.title(f'CDF of Combined Intensities for {name1}, {name2} Dataset(s)')
    # plt.xlabel('Intensity')
    # plt.ylabel('CDF')
    #plt.legend()  # Add this to display a legend
    plt.grid(False)
    plt.show()
    
def plot_violin(name, combined_intensities_array):
    combined_intensities_array = np.array(combined_intensities_array)
    sns.violinplot(data=combined_intensities_array)
    plt.title(f'Violin plot of Combined {name} Intensities')
    plt.xlabel('Intensity')
    plt.ylabel('Density')
    plt.show()

def plot_intensity_histogram(combined_intensities1, combined_intensities2, name1,name2):
    combined_intensities1 = np.array(combined_intensities1)
    plt.figure(figsize=[10, 8])
    plt.hist(combined_intensities1, bins=50, label=name1, color='#D7D7D9')

    if combined_intensities2:
        combined_intensities2 = np.array(combined_intensities2)
        # Plot for combined_intensities2
        plt.hist(combined_intensities2, bins=50, label=name2, color='#953017')
    # plt.title(f'Combined Intensity Histogram for {name1}, {name2} Dataset(s)')
    # plt.xlabel('Normalized Intensity', fontsize=20)
    # plt.ylabel('Frequency', fontsize=20)
    plt.grid(False)
    plt.show()

def print_statistics(combined_intensities1, combined_intensities2, name1,name2):
    def print_stats(combined_intensities_array, name):
        mean_intensity = np.mean(combined_intensities_array)
        median_intensity = np.median(combined_intensities_array)
        std_intensity = np.std(combined_intensities_array)
        min_intensity = np.min(combined_intensities_array)
        max_intensity = np.max(combined_intensities_array)
        percentile_25 = np.percentile(combined_intensities_array, 25)
        percentile_75 = np.percentile(combined_intensities_array, 75)

        print(f"Statistical Analysis of Intensity Data for: {name}")
        print(f"Mean Intensity: {mean_intensity:.2f}")
        print(f"Median Intensity: {median_intensity:.2f}")
        print(f"Standard Deviation: {std_intensity:.2f}")
        print(f"Minimum Intensity: {min_intensity:.2f}")
        print(f"Maximum Intensity: {max_intensity:.2f}")
        print(f"25th Percentile: {percentile_25:.2f}")
        print(f"75th Percentile: {percentile_75:.2f}")
    combined_intensities1 = np.array(combined_intensities1)
    print_stats(combined_intensities1, name1)
    if combined_intensities2 != None:
        combined_intensities2 = np.array(combined_intensities2)

        print_stats(combined_intensities2, name2)

def generate_data(selections,  name1=None, name2=None, combined_intensities1=None, combined_intensities2=None,):
    if selections['stats']:
        print_statistics(combined_intensities1, combined_intensities2, name1, name2)
    if selections['hist']:
        plot_intensity_histogram(combined_intensities1, combined_intensities2, name1, name2)
    if selections['violin']:
        plot_violin(name1, combined_intensities1)
    if selections['cdf']:
        plot_cdf(combined_intensities1, combined_intensities2, name1,name2)

def data_visualization_ui(name1=None, name2=None, combined_intensities1=None, combined_intensities2=None):
    def on_generate():
        local_combined_intensities1 = combined_intensities1
        local_combined_intensities2 = combined_intensities2

        selections = {
            'stats': stats_cb.instate(['selected']),
            'hist': hist_cb.instate(['selected']),
            'violin': violin_cb.instate(['selected']),
            'cdf': cdf_cb.instate(['selected'])
        }
        
        # Pass the locally defined numpy arrays to the generate_data function
        generate_data(selections, name1, name2, local_combined_intensities1, local_combined_intensities2)

        root.destroy()  # Close the UI window after gathering selections

    root = tk.Tk()
    root.title("Data Visualization Options")

    stats_cb = ttk.Checkbutton(root, text="Basic Statistics")
    hist_cb = ttk.Checkbutton(root, text="Intensity Histogram")
    violin_cb = ttk.Checkbutton(root, text="Violin Plot")
    cdf_cb = ttk.Checkbutton(root, text="CDF Plot")

    # Adjusting checkbutton visibility as per the logic
    if combined_intensities1 is not None or combined_intensities2 is not None:
        stats_cb.grid(sticky='w')
        hist_cb.grid(sticky='w')
        violin_cb.grid(sticky='w')
        cdf_cb.grid(sticky='w')
    elif combined_intensities1 and combined_intensities2:
        stats_cb.grid(sticky='w')
        hist_cb.grid(sticky='w')
        cdf_cb.grid(sticky='w')

    ttk.Button(root, text="Generate", command=on_generate).grid()

    root.mainloop()

if __name__ == "__main__":
    # folder_path = "Analysis\Control_Patients_Segmented"
    # combined_intensities, dimensions = process_data(folder_path)
    # print("COMBINED INTENSITIES:", combined_intensities)
    # print("DIMENSIONS:", dimensions)
    # plot_largest_rectangle_sizes(dimensions,folder_path)
    name1, name2, combined_intensities1, combined_intensities2  = user_interface()
    data_visualization_ui(name1, name2, combined_intensities1, combined_intensities2, )
