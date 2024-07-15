from collections import Counter
import os
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import tkinter as tk
from tkinter import *
from PIL import Image, ImageDraw
import seaborn as sns
import scipy.stats as stats
import csv
from tkinter import ttk
import ast


PROGRAM_OPTIONS = [
    "Process New Data",
    "Load Existing Data",
    "Compare Two Datasets",
    "Manage Data"
]

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
 
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        print("No folder selected, exiting program.")
        return None
    return folder_path

def find_file_pairs(folder_path):
    file_pairs = {}
    # Walk through all directories and subdirectories in the given folder path
    for root, dirs, files in os.walk(folder_path):
        if root.endswith("Unlabelled Clarius Images"):
            for file in files:
                if file.endswith(".jpeg"):
                    # Extract the base name without the extension
                    file_name, _ = os.path.splitext(file)
                    # Construct the relative path from the folder path to the file
                    relative_path = os.path.relpath(root, folder_path)
                    # Use the relative path as part of the key to avoid name clashes in different subdirectories
                    file_key = os.path.join(relative_path, file_name)
                    predicted_mask_path = os.path.join("Output_Masks", f"{file_name}_mask.jpg")
                    if os.path.exists(predicted_mask_path):
                        file_pairs[file_key] = {"plain": os.path.join(root, file), "masked": predicted_mask_path}
    return file_pairs

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

def extract_fixed_size_rectangle_intensities_and_features(dicom_array, binary_mask):
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
                    dimensions.append(size)
                    extracted_pixels = dicom_array[row:row + size[0], col:col + size[1]]
                    normalized_intensities = extracted_pixels.astype(np.float32) / np.max(dicom_array)
                    glcm_features = compute_glcm_features(extracted_pixels)
                    return normalized_intensities.ravel(), glcm_features, dimensions
    return [], {}, []

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

def process_data(folder_path):
    combined_intensities = []
    dimensions = []
    file_pairs = find_file_pairs(folder_path)
    for file_name, file_types in file_pairs.items():
        if file_types['masked'] == None:
            continue
        # Your existing code to read DICOM and binary mask here
        plain_data = cv2.imread(os.path.join(folder_path, file_types['plain']))
        print(f"Processing {file_types['masked']}...")
        mask_data = cv2.imread(os.path.join(file_types['masked']))
        binary_mask = mask_data > 0
        if plain_data.ndim == 3 and plain_data.shape[2] == 3:  # RGB
            plain_data_array = plain_data
        else:  # Grayscale
            plain_data_array = np.stack((plain_data,)*3, axis=-1)

        # Ensure binary_mask is 2D before passing it
        if binary_mask.ndim > 2:
            binary_mask = binary_mask[:, :, 0]

        intensities, glcm_features, dimension = extract_fixed_size_rectangle_intensities(plain_data_array[:, :, 0], binary_mask)

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
    def get_mode():
        selection = cb.current() + 1
        run_program(selection)
    root = tk.Tk()
    root.geometry("200x200")

    def run_program(choice):
        if choice == 1:
            folder_path = select_directory()
            if folder_path:
                combined_intensities, glcm_features, dimensions = process_data(folder_path)
                if messagebox.askyesno("Save Data", "Do you want to save this data for future use?", parent=root):
                    name = simpledialog.askstring("Input", "What do you want to save this dataset as?", parent=root)
                    if name:
                        save_data(name, folder_path, combined_intensities, dimensions) #needs work
                
                # Draw plot for data
                plot_dimensions(dimensions)
                print_statistics(combined_intensities, None, name, None)
                plot_intensity_histogram(combined_intensities, None, name, None)
                plot_glcm_features(glcm_features)


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


    variable = StringVar()
    cb = ttk.Combobox(root, textvariable=variable, values=PROGRAM_OPTIONS, state='readonly')
    cb.pack(fill='x', side='top', padx='5', pady='5')
    cb.current(0)
    submit_button = ttk.Button(root, text="Submit", command=lambda: get_mode())
    submit_button.pack(fill='x', side='top', padx='5', pady='5')
    root.mainloop()


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

def plot_glcm_features(features):
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

def generate_data(selections,  name1=None, name2=None, combined_intensities1=None, combined_intensities2=None,features=None):
    if selections['stats']:
        print_statistics(combined_intensities1, combined_intensities2, name1, name2)
    if selections['hist']:
        plot_intensity_histogram(combined_intensities1, combined_intensities2, name1, name2)
    if selections['glcm']:
        plot_glcm_features(features)
    # if selections['violin']:
    #     plot_violin(name1, combined_intensities1)
    # if selections['cdf']:
    #     plot_cdf(combined_intensities1, combined_intensities2, name1,name2)

def data_visualization_ui(name1=None, name2=None, combined_intensities1=None, combined_intensities2=None):
    def on_generate():
        local_combined_intensities1 = combined_intensities1
        local_combined_intensities2 = combined_intensities2

        selections = {
            'stats': stats_cb.instate(['selected']),
            'hist': hist_cb.instate(['selected']),
            'glcm': glcm_cb.instate(['selected']),
            # 'violin': violin_cb.instate(['selected']),
            # 'cdf': cdf_cb.instate(['selected'])
        }
        
        # Pass the locally defined numpy arrays to the generate_data function
        generate_data(selections, name1, name2, local_combined_intensities1, local_combined_intensities2)

        root.destroy()  # Close the UI window after gathering selections

    root = tk.Tk()
    root.title("Data Visualization Options")

    stats_cb = ttk.Checkbutton(root, text="Basic Intensity Statistics")
    hist_cb = ttk.Checkbutton(root, text="Intensity Histogram")
    glcm_cb = ttk.Checkbutton(root, text="GLCM Features")
    # violin_cb = ttk.Checkbutton(root, text="Violin Plot")
    # cdf_cb = ttk.Checkbutton(root, text="CDF Plot")

    # Adjusting checkbutton visibility as per the logic
    if combined_intensities1 is None and combined_intensities2 is None:
        msg = tk.Label(root, text="No data found for visualization.").pack(padx=10, pady=10)
        ttk.Button(root, text="Close", command=root.destroy).pack()
    if combined_intensities1 and combined_intensities2:
        stats_cb.grid(sticky='w')
        hist_cb.grid(sticky='w')
        # cdf_cb.grid(sticky='w')
        ttk.Button(root, text="Generate", command=on_generate).grid()
    elif combined_intensities1 is not None or combined_intensities2 is not None:
        stats_cb.grid(sticky='w')
        hist_cb.grid(sticky='w')
        # violin_cb.grid(sticky='w')
        # cdf_cb.grid(sticky='w')
        ttk.Button(root, text="Generate", command=on_generate).grid()    

    root.mainloop()

if __name__ == "__main__":
    name1, name2, combined_intensities1, combined_intensities2  = user_interface()
    data_visualization_ui(name1, name2, combined_intensities1, combined_intensities2)