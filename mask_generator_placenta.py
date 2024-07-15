import os
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image

def calculate_area_of_segmented_region(dicom_array, mha_array, dicom_data):
    # Get pixel spacing information from DICOM metadata
    pixel_spacing = dicom_data.PixelSpacing  # This should return a tuple (row spacing, column spacing)

    # Calculate the area of a single pixel in physical units (e.g., square millimeters)
    pixel_area = pixel_spacing[0] * pixel_spacing[1]  # Assuming square pixels

    # Create binary mask
    binary_mask = mha_array > 0

    # Count the number of segmented pixels
    num_segmented_pixels = np.sum(binary_mask)

    # Calculate the area of the segmented region in physical units
    area_in_physical_units = num_segmented_pixels * pixel_area

    return area_in_physical_units

def save_and_plot_segmented_pixels(dicom_array, mha_array, output_folder, file_name, intensity_list, dicom_data):
    # Convert to numpy arrays
    dicom_image = dicom_array
    mha_mask = mha_array

    # Create binary mask
    binary_mask = mha_mask > 0

    # Apply the binary mask to the DICOM image to extract the segmented pixels
    segmented_pixels = dicom_image * binary_mask

    # Save the images
    Image.fromarray(dicom_image).save(os.path.join(output_folder, f'{file_name}_original_dicom_image.png'))
    Image.fromarray((binary_mask * 255).astype(np.uint8)).save(os.path.join(output_folder, f'{file_name}_binary_mask.png'))
    Image.fromarray(segmented_pixels).save(os.path.join(output_folder, f'{file_name}_segmented_pixels.png'))

    # Calculate and store the normalized intensities of the segmented pixels
    normalized_intensities = segmented_pixels[binary_mask].astype(np.float32) / np.max(dicom_image)
    intensity_list.append(normalized_intensities)

    area = calculate_area_of_segmented_region(dicom_image, mha_mask, dicom_data)
    print(f'Area of segmented region for {file_name}: {area:.2f} square millimeters')

    # Plot individual histogram for this file
    plt.figure()
    plt.figure(figsize=[10,8])
    plt.hist(normalized_intensities, bins=50, color='grey', alpha=0.7) #for red switch color to 9f3017
    plt.title(f'Histogram of Normalized Intensities for {file_name}\nArea: {area:.2f} sq. mm')  # Include the area in the title
    plt.xlim(xmin=0.001, xmax = 1.0)
    plt.ylim(ymin=0, ymax = 4000)
    plt.tick_params(direction='in')
    plt.xlabel('Intensity',fontname="Arial",fontsize=20)
    plt.ylabel('Frequency',fontname="Arial",fontsize=20)
    plt.xticks(fontname="Arial",fontsize=20)
    plt.yticks(fontname="Arial",fontsize=20)
    plt.grid(False)
    plt.savefig(os.path.join(output_folder, f'{file_name}_intensity_histogram.png'))
    plt.close()
    return segmented_pixels, area

def save_fft_image(segmented_pixels, output_folder, file_name):
    # Perform FFT
    f_transform = np.fft.fft2(segmented_pixels)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    # Save the FFT image
    plt.figure()
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f'FFT of {file_name}')
    plt.colorbar()
    plt.savefig(os.path.join(output_folder, f'{file_name}_fft.png'))
    plt.close()

def print_dicom_metadata(dicom_data, file_name):
    print(f"Metadata for {file_name}:")
    for tag in dicom_data.dir():
        try:
            value = dicom_data.data_element(tag).value
            # Optionally filter out certain sensitive tags like PatientName
            if tag not in ['PatientName', 'PatientID']:  # Add more tags as needed
                print(f"{tag}: {value}")
        except:
            pass
    print("\n")

def process_folder(folder_path):
    file_pairs = {}
    all_intensities = []
    all_areas = []

    # Iterate over files in the folder
    for file in os.listdir(folder_path):
        if file.endswith('.dcm') or file.endswith('.mha'):
            file_name, ext = os.path.splitext(file)
            if file_name not in file_pairs:
                file_pairs[file_name] = {}
            file_pairs[file_name][ext] = file

    # Process each pair
    for file_name, file_types in file_pairs.items():
        if '.dcm' in file_types and '.mha' in file_types:
            dicom_file_path = os.path.join(folder_path, file_types['.dcm'])
            mha_file_path = os.path.join(folder_path, file_types['.mha'])

            dicom_data = pydicom.dcmread(dicom_file_path)
            #print_dicom_metadata(dicom_data, file_name)

            dicom_array = dicom_data.pixel_array
            if dicom_array.ndim == 3 and dicom_array.shape[-1] == 3:
                dicom_array = dicom_array[:, :, 0]

            mha_image = sitk.GetArrayFromImage(sitk.ReadImage(mha_file_path))
            mha_array = mha_image[0] if mha_image.ndim > 2 else mha_image

            assert dicom_array.shape == mha_array.shape, "The DICOM and MHA images must have the same dimensions."

            segmented_pixels, area = save_and_plot_segmented_pixels(dicom_array, mha_array, folder_path, file_name, all_intensities, dicom_data)
            save_fft_image(segmented_pixels, folder_path, file_name)
            
            all_areas.append(area)
    
    # Calculate the average area of all segmented regions
    average_area = np.mean(all_areas)
    print(f'Average Area of Segmented Regions: {average_area:.2f} square millimeters')

    # Plot the combined histogram of all normalized intensities
    plt.figure()
    plt.figure(figsize=[10,8])
    for intensities in all_intensities:
        plt.hist(intensities, bins=50, color='grey', alpha=0.7, label='Individual') #for red switch color to 9f3017
    plt.title(f'Combined Histogram of Normalized Intensities\nAverage Area: {average_area:.2f} sq. mm')
    plt.tick_params(direction='in')
    plt.xlim(xmin=0.001, xmax = 1.0)
    plt.ylim(ymin=0, )
    plt.xlabel('Intensity',fontname="Arial",fontsize=20)
    plt.ylabel('Frequency',fontname="Arial",fontsize=20)
    plt.xticks(fontname="Arial",fontsize=20)
    plt.yticks(fontname="Arial",fontsize=20)
    plt.grid(False)
    plt.savefig(os.path.join(folder_path, 'combined_intensity_histogram.png'))
    plt.close()

# Example usage
input_folder = './NORMAL/'
process_folder(input_folder)