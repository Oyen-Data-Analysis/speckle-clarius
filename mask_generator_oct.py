import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_and_plot_segmented_pixels(original_folder, segmented_folder, output_folder):
    all_intensities = []

    # Iterate over files in the original folder
    for file in os.listdir(original_folder):
        if file.endswith('.tif'):
            file_name, ext = os.path.splitext(file)

            # Construct the segmented image file path based on the naming scheme
            segmented_file_name1 = file_name.replace('raw', 'segmented') + ext 
            segmented_file_path1 = os.path.join(segmented_folder, segmented_file_name1)
            segmented_file_name2 = file_name.replace('raw', 'masked') + ext 
            segmented_file_path2 = os.path.join(segmented_folder, segmented_file_name2)

            if os.path.isfile(segmented_file_path1):
                segmented_file_path = segmented_file_path1
            elif os.path.isfile(segmented_file_path2):
                segmented_file_path = segmented_file_path2
            else:
                continue

            # Load original and segmented images
            original_image = Image.open(os.path.join(original_folder, file))
            segmented_image = Image.open(segmented_file_path)

            # Calculate and store the normalized intensities of the segmented pixels
            original_image_array = np.array(original_image)
            normalized_intensities = original_image_array[np.array(segmented_image) > 0].astype(np.float32) / 255.0
            all_intensities.append(normalized_intensities)

            # Save the images
            original_image.save(os.path.join(output_folder, f'{file_name}_original_image.png'))
            segmented_image.save(os.path.join(output_folder, f'{file_name}_segmented_image.png'))

            # Plot individual histogram for this file
            plt.figure()
            plt.hist(normalized_intensities, bins=50, color='blue', alpha=0.7)
            plt.title(f'Histogram of Normalized Intensities for {file_name}')
            plt.xlabel('Intensity')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(output_folder, f'{file_name}_intensity_histogram.png'))
            plt.close()

            segmented_pixels = original_image_array * (np.array(segmented_image) > 0).astype(np.uint8)
            f_transform = np.fft.fft2(segmented_pixels)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift))

            # Save the FFT image
            plt.figure()
            plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title(f'FFT for {file_name}')
            plt.colorbar()
            plt.savefig(os.path.join(output_folder, f'{file_name}_fft.png'))
            plt.close()

    # Plot the combined histogram of all normalized intensities
    plt.figure()
    for intensities in all_intensities:
        plt.hist(intensities, bins=50, color='blue', alpha=0.7, label='Individual')
    plt.title(f'Combined Histogram of Normalized Intensities')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'combined_intensity_histogram.png'))
    plt.close()

# Example usage
original_image_folder = '.\\FGR_Normal\\107_Unsegmented'
segmented_image_folder = '.\\FGR_Normal\\107_Segmented'
output_result_folder = '.\\FGR_Normal\\107_  Normal'
save_and_plot_segmented_pixels(original_image_folder, segmented_image_folder, output_result_folder)
