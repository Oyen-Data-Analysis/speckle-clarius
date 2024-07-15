import os
import numpy as np
import SimpleITK as sitk
import pydicom
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import nibabel as nib

from skimage.transform import resize

def get_pixel_intensities_dcm(image_path, mask_path):
    dcm = pydicom.dcmread(image_path)
    img_array = dcm.pixel_array

    mask_sitk = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_sitk)

    all_pixel_data = []

    for img_slice, mask_slice in zip(img_array, mask_array):
        resized_mask_slice = resize(mask_slice, img_slice.shape, order=0, preserve_range=True, anti_aliasing=False)

        pixel_intensities = img_slice.flatten()
        mask = resized_mask_slice.flatten()
        masked_pixels = pixel_intensities[mask > 0]

        if len(masked_pixels) > 0:
            masked_pixels_normalized = (masked_pixels - np.mean(masked_pixels)) / np.std(masked_pixels)
            all_pixel_data.append(masked_pixels_normalized)
        else:
            print(f"No masked pixels found for slice. Image: {image_path}, Mask: {mask_path}")

    if len(all_pixel_data) > 0:
        return np.concatenate(all_pixel_data)
    else:
        return np.array([])


def determine_file_type(image_path, mask_path):
    if image_path.lower().endswith('.dcm'):
        return get_pixel_intensities_dcm(image_path)
    elif mask_path.lower().endswith('.mha'):
        return get_pixel_intensities_mha(image_path, mask_path)
    else:
        raise ValueError("Unsupported file format")


def get_pixel_intensities_mha(image_path, mask_path):
    img_sitk = sitk.ReadImage(image_path)
    mask_sitk = sitk.ReadImage(mask_path)

    img_array = sitk.GetArrayFromImage(img_sitk)
    mask_array = sitk.GetArrayFromImage(mask_sitk)

    all_pixel_data = []
    for img_slice, mask_slice in zip(img_array, mask_array):
        pixel_intensities = img_slice.flatten()
        mask = mask_slice.flatten()
        masked_pixels = pixel_intensities[mask > 0]  # Assuming mask values are either 0 or 255
        masked_pixels_normalized = (masked_pixels - np.mean(masked_pixels)) / np.std(masked_pixels)
        all_pixel_data.append(masked_pixels_normalized)

    return np.concatenate(all_pixel_data)


def get_patient_id(directory_path):
    return os.path.basename(os.path.normpath(directory_path))
    
def print_metadata(file_path, nii_file):
    print(f"Metadata for {file_path}:")
    header = nii_file.header
    for key, value in header.items():
        print(f"  {key}: {value}")

    print(f"  Dimensions: {header.get_data_shape()}")
    print(f"  Voxel sizes: {header.get_zooms()}")

def get_pixel_intensities_mnc(image_path, mask_path):
    img_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)

    # Extract and print metadata
    print_metadata(image_path, img_nii)
    print_metadata(mask_path, mask_nii)

    img_stack = img_nii.get_fdata()
    mask_stack = mask_nii.get_fdata()
    all_pixel_data = []
    for img, mask_img in zip(img_stack, mask_stack):
        pixel_intensities = img.flatten()

        mask = mask_img.flatten()
        masked_pixels = pixel_intensities[mask > 0]  # Assuming mask values are either 0 or 255
        masked_pixels_normalized = (masked_pixels - np.mean(masked_pixels)) / np.std(masked_pixels)
        all_pixel_data.append(masked_pixels_normalized)

    return np.concatenate(all_pixel_data)

def get_pixel_intensities_tiff(image_path, mask_path):
    all_pixel_data = []
    with Image.open(image_path) as img_stack:
        with Image.open(mask_path) as mask_stack:
            for img, mask_img in zip(ImageSequence.Iterator(img_stack), ImageSequence.Iterator(mask_stack)):
                img_gray = img.convert('L')
                pixel_intensities = np.array(img_gray)

                mask = np.array(mask_img)
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]

                masked_pixels = pixel_intensities[mask > 0]  # Assuming mask values are either 0 or 255
                masked_pixels_normalized = (masked_pixels - np.mean(masked_pixels)) / np.std(masked_pixels)
                all_pixel_data.append(masked_pixels_normalized.flatten())

    combined_pixel_data = np.concatenate(all_pixel_data)
    return combined_pixel_data

def plot_histogram(pixel_data, output_dir, file_name, patient_id, overlay_data=None, num_bins=50):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(pixel_data, bins=num_bins, color='blue', alpha=0.7, label='Individual TIFF')
    mean = np.mean(pixel_data)
    std_dev = np.std(pixel_data)
    skewness = ((pixel_data - mean) ** 3).mean() / (std_dev ** 3 if std_dev != 0 else 1)

    if overlay_data is not None:
        ax.hist(overlay_data, bins=num_bins, color='green', alpha=0.5, label='Combined TIFFs')
        overlay_mean = np.mean(overlay_data)
        overlay_std_dev = np.std(overlay_data)
        overlay_skewness = ((overlay_data - overlay_mean) ** 3).mean() / (overlay_std_dev ** 3 if overlay_std_dev != 0 else 1)

    ax.set_title(f"{patient_id}+ Combined with Overall Pixel Intensity for Patient ")
    ax.set_xlabel('Normalized Pixel Intensity')
    ax.set_ylabel('Frequency')

    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    stats_text = f"Mean: {mean:.2f}\nSD: {std_dev:.2f}\nSkewness: {skewness:.2f}"
    if overlay_data is not None:
        stats_text += f"\nOverlay Mean: {overlay_mean:.2f}\nOverlay SD: {overlay_std_dev:.2f}\nOverlay Skewness: {overlay_skewness:.2f}"

    plt.gcf().text(1.0, 0.5, stats_text, fontsize=10, verticalalignment='top', horizontalalignment='left')

    plt.subplots_adjust(right=0.8, bottom=0.2, top=0.85)

    output_path = os.path.join(output_dir, f"{file_name}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main(image_dir, mask_dir, output_dir):

    patient_id = get_patient_id(image_dir)

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.dcm')]
    mask_files = {os.path.splitext(f)[0]: os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith('.mha')}

    all_pixel_data = []
    for image_file in image_files:
        file_base_name = os.path.splitext(os.path.basename(image_file))[0]
        mask_file = mask_files.get(file_base_name)
        if mask_file:
            pixel_data = get_pixel_intensities_dcm(image_file, mask_file)
            all_pixel_data.append(pixel_data)

    combined_pixel_data = np.concatenate(all_pixel_data)
    plot_histogram(combined_pixel_data, output_dir, 'Combined_DCM_Files', patient_id)

    for i, (individual_data, image_file) in enumerate(zip(all_pixel_data, image_files), start=1):
        file_name = os.path.basename(image_file).split('.')[0]
        plot_title = f'Individual_DCM_File_{i}'
        plot_histogram(individual_data, output_dir, plot_title, patient_id, overlay_data=combined_pixel_data)




if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    image_dir = filedialog.askdirectory(title="Select Image Directory")
    mask_dir = filedialog.askdirectory(title="Select Mask Directory")
    output_dir = filedialog.askdirectory(title="Select Output Directory")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(image_dir, mask_dir, output_dir)