import SimpleITK as sitk

path = 'C:/Users/DRACula/Documents/Research/Oyen Lab/speckle-clarius/Export for Globius/Controlled\\Output_Masks\\13_FGR175-1_1_36.0_Plac_1_mask.jpg'
image = sitk.ReadImage(path)
image = sitk.GetArrayFromImage(image)
image = image[0] if image.ndim > 2 else image
print(image.sum())