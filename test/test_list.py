import re
import os

path = "Output_Segmented_Images"
PATIENT_ID_REGEX = re.compile("FGR\d{3}-1")


patients = []
for file in os.listdir(path):
    patient_id = PATIENT_ID_REGEX.search(file)
    if patient_id != None:
        patients.append(patient_id.group())
print(patients)