import csv

PATH1 = "glcm_for_clarius_patients_on_clarius_full.csv"
PATH2 = "glcm_for_clarius_patients_on_e22_full.csv"

with open(PATH1, 'r') as file1:
    reader1 = csv.reader(file1)
    data1 = list(reader1)

with open(PATH2, 'r') as file2:
    reader2 = csv.reader(file2)
    data2 = list(reader2)

    data2 = data2[1:]

data = data1 + data2
csv.writer(open("glcm_for_clarius_patients_full.csv", "w", newline="")).writerows(data)