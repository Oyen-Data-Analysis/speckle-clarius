from config import *
import os
import re
import csv

GROUP_PATHS = {
    "e22": E22_PATH,
    "clarius": CLARIUS_PATH,
    "makerere": MARKERE_PATH
}

ID_FORMAT = re.compile(r"\d{3}-\d{1}")

def catching_placentas(group):
    list = []
    group = group.lower()
    dir = GROUP_PATHS[group]
    for root, _, files in os.walk(dir):
        # Clarius mode
        if group.lower() == "clarius":
            if root.lower().endswith("labelled clarius images"):
                id = ID_FORMAT.search(root).group()
                for file in files:
                    mask_path = os.path.join(MASK_PATH, file.replace(".jpeg", "_mask.jpg"))
                    if not os.path.exists(mask_path):
                        print(f"Mask not found for {group} patient {id}'s scan: {file}")
                    else:
                        mask_file = {
                            "ID": id,
                            "Trimester": "3",
                            "Group": "control" if "controlled" in root.lower() else "fgr",
                            "Machine": "clarius",
                            "File Type": "mask",
                            "Path": mask_path
                        }
                        if mask_file not in list:
                            list.append(mask_file)
                    segmentation_path = os.path.join(SEGMENTED_PATH, file.replace(".jpeg", "_segmented.jpg"))
                    if not os.path.exists(segmentation_path):
                        print(f"Segmentation not found for {group} patient {id}'s scan: {file}")
                    else:
                        segmentation_file = {
                            "ID": id,
                            "Trimester": "3",
                            "Group": "control" if "controlled" in root.lower() else "fgr",
                            "Machine": "clarius",
                            "File Type": "segmentation",
                            "Path": segmentation_path
                        }
                        if segmentation_file not in list:
                            list.append(segmentation_file)
        # E-22 mode
        if group.lower() == "e22":
            # Find Patient ID from file name
            id = ID_FORMAT.search(root)
            if not id:
                continue
            id = id.group()
            for file in files:
                # Check if file is a mask
                if file.endswith(".mha"):
                    trimester = "3" if "Visit 2" in root else "2"
                    mask_path = os.path.join(MASK_PATH, file.replace(".mha", "_mask.jpg"))
                    if not os.path.exists(mask_path):
                        print(f"Mask not found for {group} patient {id}'s scan: {file}")
                    else:
                        mask_file = {
                            "ID": id,
                            "Trimester": trimester,
                            "Group": "control" if "control" in root.lower() else "fgr",
                            "Machine": "e-22",                            
                            "File Type": "mask",
                            "Path": mask_path
                        }
                        if mask_file not in list:
                            list.append(mask_file)
                    segmentation_path = os.path.join(SEGMENTED_PATH, file.replace(".mha", "_segmented.jpg"))
                    if not os.path.exists(segmentation_path):
                        print(f"Segmentation not found for {group} patient {id}'s scan: {file}")
                    else:
                        segmentation_file = {
                            "ID": id,
                            "Trimester": trimester,
                            "Group": "control" if "control" in root.lower() else "fgr",
                            "Machine": "e-22",
                            "File Type": "segmentation",
                            "Path": segmentation_path
                        }
                        if segmentation_file not in list:
                            list.append(segmentation_file)
    return list

def storing_placentas(list):
    with open("File_Patient_Info.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Trimester", "Group", "Machine", "File Type", "Path"])
        for item in list:
            writer.writerow(item.values())
        f.close()

clarius_results = catching_placentas("clarius")
e22_results = catching_placentas("e22")
storing_placentas(clarius_results + e22_results)