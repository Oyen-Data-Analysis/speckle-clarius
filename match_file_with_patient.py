from config import *
import os
import re
import csv
import pandas as pd

GROUP_PATHS = {
    "e22": E22_PATH,
    "clarius": CLARIUS_PATH,
    "makerere": MAKERERE_PATH
}

ID_FORMAT = re.compile(r"\d{3}-\d{1}")
ITECH_FORMAT = re.compile(r"ITECH\d{1}-\d{4}")

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
        if group.lower() == 'makerere':
            id = ITECH_FORMAT.search(root)
            if not id:
                continue
            id = id.group()
            trimester = "/"
            for file in files:
                if file.endswith(".mhd") or file.endswith(".mha"):
                    mask_path = os.path.join(MASK_PATH, file.replace(".mhd", "_mask.jpg")) if file.endswith(".mhd") else os.path.join(MASK_PATH, file.replace(".mha", "_mask.jpg"))
                    if not os.path.exists(mask_path):
                        print(f"Mask not found for {group} patient {id}'s scan: {file}")
                    else:
                        mask_file = {
                            "ID": id,
                            "Trimester": trimester,
                            "Group": "Unknown",
                            "Machine": "e-10",
                            "File Type": "mask",
                            "Path": mask_path
                        }
                        if mask_file not in list:
                            list.append(mask_file)
                    segmentation_path = os.path.join(SEGMENTED_PATH, file.replace(".mhd", "_e10_segmented.jpg")) if file.endswith(".mhd") else os.path.join(SEGMENTED_PATH, file.replace(".mha", "_e10_segmented.jpg"))
                    if not os.path.exists(segmentation_path):
                        print(f"Segmentation not found for {group} patient {id}'s scan: {file}")
                    else:
                        segmentation_file = {
                            "ID": id,
                            "Trimester": trimester,
                            "Group": "Unknown",
                            "Machine": "e-10",
                            "File Type": "segmentation",
                            "Path": segmentation_path
                        }
                        if segmentation_file not in list:
                            list.append(segmentation_file)
                if file.endswith(".jpeg") and not file.endswith("(2).jpeg"):
                    mask_path = os.path.join(MASK_PATH, file.replace(".jpeg", " _mask.jpg"))
                    if not os.path.exists(mask_path):
                        print(f"Mask not found for {group} patient {id}'s scan: {file}")
                    else:
                        mask_file = {
                            "ID": id,
                            "Trimester": trimester,
                            "Group": "Unknown",
                            "Machine": "clarius",
                            "File Type": "mask",
                            "Path": mask_path
                        }
                        if mask_file not in list:
                            list.append(mask_file)
                    segmentation_path = os.path.join(SEGMENTED_PATH, file.replace(".jpeg", "_clarius_segmented.jpg"))
                    if not os.path.exists(segmentation_path):
                        print(f"Segmentation not found for {group} patient {id}'s scan: {file}")
                    else:
                        segmentation_file = {
                            "ID": id,
                            "Trimester": trimester,
                            "Group": "Unknown",
                            "Machine": "clarius",
                            "File Type": "segmentation",
                            "Path": segmentation_path
                        }
                        if segmentation_file not in list:
                            list.append(segmentation_file)
    return list

def storing_placentas(list):
    existing_df = pd.read_csv("File_Patient_Info.csv")
    new_df = pd.DataFrame(list)
     # Ensure consistent data types
    for column in existing_df.columns:
        if column in new_df.columns:
            new_df[column] = new_df[column].astype(existing_df[column].dtype)
    output_df = pd.concat([existing_df, new_df]).drop_duplicates().reset_index(drop=True)
    output_df.to_csv("File_Patient_Info.csv", index=False)

# clarius_results = catching_placentas("clarius")
# e22_results = catching_placentas("e22")
makerere_results = catching_placentas("makerere")
storing_placentas(makerere_results)
# clarius_results + e22_results + 