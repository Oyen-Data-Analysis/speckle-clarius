# calculate mean -std dev for clarius

import pandas as pd
import numpy as np

df = pd.read_csv("clarius_largest_size.csv")
areas = df["Area"].values
print("Mean:", np.mean(areas))
print("Std Dev:", np.std(areas))
print("Box", np.mean(areas) - np.std(areas))