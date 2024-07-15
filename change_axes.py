import pandas as pd
import matplotlib.pyplot as plt

path = "Output_Segmented_Images\\Combined_features.csv"
feature_names = {"Homogeneity": [0.83, 0.99],
                 "Dissimilarity": [0.0, 1.0],
                #  "Contrast":,
                #  "Correlation":,
                 "Energy": [0.70, 1.00]}

df = pd.read_csv(path)

control = df[df['Group'] == 'Control']
fgr = df[df['Group'] == 'FGR']

for feature in feature_names.keys():
    control_values = control[feature].values
    fgr_values = fgr[feature].values

    plt.figure(figsize=(8, 6))
    bp = plt.boxplot([control_values, fgr_values], patch_artist=True, labels=['Control', 'FGR'])
    plt.ylim(feature_names[feature][0], feature_names[feature][1])
        
    # Set colors for the box plots
    bp['boxes'][0].set_facecolor('#D7D7D9')
    bp['boxes'][1].set_facecolor('#953017')
    
    plt.title(feature)
    plt.ylabel('Value')
    plt.show()