import pandas as pd

# Open the CSV file
df = pd.read_csv('data.csv')

# Retrieve the data in column 'DATA' where NAME='clarius-1st-control_5000', and process the string to get a list of numbers
filtered_data_str = df.loc[df['NAME'] == 'clarius-1st-control_5000', 'DATA'].str.lstrip('[').str.rstrip(']').str.split(', ')
# Convert the list of strings to a list of floats
filtered_data = filtered_data_str.apply(lambda x: [float(i) for i in x])

# Since filtered_data is a Series of lists, to calculate statistics, we need to flatten this into a single list
flat_list = [item for sublist in filtered_data for item in sublist]

# Convert the flat list to a pandas Series to use statistical methods
data_series = pd.Series(flat_list)

# Calculate the mean
mean_intensity = data_series.mean()

# Calculate the median
median_intensity = data_series.median()

# Calculate the 25th percentile
percentile_25 = data_series.quantile(0.25)

# Calculate the 75th percentile
percentile_75 = data_series.quantile(0.75)

# Calculate the standard deviation
std_dev = data_series.std()

# Calculate the maximum value
max_value = data_series.max()

# Calculate the minimum value
min_value = data_series.min()

print(f"Statistical Analysis of Intensity Data for: clarius-1st-control_5000")
print(f"Mean Intensity: {mean_intensity:.2f}")
print(f"Median Intensity: {median_intensity:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Minimum Intensity: {min_value:.2f}")
print(f"Maximum Intensity: {max_value:.2f}")
print(f"25th Percentile: {percentile_25:.2f}")
print(f"75th Percentile: {percentile_75:.2f}")