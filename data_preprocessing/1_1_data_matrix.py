import os
import ast
import numpy as np
import pandas as pd
from scipy.stats import iqr
import matplotlib.pyplot as plt
import pandas as pd

# Paths of all the datasets
labeled_data_path = os.getcwd()+'\\Label'
test_dataset_path = os.getcwd()+'\\Test'
evaluation_file = os.getcwd()+'\\F1_evaluation.csv'
test_label_evaluation_file = os.getcwd()+'\\F1_test_label_evaluation.csv'

# Create a new list with non-consecutive duplicates removed
def CollapseRecurringLabels(original_list):

    # Initialize the result list with the first element of the original list
    result_list = [original_list[0]]  

    for i in range(1, len(original_list)):
        if original_list[i] != original_list[i - 1]:
            result_list.append(original_list[i])

    return result_list

def compute_threshold(frequencies):
    data = frequencies
    median_value = np.median(data)
    iqr_value = iqr(data)
    k = 2

    threshold = median_value + k * iqr_value

    return threshold

def find_label_for_point(lat, lng, grid_info):
    for index, row in grid_info.iterrows():
        # Convert string representations to tuples
        min_lat, min_lng = ast.literal_eval(row['Min Latitude, Min Longitude'])
        max_lat, max_lng = ast.literal_eval(row['Max Latitude, Max Longitude'])

        if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
            return row['Grid Name']

    return None

# Step 1: Load Data and Collapse Recursive Data
frequency_dict = {}

for file_name in os.listdir(labeled_data_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(labeled_data_path, file_name)
        df = pd.read_csv(file_path)

        # Collapse recursive data in 'grid_label' column
        labels = df['grid_label'].tolist()
        unique_labels = CollapseRecurringLabels(labels)
        df_ = pd.DataFrame(unique_labels, columns=['grid_label'])

        # Step 2: Compute Frequency of Each Label
        label_counts = df_['grid_label'].value_counts().to_dict()

        # Step 3: Update Frequency Dictionary
        for label, count in label_counts.items():
            frequency_dict[label] = frequency_dict.get(label, 0) + count

# Create an empty DataFrame
evaluation_df = pd.DataFrame(columns=['cell_GeoLabel', 'lF', 'F1'])
evaluation_df.to_csv(evaluation_file, index=False)

# Create an empty DataFrame
testeval_df = pd.DataFrame(columns=['cell_GeoLabel', 'cell_Answer'])
testeval_df.to_csv(test_label_evaluation_file, index=False)

# Save Frequency Dictionary to CSV
frequency_df = pd.DataFrame(list(frequency_dict.items()), columns=['Label', 'Frequency'])
frequency_df.to_csv('label_frequencies.csv', index=False)

# Compute the threshold based on Frequency Dictionary
labels = frequency_df['Frequency'].tolist()
threshold = 3 #compute_threshold(labels)

# Load the grid information from the CSV file
grid_info_file = 'grid_information_with_paths.csv'
grid_info = pd.read_csv(grid_info_file)

latitudes = list()
longitudes = list()
anomalies = list()

# Read the test CSV file
for filename in os.listdir(test_dataset_path):
    if filename.endswith('.csv'):
        csv_file_path = os.path.join(test_dataset_path, filename)
        df = pd.read_csv(csv_file_path)
        previous_label = ""

        # Iterate over rows in a streaming way
        for index, row in df.iterrows():
            # Process the current row
            lat = row['lat']
            lng = row['lng']

            label = find_label_for_point(lat, lng, grid_info)

            # Check if the label exists in label_frequencies.csv
            if label == previous_label:
                pass
            else:
                previous_label = label

                latitudes.append(lat)
                longitudes.append(lng)

                if pd.notna(label):
                    if label in frequency_df['Label'].values:
                        # Get the corresponding frequency
                        frequency = frequency_df.loc[frequency_df['Label'] == label, 'Frequency'].values[0]

                        # Compare with the threshold
                        if frequency >= threshold:
                            Anomaly_type = "N"
                            # print(f"Label: {label} 정상 (Frequency: {frequency})")
                            anomalies.append(0)
                        else:
                            # print(f"Label: {label} 비정상 (Frequency: {frequency})")
                            Anomaly_type = "AN"
                            anomalies.append(1)
                    else:
                        # print(f"Label: {label} 비정상 (Frequency: {frequency})")
                        frequency = 0
                        Anomaly_type = "AN"
                        anomalies.append(1)
                else:
                    # print(f'Label: {label} 비정상 (Frequency: {frequency})')
                    frequency = 0
                    Anomaly_type = "AN"
                    anomalies.append(1)

                current_data = pd.DataFrame({'cell_GeoLabel': [label], 'F1': [Anomaly_type], 'lF': [frequency]})
                evaluation_df = pd.concat([evaluation_df, current_data], ignore_index=True)
                
                if frequency >= 3:
                    bit = 0
                else:
                    bit = 1
                
                current_data2 = pd.DataFrame({'cell_GeoLabel': [label], 'cell_Answer': [bit]})
                testeval_df = pd.concat([testeval_df, current_data2], ignore_index=True)

evaluation_df.to_csv(evaluation_file, index=False)
testeval_df.to_csv(test_label_evaluation_file, index=False)

Method = 'F1'
