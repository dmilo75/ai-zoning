import os
import pandas as pd
import pickle
import yaml

os.chdir('../../')

# Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define the path where the pickled files are stored
export_path = os.path.join(config['processed_data'], 'Area Incorporated')

# Initialize an empty dictionary to store the results
results_dict = {}

# Loop through all pickled files in the export path
for file_name in os.listdir(export_path):
    if file_name.endswith('.pkl'):
        county_id = os.path.splitext(file_name)[0]  # Extract county id from file name
        file_path = os.path.join(export_path, file_name)
        # Load the pickled data
        with open(file_path, 'rb') as f:
            overlap = pickle.load(f)
        # Store the data in the dictionary
        results_dict[county_id] = overlap

# Convert the dictionary to a pandas Series
results_series = pd.Series(results_dict)

# Define the path to save the resultant Series
output_file_path = os.path.join(export_path, 'area_incorporated_series.pkl')

# Save the Series to a pickle file
with open(output_file_path, 'wb') as f:
    pickle.dump(results_series, f)

# Print a confirmation message
print(f"Resultant series has been saved to {output_file_path}")
