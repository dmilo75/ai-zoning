import os
import pandas as pd
import time
from datetime import datetime, timedelta
import yaml

# Change to the script's directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Change to the parent directory
os.chdir('../../')

# Load the config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load the Sample Data
data = pd.read_excel(os.path.join(config['raw_data'], 'Sample Data.xlsx'))

# Directory containing the files
directory = '/vast/dm4766/context_building_refine/embed/Feb Embeddings/'

# Get the current time
now = datetime.now()

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        try:
            file_path = os.path.join(directory, filename)

            # Check if the file was modified within the last month
            if (now - datetime.fromtimestamp(os.path.getmtime(file_path))).days <= 30:
                # Extract CENSUS_ID_PID6 from the filename
                census_id = filename.split('#')[1]

                # Find the corresponding row in the Sample Data
                row = data[data['CENSUS_ID_PID6'] == int(census_id)]

                if not row.empty:
                    # Construct the new filename
                    new_filename = f"{row['UNIT_NAME'].values[0]}#{row['CENSUS_ID_PID6'].values[0]}#{row['State'].values[0]}.pkl"

                    # Rename the file
                    new_file_path = os.path.join(directory, new_filename)
                    os.rename(file_path, new_file_path)

                    print(f"Renamed {filename} to {new_filename}")
                else:
                    print(f"No matching row found for CENSUS_ID_PID6: {census_id}")
        except:
            print('Error')
            print(filename)

print("File renaming process completed.")