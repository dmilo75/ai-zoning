

import re
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import shutil
import yaml
import pickle

#Load config file
with open('../../config.yaml'', 'r') as file:
    config = yaml.safe_load(file)
    
# Map 'Source' to the correct folder name
source_mapping = {
    'Ordinance.com': 'ordDotCom',
    'American Legal Publishing': 'alp',
    'Municode': 'municode'
}

#%%Function to check whether text is about zoning

def is_zoning_text(text):
    text = text.lower()

    conditions = {
    "land use districts and allowed uses": "land use districts and allowed uses" in text,
    "r-\\d+ single-family regex": bool(re.search(r'r-\d+ single-family', text)),
    "r-\\d+ one-family residential regex": bool(re.search(r'r-\d+ one-family residential', text)),
    "r-\\d+ residential district regex": bool(re.search(r'r-\d+ residential district', text)),
    "r-\\d+,single family regex": bool(re.search(r'r-\d+,single family', text)),
    "d-\\d+ dwelling district regex": bool(re.search(r'd-\d+ dwelling district', text)),
    "r-\\d+ district": bool(re.search(r'r-\d+ district', text)),
    "R[\\w\\d] - One-Family regex": bool(re.search(r'r[\w\d] - one-family', text)),
    "residence [\\w\\d] district regex": bool(re.search(r'residence [\w\d] district', text)),
    "single-family residential district": 'single-family residential district' in text,
    "lot and yard requirements and standards": 'lot and yard requirements and standards' in text,
    "permitted residential uses": 'permitted residential uses' in text,
    "building height": 'building height regulation' in text,
    "minimum lot width": 'minimum lot width' in text,
    "frontage length": 'frontage length' in text,
    "establing zoning regs": 'hereby established zoning regulations' in text,
    'establing land use dev':' hereby adopts this land use development' in text,
    "enacted for the purpose of promoting the health": 'enacted for the purpose of promoting the health' in text,
    "is divided into the following districts": 'is divided into the following districts' in text,
    "the [any word] is hereby divided into zoning districts regex": bool(re.search(r'the \w+ is hereby divided into zoning districts', text.lower())),
    }
    matched_conditions = [condition for condition, matched in conditions.items() if matched]
    
    return len(matched_conditions) > 0

def get_file_path(row):
    """Generate the file path based on the row details."""
    
    base_dir = config['muni_text']
    
    source_folder = source_mapping[row['Source']]
    
    state_folder = row['State']
    file_name = row['UNIT_NAME'] + '#' + str(row['CENSUS_ID_PID6']) + '#' + state_folder + '.pkl'
    
    return os.path.join(base_dir, source_folder, state_folder, file_name)

# Check if the file exists and its size is at least 0.05 megabytes (50 kilobytes)
def file_exists_and_size(row):
    file_path = get_file_path(row)
    print(file_path)
    if os.path.exists(file_path):
        print("Exists")
        size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to megabytes
        print(size_mb)
        return size_mb >= 0.05
    return False

def concatenate_leaf_strings(d):
    """
    Concatenate all leaf string values from a hierarchical dictionary.
    
    :param d: Dictionary to traverse
    :return: Concatenated string of all leaf values
    """
    concatenated_string = ""
    for key, value in d.items():
        if isinstance(value, dict):
            # If value is a dictionary, recurse
            concatenated_string += concatenate_leaf_strings(value)
        else:
            # If value is a string (or other non-dict type), concatenate
            concatenated_string += str(value) + " "
    return concatenated_string

# Determine if the text is about zoning
def determine_zoning(row):
    print(row['Muni'])
    # Check if file exists and size is appropriate
    if not row['file_exists']:
        return False

    if row['Source'] == 'Ordinance.com':
        return True
    else:
        # Reading data from the pickle file
        with open(get_file_path(row), 'rb') as file:
            data = pickle.load(file)
        
        #Turn into string
        text = concatenate_leaf_strings(data)
        
        #Run through is_zoning_text
        whether_zoning = is_zoning_text(text)
        
        return whether_zoning


#%%Bring in matched excel and identify valid data

print("Drawing in matched munis")
#Draw in Excel that matched TOC with COG
matched = pd.read_excel(os.path.join(config['raw_data'],'matched_munis.xlsx'),index_col = 0)

# Apply the functions to the dataframe
print("Checking if file exists")
matched['file_exists'] = matched.apply(file_exists_and_size, axis=1)

print("Checking if about zoning")
matched['is_zoning'] = matched.apply(determine_zoning, axis=1)

#matched.to_excel(os.path.join(config['raw_data'],'matched_munis_filtered.xlsx'))

# Fill NaN values in 'is_zoning' column with False (where file doesn't exist or is too small)
matched['is_zoning'].fillna(False, inplace=True)

#%%Now select our sample

# Step 1: Filter out rows where 'is_zoning' is not True
filtered_df = matched[matched['is_zoning']]

# Step 2: Sort the DataFrame based on the 'Source' column with custom ordering
order_priority = {
    'Ordinance.com': 1,
    'Municode': 2,
    'American Legal Publishing': 3
}
filtered_df = filtered_df.sort_values(by='Source', key=lambda x: x.map(order_priority))

# Step 3: Drop duplicates based on the 'CENSUS_ID_PID6' column
final_df = filtered_df.drop_duplicates(subset='CENSUS_ID_PID6', keep='first')

#Step 4: Save sample data to use in charts/tables code
final_df.to_excel(os.path.join(config['raw_data'],'Sample Data.xlsx'))





