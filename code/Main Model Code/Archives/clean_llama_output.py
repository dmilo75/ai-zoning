import os
import pickle
import pandas as pd
import re
import numpy as np
import yaml

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

questions = pd.read_excel(os.path.join(config['raw_data'],'Questions.xlsx'))

yes_no_qs = questions[questions['Question Type'] == 'Binary']['ID'].to_list()
numerical = questions[questions['Question Type'] == 'Numerical']['ID'].to_list()
categorical = questions[questions['Question Type'] == 'Categorical']['ID'].to_list()
lot_size = questions[questions['Question Type'] == 'lot_size']['ID'].to_list()


#%%First, let's draw in all of the data

#Which model folder to draw in pickle files from
model = 'Full Run'

# Directory details    
dir_path = os.path.join(config['processed_data'],model)

# Get all pickle files from the directory and sort them by last modified time
all_files = [f for f in os.listdir(dir_path) if f.endswith(".pkl")]
all_files.sort(key=lambda x: os.path.getmtime(os.path.join(dir_path, x)), reverse=True)

# Load sorted pickle files from the directory
all_data = []
for file in all_files:
    with open(os.path.join(dir_path, file), 'rb') as f:
        all_data.extend(pickle.load(f))

# Extract relevant data and process it
processed_data = []
for entry in all_data:
    muni_name, _, muni_fips = entry['Muni'].partition('#')
    processed_data.append({
        'Answer': entry['Answer'],
        'Question': entry['Question'],
        'muni_name': muni_name,
        'muni_fips': muni_fips
    })

# Convert to dataframe
data = pd.DataFrame(processed_data)
print(data)
# Drop duplicates, keep the first (which will be the most recent due to sorting)
#This is to handle cases where we run the same muni-question twice in fragmented overlapping runs
data = data.drop_duplicates(subset=['Question', 'muni_fips'], keep='first')


#%%Now we need to clean the data, we'll borrow code from the compare results code 
df = data.copy()
def clean_yes_no(answer):
    match = re.search(r'Yes|No|I don\'t know', answer)
    if match:
        value = match.group()
        if value == "Yes":
            return 1
        elif value == "No":
            return 0
        else:
            return np.nan
    else:
        return np.nan

# def clean_numerical(answer):
#     match = re.search(r'\d+', answer)
#     if match:
#         return int(match.group())
#     return np.nan

def clean_numerical(answer):
    try:
        match = re.search(r'\d+', str(answer))
        if match:
            return int(match.group())
        return np.nan
    except (TypeError, ValueError):
        return np.nan

def clean_categorical(answer):
    
    # Check for "I don't know" first
    if answer == "I don't know":
        return np.nan
    
    # Check if the answer is in the list format
    if answer.startswith('[') and answer.endswith(']'):
        # Convert the string representation of the list into an actual list
        try:
            list_representation = eval(answer)
            if isinstance(list_representation, list):
                return list_representation
        except:
            pass

    # Splitting on ' and ', '&' and then on ','
    tokens = [token.strip() for part in answer.split(' and ') for subpart in part.split('&') for token in subpart.split(',')]
    
    # Remove empty tokens if any
    tokens = [token for token in tokens if token]
    return tokens
    
    # Otherwise return a null value
    return np.nan

# Applying the cleaning functions based on the columns
for idx, row in df.iterrows():
    if row['Question'] in yes_no_qs:
        df.at[idx, 'Answer'] = clean_yes_no(row['Answer'])
    elif row['Question'] in numerical:
        df.at[idx, 'Answer'] = clean_numerical(row['Answer'])
    elif row['Question'] in categorical:
        df.at[idx, 'Answer'] = clean_categorical(row['Answer'])
        
        
        
        
#%%Clean min lot size variables

#Make column for which type of quesiton 27 answer we have
df['Type'] = np.nan

# Extract values function remains the same as provided
# def extract_values(data):
#     results = []
#     string_values = re.findall(r':(.*?)(,|$)', data)
#     for val in string_values:
#         match = re.search(r'\d+', val[0])
#         if match:
#             results.append(int(match.group()))
#     return results

'''
Need to investigate why min lot size is showing up as zero so frequently 
'''

# Filtering data for question 27
df_q27 = df[df['Question'] == 27]

# Lists to store the new rows
new_rows = []

# Iterate over rows in df_q27 to compute the metrics and create new rows
for idx, row in df_q27.iterrows():
    try:
        values = [x['min_lot_size'] for x in row['Answer']]

        #Drop null values
        values = [x for x in values if x is not None]
    
        if values:  # If there are valid values extracted
            print(values)
            mean_val = np.mean(values)
            min_val = np.min(values)
            print(min_val)
        else:
            mean_val = np.nan
            min_val = np.nan

    except:
        mean_val = np.nan
        min_val = np.nan
         
    # Create new rows for 27mean and 27min
    new_rows.append({'Answer': mean_val, 'Question': 27, 'muni_name': row['muni_name'], 'muni_fips': row['muni_fips'], 'Type':'Mean'})
    new_rows.append({'Answer': min_val, 'Question': 27, 'muni_name': row['muni_name'], 'muni_fips': row['muni_fips'], 'Type':'Min'})

# Remove original rows for question 27 from the dataframe
df = df[~((df['Question'] == 27) & (df['Type'].isnull()))]

# Append the new rows to the dataframe
df = df.append(new_rows, ignore_index=True)

#%%Extra cleaning

df['State'] = df['muni_fips'].str.split('#').str[1]

df = df.rename(columns = {'muni_name':'Muni'})

df['fips'] = df['muni_fips'].str.split('#').str[0]

#%%Export as Excel
print(df.head())
try:
    df.to_excel(os.path.join(config['processed_data'], "Full Run"+".xlsx"))
    print("File successfully saved.")
except Exception as e:
    print("Error:", e)



