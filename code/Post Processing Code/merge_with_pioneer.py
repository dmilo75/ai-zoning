#Load config file
import os
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

import numpy as np
import pickle
import pandas as pd
yes_no_qs = [4,5,6,8,9,11,13,14,17,20,21]
numerical = [2,22]
categorical = [7,12,15]
lot_size = [27]


#Used to map naming conventions
sample_map = pd.read_excel(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Charts and Tables\Data\Sample Data.xlsx")

def calc_min_lot_size(item):
    min_lot_size = item['min_lot_size']

    if 'unit' in item:
        unit = item['unit']

        if unit == 'Acres':
            # Convert acres to square feet
            min_lot_size = min_lot_size * 43560

    return min_lot_size

def calc_lot_size_metric(column,type):

    # Ensure column is iterable; if not, return NaN
    if not isinstance(column, list):
        return np.nan

    # Initialize min_lot_sizes list
    min_lot_sizes = []

    # Try to extract numeric min_lot_sizes and handle errors
    try:
        for item in column:
            if isinstance(item, dict) and 'min_lot_size' in item:
                min_lot_size = calc_min_lot_size(item)
                if isinstance(min_lot_size, (int, float)) and not np.isnan(min_lot_size):
                    min_lot_sizes.append(min_lot_size)
    except Exception as e:
        return np.nan

    # Calculate average and minimum if possible
    if min_lot_sizes:
        if type == 'mean':
            return np.mean(min_lot_sizes)
        elif type == 'min':
            return np.min(min_lot_sizes)
    else:
        return np.nan

def process_min_lot_size(df):

    min_lot_size_qs = ['27','28']

    for question in min_lot_size_qs:
        # Get the rows for the question
        question_rows = df[df['Question'] == question]

        #Append the 'Answer' as a string to the 'Explanation' column after a newline
        question_rows['Evidence'] = question_rows['Answer'].astype(str)

        #Copy question_rows twice, once for min and once for mean
        question_rows_min = question_rows.copy()
        question_rows_mean = question_rows.copy()

        #Change the question number in both to question + 'Min' or question + 'Mean'
        question_rows_min['Question'] = question + 'Min'
        question_rows_mean['Question'] = question + 'Mean'

        #Now we need to calculate the min and mean for each muni
        question_rows_min['Answer'] = question_rows_min['Answer'].apply(lambda x: calc_lot_size_metric(x,'min'))
        question_rows_mean['Answer'] = question_rows_mean['Answer'].apply(lambda x: calc_lot_size_metric(x,'mean'))

        #Now drop the original question rows and append the new ones
        df = df[df['Question'] != question]
        df = pd.concat([df, question_rows_min, question_rows_mean], ignore_index=True)

    return df

#Adjust naming convention so we can merge
def adjust_muni_name(row):
    if '#' in row['Muni']:
        return row['Muni'].split('#')[0]
    else:
        lowercase_muni = row['Muni'].lower()
        matching_row = sample_map[(sample_map['State'] == 'ma') & (sample_map['Muni'] == lowercase_muni)]
        
        # If a matching row exists, replace with 'UNIT_NAME'. Otherwise, use the original 'Muni'.
        return matching_row['UNIT_NAME'].iloc[0] if not matching_row.empty else row['Muni']

def load_data(file_path):
    all_data = []  # Initialize an empty list to store all data

    if os.path.isdir(file_path):
        # Get list of .pkl files in the directory
        files = [f for f in os.listdir(file_path) if f.endswith(".pkl")]

        for file in files:
            file_full_path = os.path.join(file_path, file)
            with open(file_full_path, 'rb') as f:
                data = pickle.load(f)
                # Append the data from each file to the all_data list
                all_data.extend(data)
    else:
        # If the file_path is not a directory, assume it's a single file
        with open(file_path, 'rb') as file:
            all_data = pickle.load(file)

    return all_data

def load_pioneer():

    # Load the .pkl file into a dictionary
    loaded_data = load_data(os.path.join(config['raw_data'],'pioneer.pkl'))

    # Get the questions from the keys of the dictionary
    questions = loaded_data.keys()

    # Convert the dictionary to a DataFrame
    df_list = []
    for idx, (question, answers) in enumerate(loaded_data.items(), start=1):
        temp_df = pd.DataFrame(answers)
        temp_df['Question'] = str(idx)
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)

    # Specific renaming for Pioneer data
    df['Muni'] = df['Muni'].replace({'Manchester-by-the-Sea': 'Manchester'})

    #Make muni name lowercase
    df['Muni'] = df['Muni'].str.lower()

    gis_data = load_gis()
    pioneer_combined = pd.concat([df, gis_data], axis=0)

    # Dictionary to convert GIS naming format to census naming format
    rename_dic = {
        'braintree': 'braintree town',
        'methuen': 'methuen town',
        'winthrop': 'winthrop town',
        'manchester': 'manchester-by-the-sea',
        'watertown': 'watertown town',
        'north attleborough': 'north attleborough town',
        'franklin': 'franklin town',
        'weymouth': 'weymouth town'
    }

    pioneer_combined['Muni'] = pioneer_combined['Muni'].map(rename_dic).fillna(pioneer_combined['Muni'])

    #Now process min lot size questions
    pioneer_combined = process_min_lot_size(pioneer_combined)
    
    return pioneer_combined, questions

def process_gis(data, residential = False):

    # Min lot size of 999 means that there are no requirements so we drop that
    data= data[data['minlotsize'] != 999]

    #Drop min lot size of -999 too
    data = data[data['minlotsize'] != -999]

    # Also drop min lot sizes of 0 from the analysis
    data = data[data['minlotsize'] != 0]

    #If residential, then filter for singlefamilydummy == 1
    if residential:
        data = data[data['singlefamilydummy'] == 1]

    # Filter the required columns
    data_filtered = data[['municipality', 'zonedist', 'minlotsize']]


    # Group by muni and zonedist and take the min min lot size
    data_filtered = data_filtered.groupby(['municipality', 'zonedist'])['minlotsize'].min().reset_index()

    # Create a list to store the final data
    final_data = []

    # Iterate over each unique municipality
    for muni in data_filtered['municipality'].unique():
        subset = data_filtered[data_filtered['municipality'] == muni]
        list_of_dicts = [{'district_name': zonedist, 'min_lot_size': minlotsize}
                         for zonedist, minlotsize in zip(subset['zonedist'], subset['minlotsize'])]
        final_data.append({'Muni': muni, 'Answer': list_of_dicts})

    # Convert the list to a DataFrame
    final_df = pd.DataFrame(final_data)

    # Clean/format
    final_df['Muni'] = final_df['Muni'].str.lower()

    if residential:
        final_df['Question'] = '28'
    else:
        final_df['Question'] = '27'

    return final_df

def load_gis():
    
    # Load the .dta file into a pandas DataFrame
    data = pd.read_stata(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Data\MASS GIS\Stata files\Combined_Main_Tables_MASS_GIS.dta")

    residential_gis = process_gis(data, residential=True)
    total_gis = process_gis(data, residential=False)

    final_df = pd.concat([residential_gis, total_gis], ignore_index=True)

    return final_df


def load_excel_files(file_path):

    #Load the light file
    light_df = pd.read_excel(os.path.join(file_path,'Light Data.xlsx'))

    #Load the full file
    full_df = pd.read_excel(os.path.join(file_path,'Full Data.xlsx'))

    #Return both as one dictionary
    return {'light':light_df,'full':full_df}


def merge_data(data, pioneer):
    for key, df in data.items():
        # First, adjust the name
        df['Muni'] = df.apply(lambda row: adjust_muni_name(row), axis=1)

        # Ensure 'Question' column is of type string
        df['Question'] = df['Question'].astype(str)
        pioneer['Question'] = pioneer['Question'].astype(str)

        # Then, merge with pioneer
        merged = pd.merge(df, pioneer, on=['Muni', 'Question'], suffixes=('', '_pioneer'))

        # Add back to dictionary
        data[key] = merged

        # If key is 'light' then drop the 'Evidence' and 'Ordinance.com' columns
        if key == 'light':
            data[key].drop(columns=['Evidence', 'Ordinance.com'], inplace=True)

        # Now create column for whether the answer is the same
        data[key]['Correct'] = data[key]['Answer'] == data[key]['Answer_pioneer']
        data[key]['Correct'] = data[key]['Correct'].astype(int)

    return data


def export_data(merged,file_path):

    # First, we need to clean the data
    def sanitize_string(s):
        if isinstance(s, str):
            # Replace surrogate pairs with a placeholder or remove them
            return s.encode('utf-8', 'replace').decode('utf-8')
        else:
            # Return the original input if it's not a string
            return s

    # Apply the function to all string columns
    for key, df in merged.items():
        for column in df.columns:
            if df[column].dtype == 'object':  # For columns with string values
                df[column] = df[column].apply(sanitize_string)

    #First, export the light dataset
    merged['light'].to_excel(os.path.join(file_path,'Light Data Merged.xlsx'),index = False)

    #Now export the full dataset
    merged['full'].to_excel(os.path.join(file_path,'Full Data Merged.xlsx'),index = False)

    return

##


def merge_w_pioneer(file_path):

    # Load the data
    data = load_excel_files(file_path)

    # Load the pioneer data
    pioneer, questions = load_pioneer()

    # Merge the two together and see which answers are correct
    merged = merge_data(data, pioneer)

    # Now export the data
    export_data(merged, file_path)

    return merged['light']


if __name__ == 'Main':

    #Define which dataset to use
    dataset = 'Claude Opus Testing'

    file_path = os.path.join(config['processed_data'], 'Model Output', dataset)

    merge_w_pioneer(file_path)


