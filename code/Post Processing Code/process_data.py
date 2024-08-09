#Load config file
import os
import pickle
import pandas as pd
import numpy as np
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

yes_no_qs = [4,5,6,8,9,11,13,14,17,20,21]
numerical = [2,22]
categorical = [7,12,15]
lot_size = [27]
root = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning"
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')



'''
This code intakes the model output and cleans it into an Excel
'''

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
        elif type == 'max':
            return np.max(min_lot_sizes)
    else:
        return np.nan


def process_min_lot_size(df):
    lot_size_qs = ['27', '28']

    for question in lot_size_qs:
        # Get the rows for the question
        question_rows = df[df['Question'] == question]

        # Append the 'Answer' as a string to the 'Explanation' column after a newline
        question_rows['Explanation'] = question_rows['Explanation']

        # Copy question_rows thrice, once for min, once for max, and once for mean
        question_rows_min = question_rows.copy()
        question_rows_max = question_rows.copy()
        question_rows_mean = question_rows.copy()

        # Change the question number in all three to question + 'Min', question + 'Max', or question + 'Mean'
        question_rows_min['Question'] = question + 'Min'
        question_rows_max['Question'] = question + 'Max'
        question_rows_mean['Question'] = question + 'Mean'

        # Now we need to calculate the min, max, and mean for each muni
        question_rows_min['Answer'] = question_rows_min['Answer'].apply(lambda x: calc_lot_size_metric(x, 'min'))
        question_rows_max['Answer'] = question_rows_max['Answer'].apply(lambda x: calc_lot_size_metric(x, 'max'))
        question_rows_mean['Answer'] = question_rows_mean['Answer'].apply(lambda x: calc_lot_size_metric(x, 'mean'))

        # Now drop the original question rows and append the new ones
        df = df[df['Question'] != question]
        df = pd.concat([df, question_rows_min, question_rows_max, question_rows_mean], ignore_index=True)

    return df

def load_data(file_path):
    all_data = []  # Initialize an empty list to store all data

    if os.path.isdir(file_path):
        # Get list of .pkl files in the directory
        files = [f for f in os.listdir(file_path) if f.endswith(".pkl") and 'output' in f]

        for file in files:
            file_full_path = os.path.join(file_path, file)
            with open(file_full_path, 'rb') as f:
                data = pickle.load(f)
                # Get the file's last modified date
                last_modified_date = os.path.getmtime(file_full_path)
                for item in data:
                    item['Date'] = last_modified_date
                # Append the data from each file to the all_data list
                all_data.extend(data)
    else:
        # If the file_path is not a directory, assume it's a single file
        with open(file_path, 'rb') as file:
            all_data = pickle.load(file)
            # Get the file's last modified date
            last_modified_date = os.path.getmtime(file_path)
            for item in all_data:
                item['Date'] = last_modified_date

    return all_data


# Process answers for "I don't know" values
def process_answers(row):

    # If empty answer then that also counts as "I don't know"
    if row['Answer'] == "":
        return np.nan

    # If in yes_no_qs then we need to ensure we get a 'Yes' or 'No' answer
    if row['Question'] in yes_no_qs:
        if row['Answer'] not in ['Yes', 'No']:
            return np.nan

    #Ensure integer for numerical questions
    elif row['Question'] in numerical:

        try:
            float(row['Answer'])

        except:
            return np.nan

    return row['Answer']

def update_dont_know(entry):

    # Skip entry if 'dont_know' is a Series because then min lot size question
    if isinstance(entry.get('dont_know'), pd.Series):
        return entry

    # Get the value of 'Dont_Know' or 'dont_know' parameter
    dont_know_value = entry.get('Dont_Know') or entry.get('dont_know')

    # If 'Dont_Know' or 'dont_know' keys exist and have a value of True, set Answer to NaN
    if dont_know_value:
        entry['Answer'] = np.nan

    return entry

# Load and process a general pickle file (e.g., data, gpt35, etc.)
def load_dataset(file_paths):

    #If file_paths is type string make list
    if isinstance(file_paths,str):
        file_paths = [file_paths]

    loaded_data = []

    for file_path in file_paths:

        loaded_data.extend(load_data(file_path))
        
    processed_data = []

    # Loop through all answers to update based on Dont_Know/dont_know parameter
    for entry in loaded_data:
        updated_entry = update_dont_know(entry)
        processed_data.append(updated_entry)

    #Turn to dataframe
    df = pd.DataFrame(processed_data)

    # Ensure answers have correct format
    df['Answer'] = df.apply(process_answers, axis=1)

    # Ensure 'Question' column is string type
    df['Question'] = df['Question'].astype(str)

    #Sort by question-muni and date
    df = df.sort_values(by = ['Question','Muni','Date'], ascending = False)

    #Keep the first questino muni by date
    df = df.drop_duplicates(subset = ['Question','Muni'],keep = 'first')

    #Process min lot size question
    df = process_min_lot_size(df)

    #Add census id as a column
    df['CENSUS_ID_PID6'] = df['Muni'].apply(lambda x: int(x.split('#')[1]))

    
    return df

def export_data(df,file_path,only_light):

    #Ensure file path exists
    os.makedirs(file_path, exist_ok=True)

    #First, we need to clean the data
    def sanitize_string(s):
        if isinstance(s, str):
            # Replace surrogate pairs with a placeholder or remove them
            return s.encode('utf-8', 'replace').decode('utf-8')
        else:
            # Return the original input if it's not a string
            return s

    # Apply the function to all string columns
    for column in df.columns:
        if df[column].dtype == 'object':  # For columns with string values
            df[column] = df[column].apply(sanitize_string)

    #Now we export the entire dataframe
    if not only_light:
        df.to_excel(os.path.join(file_path,'Full Data.xlsx'),index = False)

    #Now we export the dataframe again but drop the columns for 'Explanation', 'Context', 'Dont_Know', 'and 'Cost'
    #Define columns we want to drop
    to_drop_columns = ['Explanation', 'Context', 'Dont_Know', 'Cost']
    #intersect them with existing columns
    to_drop_columns = list(set(to_drop_columns).intersection(df.columns))
    df.drop(columns = to_drop_columns,inplace = True)
    df.to_excel(os.path.join(file_path,'Light Data.xlsx'),index = False)
    return

##

def process_dataset(file_paths, export_path = None, only_light = False):
    # Load and merge the data
    df = load_dataset(file_paths)

    # Export the data
    export_data(df.copy(), export_path, only_light)

    return df

# Main code here
if __name__ == "__main__":
    # Process the dataset
    dataset_name = 'augest_latest'

    file_path1 = os.path.join(config['processed_data'], 'Model Output', 'process_multi')
    file_path2 = os.path.join(config['processed_data'], 'Model Output', 'national_run_may')
    file_path3 = os.path.join(config['processed_data'], 'Model Output', 'aug_rerun')
    file_path4 = os.path.join(config['processed_data'], 'Model Output', 'national_parking')

    file_paths = [file_path1, file_path2, file_path3, file_path4]

    export_path = os.path.join(config['processed_data'], 'Model Output', dataset_name)

    process_dataset(file_paths, export_path = export_path, only_light = True)



