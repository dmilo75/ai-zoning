import pickle
import pandas as pd
import numpy as np
import os
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


#Used to map naming conventions
sample_map = pd.read_excel(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Charts and Tables\Data\Sample Data.xlsx")


#Model you are running this on 
model = 'Full Run'

dir_path = os.path.join(config['processed_data'],'Model Output',model)

#%% Reformat and merge the data

def load_data(file_path):
    if os.path.isdir(file_path):
        all_data = []
        for file in os.listdir(file_path):
            if file.endswith(".pkl"):
                file_full_path = os.path.join(file_path, file)
                with open(file_full_path, 'rb') as f:
                    data = pickle.load(f)
                    all_data.extend(data)
        return all_data
    else:
        with open(file_path, 'rb') as file:
            return pickle.load(file)

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
            int(row['Answer'])
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

#Process the minimum lot size
def process_min_lot_size(column):
    # Initialize min_lot_sizes list
    min_lot_sizes = []

    # Ensure column is iterable; if not, return NaNs
    if not isinstance(column, list):
        return np.nan, np.nan

    # Try to extract numeric min_lot_sizes and handle errors
    try:
        for item in column:
            if isinstance(item, dict) and 'min_lot_size' in item:
                min_lot_size = item['min_lot_size']
                if isinstance(min_lot_size, (int, float)) and not np.isnan(min_lot_size):
                    min_lot_sizes.append(min_lot_size)
    except Exception as e:
        return np.nan, np.nan

    # Calculate average and minimum if possible
    if min_lot_sizes:
        return np.mean(min_lot_sizes), np.min(min_lot_sizes), np.max(min_lot_sizes)
    else:
        return np.nan, np.nan
    
def lot_metrics(df):
    
    df['Type'] = ""
    
    # Create a copy of the rows for question 27 to modify
    question_27_rows = df[df['Question'] == 27].copy()
    
    # Apply the function to each row and store the results in new columns
    question_27_rows[['Mean_Lot_Size', 'Min_Lot_Size','Max_Lot_Size']] = question_27_rows.apply(
        lambda row: pd.Series(process_min_lot_size(row['Answer'])), axis=1
    )
    
    # Create two new DataFrames for Mean and Min values, respectively
    mean_rows = question_27_rows.copy()
    min_rows = question_27_rows.copy()
    max_rows = question_27_rows.copy()
    
    # Update the 'Type' column to differentiate between Mean and Min
    mean_rows['Type'] = 'Mean'
    min_rows['Type'] = 'Min'
    max_rows['Type'] = 'Max'
    
    # Replace the 'Answer' column with the calculated Mean and Min values
    mean_rows['Answer'] = mean_rows['Mean_Lot_Size']
    min_rows['Answer'] = min_rows['Min_Lot_Size']
    max_rows['Answer'] = max_rows['Max_Lot_Size']
    
    # Concatenate the modified rows back into the original DataFrame
    new_rows = pd.concat([mean_rows, min_rows,max_rows], ignore_index=True)
    old_rows = df[df['Question'] != 27]
    df_extended = pd.concat([old_rows, new_rows], ignore_index=True)
    
    # Drop the temporary columns if no longer needed
    df_extended.drop(['Mean_Lot_Size', 'Min_Lot_Size','Max_Lot_Size'], axis=1, inplace=True)
        
    return df_extended

# Load and process a general pickle file (e.g., data, gpt35, etc.)
def load_and_process_pickle(file_path):
    
    loaded_data = load_data(file_path)
        
    processed_data = []

    # Loop through all answers to update based on Dont_Know/dont_know parameter
    for entry in loaded_data:
        updated_entry = update_dont_know(entry)
        processed_data.append(updated_entry)

    #Turn to dataframe
    df = pd.DataFrame(processed_data)

    # Ensure answers have correct format
    df['Answer'] = df.apply(process_answers, axis=1)
    
    #Now process question 27 min lot size
    df_ext = lot_metrics(df)

    #Ensure no duplicate muni question pairs, keep the first when duplicated
    df_ext = df_ext.drop_duplicates(subset = ['Muni','Question','Type'], keep = 'first')
    
    # Ensure 'Question' column is string type
    df_ext['Question'] = df_ext['Question'].astype(str)
    
    #Rename
    df_ext = df_ext.rename(columns = {'Muni':'muni_full'})
    
    # Split the 'Muni' column into three new columns
    df_ext[['Muni', 'geoid', 'State']] = df_ext['muni_full'].str.split('#', expand=True)

    return df_ext

#%%Main function

#Load and process data
data = load_and_process_pickle(dir_path)

#Select columns we want
data_selected = data[['Answer','Question','Muni','Type','State','geoid']]

#Export
data_selected.to_excel(os.path.join(config['processed_data'], model+".xlsx"))




