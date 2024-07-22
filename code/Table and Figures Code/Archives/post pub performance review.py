#Load config file
import os
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

import pickle
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mstats
from scipy import stats
from importlib import reload
from collections import OrderedDict

yes_no_qs = [4,5,6,8,9,11,13,14,17,20,21]
numerical = [2,22]
categorical = [7,12,15]
lot_size = [27]
root = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning"
import os
import sys
sys.path.append(root+r"\Embedding_Project\Compare Results\Chart Formatter")
import ChartFormatter as cf
reload(cf)
cf.export = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\results\figures"

root = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Embedding_Project\Compare Results\Data"
os.chdir(root)

#Used to map naming conventions
sample_map = pd.read_excel(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Charts and Tables\Data\Sample Data.xlsx")

#%% Reformat and merge the data

#Adjust naming convention so we can merge
def adjust_muni_name(row):
    if '#' in row['Muni']:
        return row['Muni'].split('#')[0]
    else:
        lowercase_muni = row['Muni'].lower()
        matching_row = sample_map[(sample_map['State'] == 'ma') & (sample_map['Muni'] == lowercase_muni)]
        
        # If a matching row exists, replace with 'UNIT_NAME'. Otherwise, use the original 'Muni'.
        return matching_row['UNIT_NAME'].iloc[0] if not matching_row.empty else row['Muni']

def convert_name_format(row):

    #First, get the muni name, fips place code and state from name format delimited by '#'
    name_format = row['Muni']
    name_format = name_format.split('#')
    muni_name, fips_place, state = name_format[0], name_format[1], name_format[2]

    #Now find the right row from sample_map
    matching_row = sample_map[(sample_map['State'] == state) & (sample_map['FIPS_PLACE'] == int(fips_place))]

    #Now return the correct format of unit name, census id and state
    return matching_row['UNIT_NAME'].iloc[0] + '#' + str(matching_row['CENSUS_ID_PID6'].iloc[0]) + '#' + matching_row['STATE'].iloc[0]


def load_data(file_path):
    if os.path.isdir(file_path):
        # Initialize an ordered dictionary to maintain order and uniqueness
        unique_data = OrderedDict()

        # Get list of files sorted by modification time, newest last
        files = sorted(
            [f for f in os.listdir(file_path) if f.endswith(".pkl")],
            key=lambda x: os.path.getmtime(os.path.join(file_path, x))
        )

        for file in files:
            file_full_path = os.path.join(file_path, file)
            with open(file_full_path, 'rb') as f:
                data = pickle.load(f)
                # Update the dictionary with the current batch of data
                # 'Muni' and 'Question' pairs are used as the dictionary key
                for item in data:
                    # Ensure both 'Muni' and 'Question' keys exist to avoid KeyError
                    if 'Muni' in item and 'Question' in item:
                        unique_key = (item['Muni'], item['Question'])
                        unique_data[unique_key] = item

        # Convert the dictionary back to a list, keeping only the values (the latest entries)
        all_data = list(unique_data.values())
        return all_data
    else:
        with open(file_path, 'rb') as file:
            return pickle.load(file)

def load_pioneer(file_path):

    # Load the .pkl file into a dictionary
    loaded_data = load_data(file_path)

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
    
    return pioneer_combined, questions

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

def filter_training_sample(df,training_sample):

    filtered_df = df[df['Muni'].isin(training_sample)]

    #Older versions of results file we dont have to filter, and they have a different format
    if len(filtered_df) == 0:

        #We need to convert the naming convention then
        df['Muni'] = df.apply(lambda row: convert_name_format(row), axis=1)

        filtered_df = df[df['Muni'].isin(training_sample)]

        return filtered_df
    else:
        return filtered_df


# Load and process a general pickle file (e.g., data, gpt35, etc.)
def load_and_process_pickle(file_path,training_sample):
    
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

    #Ensure no duplicate muni question pairs, keep the first when duplicated
    df = df.drop_duplicates(subset = ['Muni','Question'], keep = 'first')

    #Filter for training sample
    df = filter_training_sample(df,training_sample)
    
    # Adjust Muni names using the sample_map
    df['Muni'] = df.apply(lambda row: adjust_muni_name(row), axis=1)
   
    # Ensure 'Question' column is string type
    df['Question'] = df['Question'].astype(str)
    
    return df


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

    '''
    samp = data[data['municipality'] == 'MIDDLETON']

    #samp = samp[['municipality','zonedist','prim_use','gen_use','perm_use','minlotsize']]

    #sort on zonedist
    samp = samp.sort_values(by = 'zonedist')
    '''

    residential_gis = process_gis(data, residential=True)
    total_gis = process_gis(data, residential=False)

    final_df = pd.concat([residential_gis, total_gis], ignore_index=True)


    return final_df

def load_and_merge_data(datasets,type):
    # Load testing data
    pioneer, questions = load_pioneer('pioneer.pkl')

    # Load in training sample split
    if type == 'training':
        training_sample = pd.read_pickle(
            r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\raw data\training.pickle")
    elif type == 'testing':
        training_sample = pd.read_pickle(
            r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\raw data\testing.pickle")

    # Load and process the data
    answers = {}
    for dataset in datasets:
        answers[dataset['Name']] = load_and_process_pickle(dataset['Path'],training_sample)

    merged = {}


    # Merge in pioneer data
    for answer in answers:

        # Merge the dataframes
        merged[answer] = pd.merge(answers[answer], pioneer, on=['Muni', 'Question'],
                                   suffixes=('', '_pioneer'))

    #Ensure all dataframes share the same set of values for the 'Muni' column
    #First, loop over all the dataframes and get the unique values for the 'Muni' column
    unique_munis = [df['Muni'].unique() for df in merged.values()]
    #Take the intersection
    common_munis = set(unique_munis[0]).intersection(*unique_munis[1:])

    print("Length of common munis:")
    print(len(common_munis))

    #Now filter the dataframes
    for key in merged:
        merged[key] = merged[key][merged[key]['Muni'].isin(common_munis)]

    return merged, questions

##Performance Metrics calculation code

def process_binary_questions(df, binary_qs):
    # Initialize a dictionary to hold "Don't Know" percentages
    dont_know_percentages = {}

    # Filter for binary questions
    binary_df = df[df['Question'].isin(binary_qs)].copy()

    # Calculate "Don't Know" percentage for each binary question before mapping and dropping nulls
    for question in binary_qs:
        question_df = df[df['Question'] == question]  # Use the original df to include nulls in calculation
        total_count = question_df.shape[0]
        non_null_count = question_df.dropna(subset=['Answer', 'Answer_pioneer']).shape[0]
        dont_know_percentages[question] = ((total_count - non_null_count) / total_count) * 100 if total_count != 0 else 0

    # Turn answers into 1 and 0
    binary_df['Answer'] = binary_df['Answer'].map({'Yes': 1, 'No': 0})
    binary_df['Answer_pioneer'] = binary_df['Answer_pioneer'].map({'Yes': 1, 'No': 0})

    # Drop null answers
    binary_df = binary_df.dropna(subset=['Answer', 'Answer_pioneer'])

    # Create 'Correct' column as integer
    binary_df['Correct'] = (binary_df['Answer'] == binary_df['Answer_pioneer']).astype(int)

    return binary_df, dont_know_percentages


def calc_binary_performance(df):
    binary_qs = [str(x) for x in yes_no_qs]
    binary_df, dont_know_percentages = process_binary_questions(df, binary_qs)

    # Prepare the results DataFrame
    results = pd.DataFrame(columns=['Question', 'Main Performance Metric', 'Alternative Performance Metric', "Don't Know"])

    for question in binary_qs:
        question_df = binary_df[binary_df['Question'] == question]

        # Calculate accuracy
        accuracy = question_df['Correct'].mean() * 100  # Convert to percentage

        # Calculate RSE (Review if this calculation is appropriate for binary data)
        y_bar = question_df['Answer_pioneer'].mode()[0]
        numerator = ((question_df['Answer'] - 1) ** 2).sum()
        denominator = ((question_df['Answer_pioneer'] - y_bar) ** 2).sum()
        rse = numerator / denominator if denominator != 0 else 0

        # Append results
        new_row = pd.DataFrame({
            'Question': [question],
            'Main Performance Metric': [rse],
            'Alternative Performance Metric': [accuracy],
            "Don't Know": [dont_know_percentages.get(question, 0)]  # Retrieve "Don't Know" percentage
        })

        # Use concat to add the new row to the existing DataFrame
        results = pd.concat([results, new_row], ignore_index=True)

    return results

def process_numerical_data(df, numerical):
    numerical_string = [str(x) for x in numerical]
    # Initialize a dictionary to hold "Don't Know" percentages
    dont_know_percentages = {}

    # Filter the DataFrame for numerical questions only
    numerical_df = df[df['Question'].isin(numerical_string)]

    # Calculate "Don't Know" percentage for each question before dropping nulls
    for question in numerical_string:
        question_df = numerical_df[numerical_df['Question'] == question]
        total_count = question_df.shape[0]
        non_null_count = question_df.dropna(subset=['Answer', 'Answer_pioneer']).shape[0]
        dont_know_percentages[question] = ((total_count - non_null_count) / total_count) * 100 if total_count != 0 else 0

    # Drop null answers
    numerical_df = numerical_df.dropna(subset=['Answer', 'Answer_pioneer'])

    # Convert answers to float
    numerical_df['Answer'] = numerical_df['Answer'].astype(float)
    numerical_df['Answer_pioneer'] = numerical_df['Answer_pioneer'].astype(float)

    # Winsorize values at 1% and 99% levels for both 'Answer' and 'Answer_pioneer'
    numerical_df['Answer'] = mstats.winsorize(numerical_df['Answer'], limits=[0.05, 0.05])
    numerical_df['Answer_pioneer'] = mstats.winsorize(numerical_df['Answer_pioneer'], limits=[0.05, 0.05])

    return numerical_df, dont_know_percentages

def calc_numerical_performance(df):
    numerical_df, dont_know_percentages = process_numerical_data(df, numerical)

    # Prepare the results DataFrame
    results = pd.DataFrame(columns=['Question', 'Main Performance Metric', 'Alternative Performance Metric', "Don't Know"])


    for question in numerical:
        question_str = str(question)
        question_df = numerical_df[numerical_df['Question'] == question_str]

        # Calculate RSE
        y_bar = question_df['Answer_pioneer'].mean()
        numerator = ((question_df['Answer_pioneer'] - question_df['Answer']) ** 2).sum()
        denominator = ((question_df['Answer_pioneer'] - y_bar) ** 2).sum()
        rse = numerator / denominator if denominator != 0 else 0

        # Calculate correlation
        if len(question_df) > 1:
            correlation, _ = stats.pearsonr(question_df['Answer'], question_df['Answer_pioneer'])
        else:
            correlation = None

        # Retrieve "Don't Know" percentage for the current question
        dont_know_percentage = dont_know_percentages.get(question_str, 0)

        # Create a new DataFrame for the row you want to add, now including "Don't Know" percentage
        new_row = pd.DataFrame({
            'Question': [question_str],
            'Main Performance Metric': [rse],
            'Alternative Performance Metric': [correlation],
            "Don't Know": [dont_know_percentage]
        })

        # Use concat to add the new row to the existing DataFrame
        results = pd.concat([results, new_row], ignore_index=True)

    return results


#Function calculate min lot size by converting units
def calc_min_lot_size(item):
    min_lot_size = item['min_lot_size']

    if 'unit' in item:
        unit = item['unit']

        if unit == 'Acres':
            # Convert acres to square feet
            min_lot_size = min_lot_size * 43560

    return min_lot_size

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
                min_lot_size = calc_min_lot_size(item)
                if isinstance(min_lot_size, (int, float)) and not np.isnan(min_lot_size):
                    min_lot_sizes.append(min_lot_size)
    except Exception as e:
        return np.nan, np.nan

    # Calculate average and minimum if possible
    if min_lot_sizes:
        return np.mean(min_lot_sizes), np.min(min_lot_sizes)
    else:
        return np.nan, np.nan


def calc_min_lot_perf(df):

    total_results = pd.DataFrame(columns=['Question', 'Main Performance Metric', 'Alternative Performance Metric', "Don't Know"])

    #Loop over min lot size questions
    for question in ['27','28']:
        sub_df = df[df['Question'] == question].copy()
        results = calc_min_lot_perf_question(sub_df,question)
        total_results = pd.concat([total_results,results],ignore_index = True)
        print(total_results)

    return total_results

def calc_min_lot_perf_question(df,question):
    # Prepare the results DataFrame
    results = pd.DataFrame(columns=['Question', 'Main Performance Metric', 'Alternative Performance Metric',"Don't Know"])

    # Process min_lot_size for both 'Answer' and 'Answer_pioneer'
    df['avg_min_lot_size_answer'], df['min_min_lot_size_answer'] = zip(*df['Answer'].apply(process_min_lot_size))
    df['avg_min_lot_size_pioneer'], df['min_min_lot_size_pioneer'] = zip(*df['Answer_pioneer'].apply(process_min_lot_size))

    #Drop rows where pioneer data is missing
    df = df.dropna(subset = ['avg_min_lot_size_pioneer'])

    # Calculate "Don't Know" percentage before dropping null answers
    total_count = df.shape[0]
    non_null_count = df.dropna(subset=['avg_min_lot_size_answer']).shape[0]
    dont_know_percentage = ((total_count - non_null_count) / total_count) * 100 if total_count != 0 else 0

    #Now drop rows where our model doesn't know the answer
    df = df.dropna(subset = ['avg_min_lot_size_answer'])

    # Winsorize the processed values
    for col in ['avg_min_lot_size_answer', 'min_min_lot_size_answer', 'avg_min_lot_size_pioneer', 'min_min_lot_size_pioneer']:
        non_null_indices = df[col].notnull()
        df.loc[non_null_indices, col] = mstats.winsorize(df.loc[non_null_indices, col], limits=[0.05, 0.05])

    # Calculate RSE and correlation for 'avg_min_lot_size'
    rse_avg, corr_avg = calculate_metrics(df, 'avg_min_lot_size_answer', 'avg_min_lot_size_pioneer')
    new_row = pd.DataFrame({
        'Question': [str(question)+'mean'],
        'Main Performance Metric': [rse_avg],
        'Alternative Performance Metric': [corr_avg],
        "Don't Know": [dont_know_percentage]  # Add the same "Don't Know" percentage
    })

    # Use concat to add the new row to the existing DataFrame
    results = pd.concat([results, new_row], ignore_index=True)

    # Calculate RSE and correlation for 'min_min_lot_size'
    rse_min, corr_min = calculate_metrics(df, 'min_min_lot_size_answer', 'min_min_lot_size_pioneer')
    new_row = pd.DataFrame({
        'Question': [str(question)+'min'],
        'Main Performance Metric': [rse_min],
        'Alternative Performance Metric': [corr_min],
        "Don't Know": [dont_know_percentage]  # Add the same "Don't Know" percentage
    })

    # Use concat to add the new row to the existing DataFrame
    results = pd.concat([results, new_row], ignore_index=True)

    return results

def calculate_metrics(df, answer_col, pioneer_col):
    # Calculate RSE
    y_bar = df[pioneer_col].mean()
    numerator = ((df[pioneer_col] - df[answer_col]) ** 2).sum()
    denominator = ((df[pioneer_col] - y_bar) ** 2).sum()
    rse = numerator / denominator if denominator != 0 else np.nan

    # Calculate correlation
    temp_df = df[[pioneer_col, answer_col]].dropna()
    if len(temp_df) > 1:
        correlation, _ = pearsonr(temp_df[pioneer_col], temp_df[answer_col])
    else:
        correlation = np.nan

    return rse, correlation

def calc_perf_metrics(data):

    results = {}

    for key in data:
        print(key)
        df = data[key]
        results[key] = {}

        # Calculate performance metrics for binary questions
        results[key]['Binary'] = calc_binary_performance(df)

        # Calculate performance metrics for numerical questions
        numerical_perf = calc_numerical_performance(df)

        # Calculate performance metrics for min_lot_size question
        min_lot_perf = calc_min_lot_perf(df)

        #Combine numerical and min_lot_size performance metrics
        results[key]['Continuous'] = pd.concat([numerical_perf, min_lot_perf], ignore_index=True)


    return results

def prep_chart_data(perf_metrics):
    # Initialize empty DataFrames for Binary and Continuous with the specified columns
    columns = ['Percent Correct', 'Percent Incorrect', "Don't Know"]
    binary_df = pd.DataFrame(columns=columns)
    continuous_df = pd.DataFrame(columns=columns)

    for category, data_types in perf_metrics.items():
        for data_type, df in data_types.items():
            # Calculate mean of 'Don't Know'
            mean_dont_know = df['Don\'t Know'].mean()

            # Calculate percent correct and incorrect based on the alternative performance metric
            if data_type == 'Continuous':
                df['Alternative Performance Metric'] = 100* df['Alternative Performance Metric']
            percent_correct = (100 - mean_dont_know) * df['Alternative Performance Metric'].mean() / 100
            percent_incorrect = 100 - mean_dont_know - percent_correct

            # Append the results to the appropriate DataFrame
            if data_type == 'Binary':
                binary_df.loc[category] = [percent_correct, percent_incorrect, mean_dont_know]
            elif data_type == 'Continuous':
                continuous_df.loc[category] = [percent_correct, percent_incorrect, mean_dont_know]

    #Sort binary_df and continuous_df by 'Percent Correct'
    binary_df = binary_df.sort_values(by='Percent Correct', ascending=False)
    continuous_df = continuous_df.sort_values(by='Percent Correct', ascending=False)

    return {'Binary': binary_df, 'Continuous': continuous_df}


def dot_plt(datasets):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6),dpi = 100)

    # Colors for each dataset
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    # Check to make sure we have enough colors, otherwise use a colormap
    if len(datasets) > len(colors):
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(datasets))]

    # Dictionary to hold labels for the legend
    legend_labels = {}

    # Iterate through each dataset
    for dataset_idx, (dataset_name, data_types) in enumerate(datasets.items()):
        # Use the same color for both 'Binary' and 'Continuous'
        color = colors[dataset_idx % len(colors)]

        # Add or update the legend label for this dataset and data type
        legend_labels[f"{dataset_name}"] = color

        # Iterate through both 'Binary' and 'Continuous' data types
        for data_type, df in data_types.items():
            # Extract 'Question' and 'Alternative Performance Metric'
            questions = df['Question']
            metrics = df['Alternative Performance Metric']

            # Plot each question's metric as a dot
            ax.scatter(questions, metrics, color=color, alpha=0.6, edgecolors='w')



    # Creating custom legend entries
    legend_entries = [plt.Line2D([0], [0], color=color, lw=4, marker='o', markersize=10, label=label) for label, color
                      in legend_labels.items()]

    # Add legend to the plot with custom entries
    ax.legend(handles=legend_entries, loc='best', title='Datasets')

    # Improve the plot aesthetics
    ax.set_xlabel('Question')
    ax.set_ylabel('Alternative Performance Metric')
    plt.xticks(rotation=45, ha="right")
    plt.title('Alternative Performance Metrics by Question')
    plt.tight_layout()

    # Show plot
    plt.show()


def make_charts(perf_metrics):

    transformed_data = prep_chart_data(perf_metrics)

    #Make western scatter plot
    dot_plt(perf_metrics)

    #Now again with filtered dict
    #Make keys to keep
    keys_to_keep = ['Pub 4','End Dec Run','gpt4 4k emb feb11','Emb Filter 4k gpt4']
    # Filtered dictionary using dictionary comprehension
    filtered_dict = {k: perf_metrics[k] for k in keys_to_keep if k in perf_metrics}
    dot_plt(filtered_dict)


    cf.panel(transformed_data,'Performance Metrics','bar',stacked = True, markers = False, banner = False, label = '%', ex = True)

    return

##Latex panel functions


# Adjusting the index of result based on the pioneer dictionary keys (assuming questions dictionary is defined elsewhere)
def adjust_index(index):
    try:
        index = int(index)
        return list(questions)[index - 1]
    except:
        if index == '27mean':
            return 'Mean of Min Lot Sizes (Square Feet)'
        elif index == '27min':
            return 'Minimum of Min Lot Sizes (Square Feet)'

        return index


def interlace_dfs(d1, d2):
    """Interlace columns of two dataframes using toolz's interleave."""
    # Ensure the dataframes have the same number of columns
    assert d1.shape[1] == d2.shape[1], "DataFrames must have the same number of columns to interlace"

    # Interleave columns from both dataframes and return the interleaved dataframe
    return pd.concat([d1, d2], axis=1, keys=['d1', 'd2']).sort_index(axis=1, level=1, sort_remaining=False).droplevel(0,
                                                                                                                      axis=1)


def df_to_latex_panel(df):
        # Adjust index
        adjusted_index = [adjust_index(idx) for idx in df.index]
        df.index = adjusted_index
        df.index.name = 'Question'

        # Convert the dataframe to a LaTeX formatted string
        latex_str = df.to_latex(float_format=lambda x: '{:.1f}'.format(x),
                                index=True, escape=False, header=True)

        # Cut out formatting stuff
        latex_str = latex_str.split('midrule')[1].split('bottomrule')[0][:-1].strip()

        # Add midrule for cumulative average
        latex_str = latex_str.split('Cumulative Average')[0] + "\\midrule\nCumulative Average" + \
                    latex_str.split('Cumulative Average')[1]

        return latex_str.strip()+'\n\\bottomrule'

def process_results_data(performance_results):
    results = {}
    for data_type in ['Binary', 'Continuous']:
        # Get index from questions in one of the dataframes
        results[data_type] = pd.DataFrame()

        for key in performance_results:
            results[data_type][key] = performance_results[key][data_type]['Main Performance Metric']

        # Set the index now
        results[data_type].index = performance_results[key][data_type]['Question']

    alt_performance = {}
    for data_type in ['Binary', 'Continuous']:
        # Get index from questions in one of the dataframes
        alt_performance[data_type] = pd.DataFrame()

        for key in performance_results:
            alt_performance[data_type][key] = performance_results[key][data_type]['Alternative Performance Metric']

        # Set the index now
        alt_performance[data_type].index = performance_results[key][data_type]['Question']

    return results, alt_performance


def make_latex_tables(performance_results, export = False):

    #Process data
    results, alt_performance = process_results_data(performance_results)

    pd.set_option('display.max_colwidth', None)

    # Make cumulative means and format
    for cat, df in results.items():

        df.loc['Cumulative Average'] = df.mean(axis=0)
        df.loc['Cumulative Median'] = df.median(axis=0)

        # Formatting the columns to have one decimal point where applicable
        df = df.applymap(lambda x: f"{x:.1f}" if isinstance(x, float) else x)

        results[cat] = df

    # Format alt performance columns
    for cat, df in alt_performance.items():

        df.loc['Cumulative Average'] = df.mean(axis=0)
        df.loc['Cumulative Median'] = df.median(axis=0)

        if cat == 'Binary':
            df = df.applymap(lambda x: f"{int(x):d}" + "\\%")

        elif cat == 'Continuous':
            df = df.applymap(lambda x: f"{x:.1f}")

        alt_performance[cat] = df

    latex = {}
    for cat, df in results.items():
        models = df.columns
        # If cat is 'Binary Questions', merge with alt_performance and interlace columns
        df = interlace_dfs(df, alt_performance[cat])
        latex[cat] = df_to_latex_panel( df)

    if export:
        root = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning"
        os.chdir(root)
        with open(os.path.join(config['tables_path'], 'latex', 'cts_perf.tex'), 'w') as file:
            file.write(latex['Continuous'])

        with open(os.path.join(config['tables_path'], 'latex', 'bin_perf.tex'), 'w') as file:
            file.write(latex['Binary'])

    return latex

def idk_latex(perf_metrics, export = False):

    #Establish dictionary with continuous and binary categories
    cats = ['Continuous','Binary']
    dic = {}
    for cat in cats:
        dic[cat] = pd.DataFrame()
        for dataset in perf_metrics:
            dic[cat][dataset] = perf_metrics[dataset][cat]['Don\'t Know']
        dic[cat].index = perf_metrics[dataset][cat]['Question']

    #Make cumulative means and format
    for cat, df in dic.items():
        df.loc['Cumulative Average'] = df.mean(axis=0)

        #Formatting the columns to have no decimals
        df = df.applymap(lambda x: f"{int(x):d}" + "\\%")
        dic[cat] = df

    #Make latex
    latex = {}
    for cat, df in dic.items():
        latex[cat] = df_to_latex_panel(df)

    if export:
        with open(os.path.join(config['tables_path'], 'latex', 'cts_idk.tex'), 'w') as file:
            file.write(latex['Continuous'])

        with open(os.path.join(config['tables_path'], 'latex', 'bin_idk.tex'), 'w') as file:
            file.write(latex['Binary'])

    return latex

def calc_agg(data):
    # First, append each dataframe in data dictionary to one big dataframe
    big_df = pd.concat(data.values())

    # Custom aggregation function
    def custom_agg(x):
        # Check if majority of entries are null
        if pd.isnull(x).mean() > 0.5:
            return np.nan

        if x.dtype == 'object':
            mode_series = x.mode()
            if not mode_series.empty:
                return mode_series.iloc[0]
            else:
                return None  # Or your preferred default value for when mode is not found
        else:
            return x.mean()

    agg = big_df.groupby(['Muni', 'Question']).agg(custom_agg).reset_index()

    return agg

##


'''
Step 1: Load in and merge all data
'''

datasets = [

#{'Name': 'Testing Sample', 'Path': r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\Model Output\Testing Sample 35"},

{'Name': 'Testing 4', 'Path': r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\Model Output\Testing Sample"},
{'Name': 'Testing 3.5', 'Path': r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\Model Output\Testing Sample 35"},


]

# Load and merge the data
data, questions = load_and_merge_data(datasets,'testing')


'''
Step 2: Calculate performance metrics
'''

perf_metrics = calc_perf_metrics(data)

#%%results analysis 

'''
Step 3: Make charts
'''

make_charts(perf_metrics)

'''
Step 4: Make tables
'''

#tables = make_latex_tables(perf_metrics, export = False)
#idk_tables = idk_latex(perf_metrics, export = False)

##

