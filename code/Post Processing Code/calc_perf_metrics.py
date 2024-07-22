import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import mstats
import pickle
import os
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

yes_no_qs = config['yes_no_qs']
numerical = config['numerical']+['27Min','27Mean','28Min','28Mean']


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
    binary_qs = [str(x) for x in yes_no_qs if str(x) in df['Question'].values]
    binary_df, dont_know_percentages = process_binary_questions(df, binary_qs)

    # Prepare the results DataFrame
    results = pd.DataFrame(columns=['Question', 'Main Performance Metric', 'Alternative Performance Metric', "Don't Know","N"])

    for question in binary_qs:

        #Get sub question dataframe
        question_df = binary_df[binary_df['Question'] == question]

        # Calculate accuracy
        accuracy = question_df['Correct'].mean() * 100  # Convert to percentage

        # Calculate RSE
        y_bar = question_df['Answer_pioneer'].mode()[0]
        numerator = ((question_df['Answer_pioneer'] - question_df['Answer']) ** 2).sum()
        denominator = ((question_df['Answer_pioneer'] - y_bar) ** 2).sum()
        rse = numerator / denominator if denominator != 0 else np.nan

        # Append results
        new_row = pd.DataFrame({
            'Question': [question],
            'Main Performance Metric': [rse],
            'Alternative Performance Metric': [accuracy],
            "Don't Know": [dont_know_percentages.get(question, 0)], # Retrieve "Don't Know" percentage
            'N':len(question_df.index) # Number of observations
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
    numerical_df['Answer'] = mstats.winsorize(numerical_df['Answer'], limits=[0.01, 0.01])
    numerical_df['Answer_pioneer'] = mstats.winsorize(numerical_df['Answer_pioneer'], limits=[0.01, 0.01])

    return numerical_df, dont_know_percentages

def calc_continuous(df):
    numerical_df, dont_know_percentages = process_numerical_data(df, numerical)

    # Prepare the results DataFrame
    results = pd.DataFrame(columns=['Question', 'Main Performance Metric', 'Alternative Performance Metric', "Don't Know","N"])


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
            "Don't Know": [dont_know_percentage],
            'N': len(question_df.index)  # Number of observations
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

def load_adjustments():
    # Load in the manual corrections
    adjustments = pd.read_excel(os.path.join(config['raw_data'],'Error Adjustments', 'binary_adjustments.xlsx'))

    # Filter to 'Correct' == 0
    adjustments = adjustments[adjustments['Whether Correct'] == 0]

    #Ensure the 'Question' column is a string
    adjustments['Question'] = adjustments['Question'].astype(str)

    return adjustments

#Function to adjust Pioneer answers based on manual corrections
def binary_adjust(df, drop_ambiguous, adjustments):
    # For 'Muni', 'Question' pairs in adjustment where 'Reason for Difference' is 'Pioneer Outside', drop from df
    drop_pairs = adjustments[(adjustments['Reason for Difference'] == 'Pioneer Outside')][['Muni','Question']]
    df = df[~df.set_index(['Muni','Question']).index.isin(drop_pairs.set_index(['Muni','Question']).index)]

    # For 'Muni', 'Question' pairs in adjustment where 'Reason for Difference' is 'Pioneer Wrong', replace 'Answer_pioneer' with 'Answer'
    replace_pairs = adjustments[(adjustments['Reason for Difference'] == 'Pioneer wrong')][['Muni', 'Question']]
    for index, row in replace_pairs.iterrows():
        df.loc[(df['Muni'] == row['Muni']) & (df['Question'] == row['Question']), 'Answer_pioneer'] = \
            df.loc[(df['Muni'] == row['Muni']) & (df['Question'] == row['Question']), 'Answer']

    # If drop_ambiguous then drop 'Ambiguous' for 'Reason for Difference' too
    if drop_ambiguous:
        drop_pairs = adjustments[adjustments['Reason for Difference'] == 'Ambiguous'][['Muni','Question']]
        df = df[~df.set_index(['Muni','Question']).index.isin(drop_pairs.set_index(['Muni','Question']).index)]

    return df

#Function to adjust numerical questions
def numerical_adjust(df,drop_ambiguous):

    #Get adjustments
    adjustments = pd.read_excel(os.path.join(config['processed_data'], 'numerical_adjustments.xlsx'))

    #Filter to 'Correct' is False
    adjustments = adjustments[adjustments['Correct'] == False]

    #Update 'Answer_pioneer' with 'Correct Answer' for 'Muni' and 'Question' pairs in adjustments
    for index, row in adjustments.iterrows():
        df.loc[(df['Muni'] == row['Muni']) & (df['Question'] == str(row['Question'])), 'Answer_pioneer'] = row['Correct Answer']

    return df

def calc_perf_metrics(df, file_path, adjustment = False):


    results = {}

    if adjustment:

        # Load in the manual corrections
        adjustments = load_adjustments()

        adjusted_binary = binary_adjust(df.copy(), True, adjustments)
        combined = numerical_adjust(adjusted_binary.copy(), True)

        combined.to_excel(
            os.path.join(file_path, 'Light Data Merged Adjusted.xlsx'),
            index=False)

        # Calculate performance metrics for binary questions
        results['With Ambiguous'] = {}
        results['With Ambiguous']['Binary'] = calc_binary_performance(binary_adjust(df,False,adjustments))
        results['With Ambiguous']['Continuous'] = calc_continuous(numerical_adjust(df,False))

        # Calculate performance metrics for binary questions
        results['Without Ambiguous'] = {}
        results['Without Ambiguous']['Binary'] = calc_binary_performance(binary_adjust(df,True,adjustments))
        results['Without Ambiguous']['Continuous'] = calc_continuous(numerical_adjust(df,True))

        #Save the adjusted file


    else:
        # Calculate performance metrics for binary questions
        results['Binary'] = calc_binary_performance(df)

        # Calculate performance metrics for numerical questions
        results['Continuous'] = calc_continuous(df)

    return results


def get_perf(file_path, adjustment = True):

    results = pd.read_excel(os.path.join(file_path,'Light Data Merged.xlsx'))

    #Ensure 'Question' column is of type string
    results['Question'] = results['Question'].astype(str)

    perf_metrics = calc_perf_metrics(results, file_path, adjustment = adjustment)

    with open(os.path.join(file_path,'Performance Metrics.pkl'), 'wb') as f:
        pickle.dump(perf_metrics, f)

    return perf_metrics


if __name__ != "__main__":

    model = 'Testing Sample'

    file_path = os.path.join(config['processed_data'],'Model Output',model)

    get_perf(file_path)
