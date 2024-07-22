import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os
import numpy as np
import yaml

import sys
root = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning"
sys.path.append(root+r"\Embedding_Project\Compare Results\Chart Formatter")
import ChartFormatter as cf
cf.export = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\results\figures"


os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)




def prep_chart_data(perf_metrics, with_ambiguous=False):
    # Initialize empty DataFrames for Binary and Continuous with the specified columns
    columns = ['Percent Correct', 'Percent Incorrect', "Don't Know"]
    binary_df = pd.DataFrame(columns=columns)
    continuous_df = pd.DataFrame(columns=columns)

    # Determine the ambiguous key based on the function parameter
    ambiguous_key = 'With Ambiguous' if with_ambiguous else 'Without Ambiguous'

    for category, ambiguous_types in perf_metrics.items():
        # Access the chosen ambiguous type (With or Without)
        data_types = ambiguous_types[ambiguous_key]

        for data_type, df in data_types.items():
            # Calculate mean of 'Don't Know'
            mean_dont_know = df['Don\'t Know'].mean()

            # Calculate percent correct and incorrect based on the alternative performance metric
            if data_type == 'Continuous':
                df['Alternative Performance Metric'] = 100 * df['Alternative Performance Metric']
            percent_correct = (100 - mean_dont_know) * df['Alternative Performance Metric'].mean() / 100
            percent_incorrect = 100 - mean_dont_know - percent_correct

            # Append the results to the appropriate DataFrame
            if data_type == 'Binary':
                binary_df.loc[category] = [percent_correct, percent_incorrect, mean_dont_know]
            elif data_type == 'Continuous':
                continuous_df.loc[category] = [percent_correct, percent_incorrect, mean_dont_know]

    # Sort binary_df and continuous_df by 'Percent Correct'
    binary_df = binary_df.sort_values(by='Percent Correct', ascending=False)
    continuous_df = continuous_df.sort_values(by='Percent Correct', ascending=False)

    return {'Binary': binary_df, 'Continuous': continuous_df}


def make_charts(perf_metrics):

    transformed_data = prep_chart_data(perf_metrics)

    #Sort index by model_map values order
    transformed_data['Binary'] = transformed_data['Binary'].loc[model_map.values()]
    transformed_data['Continuous'] = transformed_data['Continuous'].loc[model_map.values()]

    for data_type in ['Binary', 'Continuous']:
        # Create a bar chart for each data type
        cf.bar(transformed_data[data_type], stacked=True,height = 3.5,width = 5.3, markers=False, label='%', ex=f'Figure 4 - {data_type} Performance', sort=False, legend_loc = 'lower right')

    return

#Specify the models
models = ['Testing Sample','Claude Opus Testing','Testing Sample 35']

#Dictionary to map names to more intuitive phrases
model_map = {
    'Testing Sample':'GPT-4 Turbo',
'Claude Opus Testing':'Claude 3 Opus',
    'Testing Sample 35':'GPT-3.5 Turbo',

}

#Load pickle files for model metrics
perf_metrics = {}
for model in models:
    with open(os.path.join(config['processed_data'],'Model Output',model,'Performance Metrics.pkl'), 'rb') as handle:
        perf_metrics[model_map[model]] = pickle.load(handle)

#Make the charts
make_charts(perf_metrics)

##

def cleveland_dot_plot(perf_metrics, with_ambiguous=False):
    fig, ax = plt.subplots(figsize=(12, 8))

    ambiguous_key = 'With Ambiguous' if with_ambiguous else 'Without Ambiguous'

    questions = set()
    for category, ambiguous_types in perf_metrics.items():
        data_types = ambiguous_types[ambiguous_key]
        for data_type in ['Binary', 'Continuous']:
            if data_type in data_types:
                df = data_types[data_type]
                questions.update(df['Question'].unique())

    questions = sorted(list(questions))

    for i, question in enumerate(questions):
        for j, (category, ambiguous_types) in enumerate(perf_metrics.items()):
            data_types = ambiguous_types[ambiguous_key]
            for data_type in ['Binary', 'Continuous']:
                if data_type in data_types:
                    df = data_types[data_type]
                    if question in df['Question'].values:
                        metric = df.loc[df['Question'] == question, 'Alternative Performance Metric'].values[0]
                        if data_type == 'Continuous':
                            metric *= 100
                        ax.plot(i, metric, 'o', markersize=8, color=f'C{j}', label=category if i == 0 else '')

    ax.set_xticks(range(len(questions)))
    ax.set_xticklabels(questions, rotation=45, ha='right')
    ax.set_ylabel('Alternative Performance Metric')
    ax.set_title('Performance by Question')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.legend(unique_handles, unique_labels, loc='best')

    plt.tight_layout()
    plt.show()

cleveland_dot_plot(perf_metrics)