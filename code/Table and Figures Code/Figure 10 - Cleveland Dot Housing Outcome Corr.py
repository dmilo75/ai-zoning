import pandas as pd
import os
import yaml
from scipy.stats import mstats
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

os.environ['MPLCONFIGDIR'] = config['raw_data']
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model = 'latest_combined'

questions = pd.read_excel(os.path.join(config['raw_data'],"Questions.xlsx"))

#Cleaned output
df = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,"Comprehensive Data.xlsx"))

##

question_dict = {str(k): v for k, v in questions.set_index('ID')['Pioneer Question'].to_dict().items()}
question_dict['28Min'] = 'Minimum Residential Min Lot Size'
question_dict['28Mean'] = 'Mean Residential Min Lot Size'

def question_name(q):
    q = str(q)
    return question_dict.get(q, q)

def clean_answer(answer):
    if isinstance(answer, (int, float)):
        return answer
    elif answer == 'Yes':
        return 1
    elif answer == 'No':
        return 0
    else:
        return np.nan
def calculate_correlation(df):
    # List of columns to calculate correlation for
    cols_to_corr = [
        'Median_Gross_Rent_2022',
        'Median_Gross_Rent_Percent_Change',
        'Median_Home_Value_2022',
        'Median_Home_Value_Percent_Change',
        'Single Unit Permits',
        'Multi-Unit Permits',
        'All Unit Permits'
    ]

    # List of zoning questions
    zoning_questions = [f'Question {i}' for i in questions['ID'].tolist()] + ['Question 28Min', 'Question 28Mean']

    correlations = {}

    for col in cols_to_corr:
        correlations[col] = {}
        for question in zoning_questions:
            if question in df.columns:
                correlations[col][question] = df[question].corr(df[col])

    return pd.DataFrame(correlations)

# Calculate correlations for the sample data
correlations = calculate_correlation(df)

# Update index
correlations.index = correlations.index.to_series().apply(lambda x: question_name(x.replace('Question ', '')))

rename_map = {
    'Median_Gross_Rent_2022': 'Median Gross Rent\n(2022)',
    'Median_Gross_Rent_Percent_Change': 'Median Gross Rent\n% Change, 2022-2010',
    'Median_Home_Value_2022': 'Median Home Value\n2022',
    'Median_Home_Value_Percent_Change': 'Median Home Value\n% Change, 2022-2010',
    'Single Unit Permits': 'Building Permits\nSingle Unit',
    'Multi-Unit Permits': 'Building Permits\nMulti-Unit',
    'All Unit Permits': 'Building Permits\nAll Units'
}


def plot_correlations(correlations, sort_by_column=None, save_filename=None):
    correlations = correlations.rename(columns=rename_map)

    if sort_by_column:
        sort_by_column = rename_map[sort_by_column]
        columns_order = [sort_by_column] + [col for col in correlations.columns if col != sort_by_column]
        correlations = correlations[columns_order]

    fontsize = 18

    df_melted = correlations.reset_index().melt(id_vars='index', value_vars=correlations.columns)
    df_melted.columns = ['Question', 'Metric', 'Correlation']

    if sort_by_column:
        sort_data = df_melted[df_melted['Metric'] == sort_by_column].sort_values('Correlation', ascending=False)
        sort_order = sort_data['Question'].tolist()
    else:
        sort_order = df_melted['Question'].drop_duplicates().tolist()

    palette = sns.color_palette(["#303F9F", "#D2691E", "#BDC3C7", "#A2D149"])

    plt.figure(figsize=(20, len(sort_order) * 1.1), dpi=100)
    ax = sns.stripplot(data=df_melted, x='Correlation', y='Question', hue='Metric', jitter=False, size=20, alpha=0.5,
                       order=sort_order, palette=palette)

    plt.axvline(0, color='grey', linestyle='--')

    ax.set_yticklabels(ax.get_yticklabels(), wrap=True, fontsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_xlabel('Correlation', fontsize=fontsize)
    ax.set_ylabel('', fontsize=fontsize)

    plt.legend(loc='lower right', ncol=1, fontsize=fontsize, edgecolor='none')

    plt.tight_layout(rect=[0.35, 0, 1, 1])

    if save_filename:
        save_path = os.path.join(config['figures_path'], save_filename)
        plt.savefig(save_path, dpi=300)

    plt.show()

plot_correlations(correlations.copy()[['Median_Gross_Rent_2022', 'Median_Home_Value_2022',
                                       'All Unit Permits']],
                  sort_by_column='Median_Home_Value_2022',
                  save_filename='Cleveland Dot1.png')

plot_correlations(correlations.copy().drop(
    columns=['Median_Gross_Rent_2022', 'Median_Home_Value_2022',
                                       'All Unit Permits']),
                  sort_by_column='Median_Home_Value_Percent_Change',
                  save_filename='Cleveland Dot2.png')