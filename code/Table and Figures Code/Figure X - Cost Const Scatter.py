import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from binsreg import binsreg

##

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

##

state_to_region = {
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast', 'RI': 'Northeast', 'VT': 'Northeast',
    'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',
    'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'WI': 'Midwest',
    'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
    'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South', 'SC': 'South', 'VA': 'South', 'DC': 'South',
    'WV': 'South', 'AL': 'South', 'KY': 'South', 'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South', 'OK': 'South',
    'TX': 'South',
    'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West', 'NM': 'West', 'UT': 'West', 'WY': 'West',
    'AK': 'West', 'CA': 'West', 'HI': 'West', 'OR': 'West', 'WA': 'West'
}

question_mapping = {
    'Question 4': 'Is multi-family housing allowed, either by right\nor special permit (including through overlays\nor cluster zoning)?',
    'Question 5': 'Are apartments above commercial (mixed use)\nallowed in any district?',
    'Question 6': 'Is multi-family housing listed as allowed through\nconversion (of either single family homes or\nnon residential buildings)?',
    'Question 8': 'Are attached single family houses (townhouses,\n3+ units) listed as an allowed use (by right\nor special permit)?',
    'Question 9': 'Does zoning include any provisions for housing\nthat is restricted by age?',
    'Question 11': 'Are accessory or in-law apartments allowed\n(by right or special permit) in any district?',
    'Question 13': 'Is cluster development, planned unit development,\nopen space residential design, or another type\nof flexible zoning allowed by right?',
    'Question 14': 'Is cluster development, planned unit development,\nopen space residential design, or another type\nof flexible zoning allowed by special permit?',
    'Question 17': 'Does the zoning bylaw/ordinance include any\nmandates or incentives for development of\naffordable units?',
    'Question 17w': 'Do developers have to comply with the requirement\nto include affordable housing, however\ndefined, in their projects?',
    'Question 20': 'Is there a town-wide annual or biannual cap\non residential permits issued, and/or is\nproject phasing required?',
    'Question 21': 'How is lot area defined and how is the lot\nsize calculated?',
    'Question 22': 'What is the longest frontage requirement\nfor single family residential development\nin any district?',
    'Question 28Min': 'What is the minimum lot size for single-family\nhomes in each residential district? (Min)',
    'Question 2': 'How many zoning districts, including overlays,\nare in the municipality?',
    'First_PC': 'First principal component\n(complexity of zoning)',
    'Second_PC': 'Second principal component\n(stringency of zoning)',
    'Third_PC': 'Third principal component\n(bulk versus complexity)'
}

##


data = pd.read_excel(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\Model Output\latest_combined\Comprehensive Data.xlsx")

data['Region'] = data['State'].str.upper().map(state_to_region)

#Find columns that are questions or PCs
questions = [col for col in data.columns if col.startswith('Question') or '_PC' in col]


##
'''
Two key values are Median_Home_Value_2022 and All Unit Permits
'''
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import numpy as np
import pandas as pd

'''
Ensure that buckets line up with bins/quadrants
'''


def plot_percentile_heatmap_with_bars(data, x_col, y_col, hue_col=None, title_suffix='', num_buckets=4, label_map=None):
    # Drop rows with NaN values in the columns of interest
    data_filtered = data.dropna(subset=[x_col, y_col, hue_col])

    # Calculate percentile values
    data_filtered.loc[:, f'{x_col}_Percentile'] = rankdata(data_filtered[x_col], method='average') / len(data_filtered) * 100
    data_filtered.loc[:, f'{y_col}_Percentile'] = rankdata(data_filtered[y_col], method='average') / len(data_filtered) * 100

    if hue_col:
        data_filtered.loc[:, f'{label_map.get(hue_col,hue_col)}_Percentile'] = rankdata(data_filtered[hue_col], method='average') / len(data_filtered) * 100

    # Determine quadrants
    data_filtered.loc[:, 'Quadrant'] = np.where((data_filtered[f'{x_col}_Percentile'] > 50) & (data_filtered[f'{y_col}_Percentile'] > 50), 'Top-Right',
                                                np.where((data_filtered[f'{x_col}_Percentile'] <= 50) & (data_filtered[f'{y_col}_Percentile'] > 50), 'Top-Left',
                                                         np.where((data_filtered[f'{x_col}_Percentile'] <= 50) & (data_filtered[f'{y_col}_Percentile'] <= 50), 'Bottom-Left', 'Bottom-Right')))

    # Create grid with specified number of buckets
    bins = np.linspace(0, 100, num_buckets + 1)
    data_filtered['x_bin'] = pd.cut(data_filtered[f'{x_col}_Percentile'], bins=bins, include_lowest=True)
    data_filtered['y_bin'] = pd.cut(data_filtered[f'{y_col}_Percentile'], bins=bins, include_lowest=True, labels=bins[1:])

    # Calculate the mean hue_col value for each bin
    heatmap_data = data_filtered.groupby(['x_bin', 'y_bin'])[hue_col].mean().unstack().T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Heatmap of average values
    sns.heatmap(heatmap_data, cmap='Reds', annot=True, fmt='.1f', cbar_kws={'label': f'Average {label_map.get(hue_col,hue_col)} Value'}, ax=ax1)

    # Reverse the y-axis labels to match the quadrant positions correctly
    ax1.invert_yaxis()

    ax1.set_xlabel(label_map.get(x_col, x_col))
    ax1.set_ylabel(label_map.get(y_col, y_col))

    # Bar chart of averages for each quadrant
    if hue_col and data_filtered[hue_col].nunique() <= 10:
        sns.barplot(data=data_filtered, x='Quadrant', y=hue_col, hue=hue_col, estimator=np.mean, errorbar=None, palette='Set1', ax=ax2)
    else:
        sns.barplot(data=data_filtered, x='Quadrant', y=hue_col, estimator=np.mean, errorbar=None, color='red', ax=ax2)

    ax2.set_xlabel('Quadrant')
    ax2.set_ylabel(f'Average {label_map.get(hue_col,hue_col)} Value')
    ax2.set_title(f'Average {label_map.get(hue_col,hue_col)} by Quadrant {title_suffix}')
    if hue_col and data_filtered[hue_col].nunique() <= 10:
        ax2.legend(title=hue_col)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    #Save
    plt.savefig(os.path.join(config['figures_path'],f'Slides - Cost Construction Scatter-{hue_col}.png'), dpi = 300)

    plt.show()

label_map = {
    'All Unit Permits': 'Percentile of Units Permitted Per Capita (2021)',
    'Median_Home_Value_2022': 'Percentile of Median Home Value (2022)',
    'First_PC': 'First Principal Component (Complexity)',
    'Second_PC': 'Second Principal Component (Strictness)',
}

# Call the function to plot with coloring by First_PC (continuous variable)
plot_percentile_heatmap_with_bars(
    data,
    x_col='Median_Home_Value_2022',
    y_col='All Unit Permits',
    hue_col='First_PC',
    num_buckets=8,
    label_map=label_map
)
