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
    'Question 17': 'Incentives or Mandates For Affordable Housing',
    'Question 17w': 'Do developers have to comply with the requirement\nto include affordable housing, however\ndefined, in their projects?',
    'Question 20': 'Is there a town-wide annual or biannual cap\non residential permits issued, and/or is\nproject phasing required?',
    'Question 21': 'How is lot area defined and how is the lot\nsize calculated?',
    'Question 22': 'What is the longest frontage requirement\nfor single family residential development\nin any district?',
    'Question 28Min': 'Minimum Residential Min Lot Size (Square Feet)',
    'Question 2': 'How many zoning districts, including overlays,\nare in the municipality?',
    'First_PC': 'First Principal Component (complexity)',
    'Second_PC': 'Second Principal Component (Strictness)',
    'Third_PC': 'Third principal component\n(bulk versus complexity)'
}



##Function to make bin scatters


def make_bin(ax, question, question_label, data, region=True, max_distance=50):
    if region:
        groups = data['Region'].unique()
        colors = ['#4E79A7', '#F28E2B', '#E15759', '#59A14F']  # Varied and aesthetically pleasing colors for regions



    else:
        groups = [None]
        colors = ['#FF5733']  # Single color for overall

    for idx, group in enumerate(groups):
        if group is not None:
            group_data = data[data['Region'] == group]
        else:
            group_data = data

        group_data = group_data[group_data['Miles to Metro Center'] <= max_distance]
        group_data = group_data.dropna(subset=['Miles to Metro Center', question])

        msa_dummies = pd.get_dummies(group_data['Nearest Metro Name'], drop_first=True)
        group_data = pd.concat([group_data, msa_dummies], axis=1)

        # Compute bin scatter with fixed effects
        w = msa_dummies.values
        est_fe = binsreg(y=question, x='Miles to Metro Center', data=group_data, noplot=True, polyreg=1)
        result_fe = est_fe.data_plot[0]
        dots_fe = pd.DataFrame(result_fe.dots)

        ax.scatter(dots_fe['x'], dots_fe['fit'], s=20, color=colors[idx % len(colors)], label=f'{group if group else "Overall"}')

        # Add dashed line plot from the 'poly' param
        poly_fe = pd.DataFrame(result_fe.poly)
        ax.plot(poly_fe['x'], poly_fe['fit'], linestyle='--', color=colors[idx % len(colors)])

    ax.set_xlabel('Miles to Metro Center')
    ax.set_ylabel(question_label)
    ax.set_title(question_label)
    ax.legend()



##
data = pd.read_excel(os.path.join(config['processed_data'], 'Model Output','augest_latest', 'Comprehensive Data.xlsx'))

data['Region'] = data['State'].str.upper().map(state_to_region)

#Keep rows where Miles_To_Metro_Center is not null
data = data[data['Miles to Metro Center'].notnull()]

##

# Create 2x2 grid of charts
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

questions_to_plot = ['Question 17', 'Question 28Min', 'First_PC', 'Second_PC']

for i, question in enumerate(questions_to_plot):
    row = i // 2
    col = i % 2
    question_label = question_mapping.get(question, question)
    make_bin(axs[row, col], question, question_label, data, region=True, max_distance = 50)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#Save figure to figures folder
plt.savefig(os.path.join(config['figures_path'], 'Figure X - Bin Scatter Dist From Center.png'))

plt.show()

##

# Create 1x2 grid of charts for Question 17 and Question 28Min
fig, axs = plt.subplots(1, 2, figsize=(2.5*4.15, 4.67))

questions_to_plot = ['Question 17', 'Question 28Min']

for i, question in enumerate(questions_to_plot):
    question_label = question_mapping.get(question, question)
    make_bin(axs[i], question, question_label, data, region=True)

plt.tight_layout()
# Save figure to figures folder
plt.savefig(os.path.join(config['figures_path'], 'Figure Y - Bin Scatter Dist From Center 1x2.png'), dpi = 300)

plt.show()
