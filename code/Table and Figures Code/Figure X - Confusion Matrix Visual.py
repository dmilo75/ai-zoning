import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os
import numpy as np
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model = 'Testing Sample'

short_questions = {
        '1': 'Bylaw Online Availability',
        '2': 'Zoning District Count',
        '3': 'Multifamily By Right',
        '4': 'Multifamily Allowed',
        '5': 'Mixed-Use Buildings',
        '6': 'Conversion To Multifamily',
        '7': 'Multifamily Permit Authority',
        '8': 'Townhouses Allowed',
        '9': 'Age-Restricted Provisions',
        '10': 'Age-Restricted Developments Built',
        '11': 'Accessory Apartments Allowed',
        '12': 'Accessory Apartment Authority',
        '13': 'Flexible Zoning By Right',
        '14': 'Flexible Zoning By Permit',
        '15': 'Flexible Zoning Authority',
        '16': 'Flexible Zoning Built',
        '17': 'Affordable Housing',
        '18': 'Inclusionary Zoning Adopted',
        '19': 'Affordable Units Built',
        '20': 'Permit Cap Or Phasing',
        '21': 'Wetlands Restricted in Lot Size Calc',
        '22': 'Longest Frontage Requirement',
        '23': 'Frontage Measurement',
        '24': 'Lot Shape Requirements',
        '25': 'Height Measurement',
        '26': 'Additional Zoning Notes',
        '27Mean': 'Mean Min Lot Size',
        '27Min': 'Minimum Min Lot Size',
        '28Mean': 'Mean Residential Min Lot Size',
        'Total': 'Total',
    }

#Draw in merged data with Pioneer
df = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,'Light Data Merged Adjusted.xlsx'))

#Filter for only 'Answer' in 'Yes' and 'No', so we drop 'I don't know' and non-binary questions
df = df[df['Answer'].isin(['Yes','No'])]

# Initialize an empty DataFrame to store confusion matrix parameters for each question
conf_matrix_df = pd.DataFrame(columns=['True Positive', 'False Positive', 'True Negative', 'False Negative'])

# Loop through each question to calculate confusion matrix parameters
for question in df['Question'].unique():
    temp_df = df[df['Question'] == question]
    tp = sum((temp_df['Answer'] == 'Yes') & (temp_df['Answer_pioneer'] == 'Yes'))
    fp = sum((temp_df['Answer'] == 'Yes') & (temp_df['Answer_pioneer'] == 'No'))
    tn = sum((temp_df['Answer'] == 'No') & (temp_df['Answer_pioneer'] == 'No'))
    fn = sum((temp_df['Answer'] == 'No') & (temp_df['Answer_pioneer'] == 'Yes'))
    conf_matrix_df.loc[question] = [tp, fp, tn, fn]

#Make a 'Total' row
conf_matrix_df.loc['Total'] = conf_matrix_df.sum()

# Calculate TPR, FPR, Precision, and Recall for the confusion matrix
conf_matrix_df['True Positive Rate'] = (conf_matrix_df['True Positive'] / (conf_matrix_df['True Positive'] + conf_matrix_df['False Negative']))
conf_matrix_df['False Positive Rate'] = conf_matrix_df['False Positive'] / (conf_matrix_df['False Positive'] + conf_matrix_df['True Negative'])
conf_matrix_df['Precision'] = conf_matrix_df['True Positive'] / (conf_matrix_df['True Positive'] + conf_matrix_df['False Positive'])

# Handle division by zero cases
conf_matrix_df.replace([np.inf, -np.inf], np.nan, inplace=True)
conf_matrix_df.fillna(0, inplace=True)  # Replace NaN with 0 for plotting


#Adjust index
conf_matrix_df.index = conf_matrix_df.index.map(lambda x: short_questions[str(x)])

#Make index name be 'Question'
conf_matrix_df.index.name = 'Question'

#Get latex
latex = conf_matrix_df.to_latex(float_format="%.2f")

#Export latex
with open(os.path.join(config['tables_path'],'latex','confusion_matrix.tex'), 'w') as f:
    f.write(latex)
