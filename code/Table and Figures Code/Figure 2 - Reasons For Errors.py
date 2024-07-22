import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import plotly.graph_objects as go
import os
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

import sys
root = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning"
sys.path.append(root+r"\Embedding_Project\Compare Results\Chart Formatter")
import ChartFormatter as cf
cf.export = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\results\figures"

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
    }

# Load in the manual corrections
adjustments = pd.read_excel(os.path.join(config['raw_data'],'Error Adjustments', 'binary_adjustments.xlsx'))

# Map the unique categories to the desired names
category_map = {
    'Ambiguous': 'Ambiguous',
    'LLM wrong because interpreted context incorrectly': 'LLM Misinterpreted Context',
    'Pioneer wrong': 'Pioneer Study Outdated/Incorrect',
    'LLM wrong because did not receive the right context': 'LLM Missed Context'
}

# Replace the categories in the DataFrame with the mapped names
adjustments['Reason for Difference'] = adjustments['Reason for Difference'].map(category_map)

# Filter the DataFrame to include only the disagreed cases
disagreements = adjustments[adjustments['Whether Correct'] == 0]

# Create a pivot table to count the disagreements for each question and reason
pivot_data = disagreements.pivot_table(index='Question', columns='Reason for Difference', aggfunc='size', fill_value=0)

#Adjust index with short questions
pivot_data.index = pivot_data.index.map(lambda x: short_questions[str(x)])

#Order columns in order of size
pivot_data = pivot_data[pivot_data.sum().sort_values(ascending = False).index]

#Make bar chart
cf.bar(pivot_data, stacked = True, markers = False, height = 5, label = 'Count',rotate = 45, ex = 'Figure 2 - Reasons For Errors')


