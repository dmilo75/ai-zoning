
import pandas as pd
import pickle

import os
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Lod in the excel
df = pd.read_excel(os.path.join(config['processed_data'],'Model Output','latest_combined','Comprehensive Data.xlsx'))

# Load the questions Excel file
questions_df = pd.read_excel(os.path.join(config['raw_data'], 'Questions.xlsx'))

#Get question columns
question_columns = df.filter(regex='^Question').columns

# Extract question numbers from column names
question_numbers = [col.split()[1] for col in question_columns]

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
        '28Mean': 'Mean Min Lot Size',
        '28Min': 'Minimum Min Lot Size',
    '30': 'Mandatory Approval Steps',
    '31': 'Distinct Approval Bodies',
    '32': 'Public Hearing Requirements',
    '34': 'Max Review Waiting Time'
    }

# Filter short_questions for questions in the dataset
short_questions = {k: v for k, v in short_questions.items() if
                   k in question_numbers}

# Create a mapping table for short questions to full questions
mapping_data = []
for question_number, short_q in short_questions.items():
    # Handle special cases for 28Mean and 28Min
    if question_number in ['28Mean', '28Min']:
        question_id = 28
    else:
        question_id = int(question_number)

    full_q = questions_df[questions_df['ID'] == question_id]['Pioneer Question'].values
    if len(full_q) > 0:
        mapping_data.append([full_q[0], short_q])

mapping_df = pd.DataFrame(mapping_data, columns=['Full Question', 'Short Question'])

# Create LaTeX
with pd.option_context("max_colwidth", 1000):
    latex = mapping_df.to_latex(index=False, column_format='p{16cm}p{6cm}')

# Export LaTeX
with open(os.path.join(config['tables_path'], 'latex', 'Appendix - Short to Full Question Mapping.tex'), 'w') as f:
    f.write(latex)

print("Mapping table created and exported to LaTeX file.")