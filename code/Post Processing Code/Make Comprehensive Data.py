
'''
We want dataset where each row is a muni and columns are questions and indices

'''

import pandas as pd
import numpy as np

import os
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model = 'latest_combined'

##Load in summary data enriched
sample_enriched = pd.read_excel(os.path.join(config['processed_data'],'Sample_Enriched.xlsx'), index_col = 0)

##Load in answers
answers = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,'Light Data.xlsx'))

#Pivot with questions as column, 'CENSUS_ID_PID6' as index and 'Answer' as values
pivot_df = answers.pivot(index = 'CENSUS_ID_PID6', columns = 'Question', values = 'Answer')

#Add prefix 'Question ' to columns
pivot_df.columns = ['Question ' + str(col) for col in pivot_df.columns]

#Reset index and name it 'CENSUS_ID_PID6'
pivot_df = pivot_df.reset_index()

##Load in PCA results
pca_results = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,'Overall_Index.xlsx'))

#Extract CENSUS_ID_PID6 from 'Muni' as in between 1st and 2nd '#'
pca_results['CENSUS_ID_PID6'] = pca_results['Muni'].str.extract(r'#(.*?)#')

#Keep census id and First_Principal_Component, Second_Principal_Component, Third_Principal_Component
pca_results = pca_results[['CENSUS_ID_PID6','First_Principal_Component','Second_Principal_Component','Third_Principal_Component']]

#Rename principal component columns to shorter names
pca_results.columns = ['CENSUS_ID_PID6','First_PC','Second_PC','Third_PC']

##Merge in on CENSUS_ID_PID6

#Ensure CENSUS_ID_PID6 is type int in each dataframe
pivot_df['CENSUS_ID_PID6'] = pivot_df['CENSUS_ID_PID6'].astype(int)
pca_results['CENSUS_ID_PID6'] = pca_results['CENSUS_ID_PID6'].astype(int)
sample_enriched['CENSUS_ID_PID6'] = sample_enriched['CENSUS_ID_PID6'].astype(int)

merged_results = pivot_df.merge(pca_results, on = 'CENSUS_ID_PID6')

final_merged = sample_enriched.merge(merged_results, on = 'CENSUS_ID_PID6')

#Find each column in lhs_vars that starts with 'Question'
questions = [col for col in final_merged.columns if col.startswith('Question')]

#Columns to winsorize
winsorize_cols = ['Question 22', 'Question 28Max', 'Question 28Mean', 'Question 28Min','Question 34']

#Values above 99th percentile make null
for col in winsorize_cols:
    #Calcualte 99th percentile
    upper_limit = final_merged[col].quantile(0.99)
    #Replace values above 99th percentile with null
    final_merged[col] = np.where(final_merged[col] > upper_limit, np.nan, final_merged[col])

def clean_bin(x):
    if x == 'Yes':
        return 1
    elif x == 'No':
        return 0
    else:
        return x

for question in questions:
    if question not in ['Question_2']+winsorize_cols:
        #Turn Yes into 1 and No into 0
        final_merged[question] = final_merged[question].apply(lambda x: clean_bin(x))

#Export to Excel
final_merged.to_excel(os.path.join(config['processed_data'],'Model Output',model,'Comprehensive Data.xlsx'), index = False)

