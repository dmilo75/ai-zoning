# -*- coding: utf-8 -*-

"""
The code is to merge the Wharton dataset and our dataset for further analysis
"""

import pandas as pd
import yaml
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

dataset = 'Full Run April 24 Wharton Qs'

# path
wharton_data = pd.read_stata(os.path.join(config['raw_data'], 'WRLURI_01_15_2020.dta'))
census_of_gov = pd.read_excel(os.path.join(config['raw_data'], 'census_of_gov.xlsx'))
our_model = pd.read_excel(os.path.join(config['processed_data'],'Model Output',dataset,'Light Data.xlsx'))

"""
Because in different Excel censes_id_pid6 is called difference (like fips code or geoid)
we will call it geoid in below
"""
# Merged the Wharton data with censes_id_pid6

# Convert 'fipsplacecode18' and 'statecode' to integers
wharton_data['fipsplacecode18'] = wharton_data['fipsplacecode18'].astype(int)
wharton_data['statecode'] = wharton_data['statecode'].astype(int)

# Convert 'FIPS_PLACE' and 'FIPS_STATE' to integers in census_of_gov
census_of_gov['FIPS_PLACE'] = census_of_gov['FIPS_PLACE'].astype(int)
census_of_gov['FIPS_STATE'] = census_of_gov['FIPS_STATE'].astype(int)

# Create a MultiIndex on census_of_gov for faster lookup
census_of_gov.set_index(['FIPS_PLACE', 'FIPS_STATE'], inplace=True)

# Use map() to find the corresponding CENSUS_ID_PID6 for each row in wharton_data
wharton_data['geoid'] = wharton_data.apply(lambda row: census_of_gov.loc[(row['fipsplacecode18'], row['statecode']), 'CENSUS_ID_PID6'] if (row['fipsplacecode18'],row['statecode']) in census_of_gov.index else "NaN", axis=1)

questions = list(set(our_model["Question"]))##Questions No. in our dataset

#Rename CENSUS_ID_PID6 to geoid
our_model = our_model.rename(columns = {'CENSUS_ID_PID6':'geoid'})

#adjust the format of our data in to Wharton format
#edited_full run is the reshaped full_run
for i in questions:
  if i == questions[0]:
    edited_our_model = our_model[our_model["Question"]==i]
    edited_our_model = edited_our_model.rename({"Answer":"A2", "Question":"Q2"}, axis='columns')
  else:
      question_filtered = our_model[our_model["Question"]==i]
      question_filtered = question_filtered[["Answer","Question","geoid"]]
      question_filtered = question_filtered.rename({"Answer":"A"+str(i), "Question":"Q"+str(i)}, axis='columns')
    ##For reading, I change Answer into A, and Question in to Q
      edited_our_model = pd.merge(edited_our_model, question_filtered, on ="geoid")

# merged_df is the merged DataFrame of Wharton dataser and our dataset
# The size of merged_df is 1283
# It only contained the geoid, state and muni name, wharton index and answers of our questions and wharton surveys
merged_df = pd.merge(edited_our_model, wharton_data[["geoid","WRLURI18", 'q118', 'q218', 'q3a18', 'q3b18', 'q3c18', 'q3d18', 'q3e18', 'q3f18', 'q3a_other18', 'q4_1a18', 'q4_1b18', 'q4_1c18', 'q4_1d18', 'q4_1e18', 'q4_1f18', 'q4_1g18', 'q4_1h18', 'q4_1i18', 'q4_1j18', 'q4a_other18', 'q4_2k18', 'q4_2l18', 'q4_2m18', 'q4_2n18', 'q4_2o18', 'q4_2p18', 'q4_2q18', 'q4_2r18', 'q4_2s18', 'q4_2t18', 'q4b_other18', 'q5a18', 'q5b18', 'q5a_m5018', 'q5b_multi18', 'q618', 'q718', 'q7a18', 'q7b18', 'q8a18', 'q8b18', 'q8c18', 'q8d18', 'q8e18', 'q8f18', 'q9a18', 'q9b18', 'q9c18', 'q1018', 'q11a18', 'q11b18', 'q11c18', 'q11d18', 'q12a_subm18', 'q12a_appr18', 'q12b_subm18', 'q12b_appr18', 'q1318', 'q14a18', 'q14b18', 'q1518', 'q16a118', 'q16b118', 'q16a218', 'q16b218', 'q17a118', 'q17a218', 'q17b118', 'q17b218', 'q18a18', 'q18b18', 'q1918', 'q19_rezoning18', 'q20a18', 'q20b18', 'q20c18', 'q2118', 'q21_subdivision18', 'q22a18', 'q22b18', 'q22c18']], on = "geoid")
merged_df.replace({'Yes': 1, 'No': 0}, inplace=True) #exchange yes and no to 1/0

merged_df.to_excel(os.path.join(config['processed_data'],'Wharton Comparison','edited_full_run_with_wharton_index_2018.xlsx'))

print(merged_df)

