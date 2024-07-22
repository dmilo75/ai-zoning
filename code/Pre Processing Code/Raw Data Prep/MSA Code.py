import pandas as pd
import os
import yaml
import helper_functions as hf

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

#Read in sample data
sample = pd.read_excel(os.path.join(raw_path,"Sample Data.xlsx"),index_col  = 0)

# Initialize FIPS_COUNTY_ADJ with the same values as FIPS_COUNTY
sample['FIPS_COUNTY_ADJ'] = sample['FIPS_COUNTY']

# Apply the update function only to rows where 'State' == 'ct'
sample.loc[sample['State'] == 'ct'] = sample[sample['State'] == 'ct'].apply(hf.update_fips_adj, axis=1)

#%%Get info on MSA and region. Probably need county info too for MSA.

msa_crosswalk = pd.read_excel(os.path.join(raw_path,"msa_crosswalk_mar_2020.xlsx"))


#Make flag for if MSA
msa_crosswalk = msa_crosswalk[msa_crosswalk['Metropolitan/Micropolitan Statistical Area'] == 'Metropolitan Statistical Area']
msa_crosswalk = msa_crosswalk.drop(columns = 'Metropolitan/Micropolitan Statistical Area')
msa_crosswalk['MSA'] = True

#Keep 'MSA', 'Central/Outlying County', 'FIPS County Code', 'FIPS State Code', and 'CSA Title'
msa_crosswalk = msa_crosswalk[['MSA','Central/Outlying County','FIPS County Code','FIPS State Code','CSA Title']]

#Now merge with sample data
sample_msa = sample.copy()
sample_msa = pd.merge(sample_msa,msa_crosswalk,left_on = ['FIPS_COUNTY_ADJ','FIPS_STATE'], right_on = ['FIPS County Code','FIPS State Code'], how = 'left')

#Clean
sample_msa.drop(columns = ['FIPS County Code','FIPS State Code'])
msa_crosswalk['MSA'] = msa_crosswalk['MSA'].fillna(False)

#Just keep new columns and 'CENSUS_ID_PID6'
sample_msa = sample_msa[['CENSUS_ID_PID6','MSA','Central/Outlying County','CSA Title']]

#Save to 'interim_data' under processed data folder
sample_msa.to_excel(os.path.join(data_path,'interim_data','MSA Data.xlsx'))
