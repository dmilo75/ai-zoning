import pandas as pd
import os
import yaml

# Load filepaths
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

#Connecticut county bridge (Some external datasets use counties and others use regional bodies in place of counties for Connecticut. We will have both ready.)
#Source: https://www1.ctdol.state.ct.us/lmi/misc/counties.asp
#Source2: https://unicede.air-worldwide.com/unicede/unicede_connecticut_fips_2.html
ct_map = {
    'norwich': {'County Name': 'New London County', 'County FIPS': 11},
    'seymour': {'County Name': 'New Haven County', 'County FIPS': 9},
    'madison': {'County Name': 'New Haven County', 'County FIPS': 9},
    'shelton': {'County Name': 'Fairfield County', 'County FIPS': 1},
    'bloomfield': {'County Name': 'Hartford County', 'County FIPS': 3},
    'enfield': {'County Name': 'Hartford County', 'County FIPS': 3},
    'vernon': {'County Name': 'Tolland County', 'County FIPS': 13},
    'southbury': {'County Name': 'New Haven County', 'County FIPS': 9},
    'avon': {'County Name': 'Hartford County', 'County FIPS': 3},
    'new haven': {'County Name': 'New Haven County', 'County FIPS': 9},
    'naugatuck': {'County Name': 'New Haven County', 'County FIPS': 9},
    'new britain': {'County Name': 'Hartford County', 'County FIPS': 3},
    'hartford': {'County Name': 'Hartford County', 'County FIPS': 3}
}

#%%Pull in 'sample' data 

#Read in sample data
sample = pd.read_excel(os.path.join(raw_path,"Sample Data.xlsx"),index_col  = 0)

# Initialize FIPS_COUNTY_ADJ with the same values as FIPS_COUNTY
sample['FIPS_COUNTY_ADJ'] = sample['FIPS_COUNTY']

# Define a function to update the FIPS_COUNTY_ADJ based on the ct_map dictionary
def update_fips_adj(row):
    muni = row['Muni'].lower()  # Ensure case-insensitive matching
    if muni in ct_map:
        row['FIPS_COUNTY_ADJ'] = ct_map[muni]['County FIPS']
    return row

# Apply the update function only to rows where 'State' == 'ct'
sample.loc[sample['State'] == 'ct'] = sample[sample['State'] == 'ct'].apply(update_fips_adj, axis=1)

# Print adjusted rows
print(sample[sample['State'] == 'ct'][['Muni', 'FIPS_COUNTY', 'FIPS_COUNTY_ADJ']])


#%%ACS Merging code

acs_vars = ['Median Household Income','Median Gross Rent','Median Home Value (Owner Occupied)']

result_df = sample.copy()

final_df = sample.copy()

for year in [2010, 2021]:
    dfs = []  

    for geo in ['place', 'county subdivision']:
        merged_data = pd.read_csv(os.path.join(raw_path,'ACS Data\\'+geo+' '+str(year)+'.csv'))

        # Define merge columns and criteria based on geography
        if geo == 'county subdivision':
            merge_on_left = ['FIPS_STATE', 'FIPS_COUNTY_ADJ', 'FIPS_PLACE']
            merge_on_right = ['state', 'county', 'county subdivision']
            criteria = result_df['UNIT_TYPE'] == '3 - TOWNSHIP'
        else:
            merge_on_left = ['FIPS_STATE', 'FIPS_PLACE']
            merge_on_right = ['state', 'place']
            criteria = result_df['UNIT_TYPE'] == '2 - MUNICIPAL'

        # Rename columns in merged_data that are not in merge_on_right
        for col in merged_data.columns:
            if col not in merge_on_right:
                merged_data.rename(columns={col: f"{col}_{str(year)}"}, inplace=True)

        # Perform the merge
        merged_df = pd.merge(
            result_df[criteria], 
            merged_data,
            left_on=merge_on_left, 
            right_on=merge_on_right,
            how='left'
        )

        # Drop merge_on_right columns
        merged_df.drop(columns=merge_on_right, inplace=True)

        # Calculate merge stats
        matched_columns = [col for col in merged_df.columns if f"_{str(year)}" in col]
        matched = merged_df[matched_columns].dropna(how='all').shape[0]
        total = merged_df.shape[0]

        # Display merge stats
        print(f"For {geo} in {year}, {matched/total:.2%} matched.")

        # Identify rows that didn't match
        unmatched_rows = merged_df[merged_df[matched_columns].isna().any(axis=1)]
        if not unmatched_rows.empty:
            print(f"Rows that didn't match for {geo} in {year}:\n", unmatched_rows[['FIPS_STATE', 'FIPS_PLACE']])

        dfs.append(merged_df)

    # Concatenate both DataFrames for 'place' and 'county subdivision' together
    concatenated_df = pd.concat(dfs, ignore_index=True)
    
    final_df = pd.merge(final_df,concatenated_df[['FIPS_PLACE','FIPS_STATE','FIPS_COUNTY_ADJ']+matched_columns], on = ['FIPS_PLACE','FIPS_STATE','FIPS_COUNTY_ADJ'])
    
    
    
#%%Print out munis that didn't match at all

# Select columns ending in "_2021" or "_2010"
cols_to_check = [col for col in final_df.columns if col.endswith('_2021') or col.endswith('_2010')]

# Filter rows where all values in the selected columns are NaN
null_rows = final_df[final_df[cols_to_check].isnull().all(axis=1)]

# Print these rows
print(null_rows)



#%%Make change variables and export 

#Make change vars
for var in acs_vars:
    final_df[var+"_Percent_Change"] = ((final_df[var+"_2021"] - final_df[var+"_2010"]) / final_df[var+"_2010"]) * 100


acs_data = final_df.copy()



#%%Merge in building permits data

bps_data = pd.read_excel(os.path.join(raw_path,"bps_raw.xlsx"))

sample_perm = sample.copy()


sample_perm = pd.merge(sample_perm, bps_data, 
                       left_on=['FIPS_PLACE', 'FIPS_STATE'], 
                       right_on=['FIPS Place', 'FIPS State'], 
                       how='left')

# Count the number of non-NaN entries in 'FIPS Place' after the merge
merged_count = sample_perm['Number of Months Reported'].notna().sum()

# Calculate the percentage merged
percentage_merged = (merged_count / len(sample_perm)) * 100

print(f"Percentage merged: {percentage_merged:.2f}%")

#%%Keep only relevant variables now

variables = [
    'Number of Months Reported',
    '1-unit Buildings (Estimates with Imputation)', '1-unit Units (Estimates with Imputation)', '1-unit Valuation (Estimates with Imputation)',
    '2-units Buildings (Estimates with Imputation)', '2-units Units (Estimates with Imputation)', '2-units Valuation (Estimates with Imputation)',
    '3-4 units Buildings (Estimates with Imputation)', '3-4 units Units (Estimates with Imputation)', '3-4 units Valuation (Estimates with Imputation)',
    '5+ units Buildings (Estimates with Imputation)', '5+ units Units (Estimates with Imputation)', '5+ units Valuation (Estimates with Imputation)',
    '1-unit Buildings (Reported Only)', '1-unit Units (Reported Only)', '1-unit Valuation (Reported Only)', '2-units Buildings (Reported Only)',
    '2-units Units (Reported Only)', '2-units Valuation (Reported Only)', '3-4 units Buildings (Reported Only)', '3-4 units Units (Reported Only)',
    '3-4 units Valuation (Reported Only)', '5+ units Buildings (Reported Only)', '5+ units Units (Reported Only)', '5+ units Valuation (Reported Only)'
]


# Create 'Single Unit Permits'
sample_perm['Single Unit Permits'] = sample_perm['1-unit Units (Estimates with Imputation)']

# Create 'Multi-Unit Permits'
sample_perm['Multi-Unit Permits'] = (sample_perm['2-units Units (Estimates with Imputation)'] + 
                                     sample_perm['3-4 units Units (Estimates with Imputation)'] +
                                     sample_perm['5+ units Units (Estimates with Imputation)'])

# Create 'All Unit Permits'
sample_perm['All Unit Permits'] = sample_perm['Single Unit Permits'] + sample_perm['Multi-Unit Permits']

# Drop the original variables from the list `variables`
sample_perm = sample_perm.drop(columns=variables)

#%%Get info on MSA and region. Probably need county info too for MSA. 

msa_crosswalk = pd.read_excel(os.path.join(raw_path,"msa_crosswalk_mar_2020.xlsx"))

#Only keep relevant columns
msa_crosswalk = msa_crosswalk[['Metropolitan/Micropolitan Statistical Area','FIPS State Code','FIPS County Code']].dropna()

#Make flag for if MSA
msa_crosswalk = msa_crosswalk[msa_crosswalk['Metropolitan/Micropolitan Statistical Area'] == 'Metropolitan Statistical Area']
msa_crosswalk = msa_crosswalk.drop(columns = 'Metropolitan/Micropolitan Statistical Area')
msa_crosswalk['MSA'] = True

#Now merge with sample data
sample_msa = sample.copy()
sample_msa = pd.merge(sample_msa,msa_crosswalk,left_on = ['FIPS_COUNTY_ADJ','FIPS_STATE'], right_on = ['FIPS County Code','FIPS State Code'], how = 'left')

#Clean
sample_msa.drop(columns = ['FIPS County Code','FIPS State Code'])
msa_crosswalk['MSA'] = msa_crosswalk['MSA'].fillna(False)



#%%Merge all of the datasets together and make a enriched sample data dataframe

urban_data = pd.read_excel(os.path.join(raw_path,"urban_raw.xlsx"))

# Merge the DataFrames
merged_df1 = pd.merge(sample_msa, urban_data, on=['FIPS_PLACE', 'FIPS_STATE'], how='inner', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
merged_df2 = pd.merge(merged_df1, final_df, on=['FIPS_PLACE', 'FIPS_STATE'], how='inner', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
final_merged_df = pd.merge(merged_df2, sample_perm, on=['FIPS_PLACE', 'FIPS_STATE'], how='inner', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

# Export so we have a dataframe with all the data
final_merged_df.to_excel(os.path.join(data_path,"Sample_Enriched.xlsx"))


