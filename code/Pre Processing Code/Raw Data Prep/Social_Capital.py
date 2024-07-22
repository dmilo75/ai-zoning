import pandas as pd
import os
import yaml
import helper_functions as hf
import numpy as np

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

# Read in sample data
sample = pd.read_excel(os.path.join(raw_path, "Sample Data.xlsx"), index_col=0)

# Initialize FIPS_COUNTY_ADJ with the same values as FIPS_COUNTY
sample['FIPS_COUNTY_ADJ'] = sample['FIPS_COUNTY']

# Apply the update function only to rows where 'State' == 'ct'
sample.loc[sample['State'] == 'ct'] = sample[sample['State'] == 'ct'].apply(hf.update_fips_adj, axis=1)

#Make FIPS_STATE a two digit string
sample['FIPS_STATE'] = sample['FIPS_STATE'].astype(str).str.zfill(2)

#Make FIPS_PLACE a five digit string
sample['FIPS_PLACE'] = sample['FIPS_PLACE'].astype(str).str.zfill(5)

# Read in social capital data
df = pd.read_csv(os.path.join(raw_path, 'Chetty Data', 'social_capital_zip.csv'))

# Zero-pad zip and keep as string
df['zip'] = df['zip'].astype(int).astype(str).str.zfill(5)

# Extract state from county, ensuring proper zero-padding
df['county_str'] = df['county'].fillna(0).astype(int).astype(str).str.zfill(5)
df['state'] = df['county_str'].str[:2]

# Identify the columns to aggregate (all except 'zip' and 'county')
columns_to_aggregate = [col for col in df.columns if col not in ['zip', 'county', 'state', 'county_str']]

# Check for duplicates in social capital data
print("Duplicates in social capital data (zip-state):")
print(df[df.duplicated(subset=['zip', 'state'], keep=False)])

# Read in geo bridge for places
geo_bridge_place = pd.read_csv(os.path.join(raw_path, 'GEO Crosswalks', 'geocorr2018_zip_place.csv'), encoding='ISO-8859-1')

# Drop first row
geo_bridge_place = geo_bridge_place.drop(0)

# Ensure zcta5 is string type (it should already be zero-padded)
geo_bridge_place['zcta5'] = geo_bridge_place['zcta5'].astype(str)

# Make state two digit zero padded string
geo_bridge_place['state'] = geo_bridge_place['state'].astype(str).str.zfill(2)

# Merge social capital data with geo bridge
df_merged = pd.merge(df, geo_bridge_place, left_on=['zip', 'state'], right_on=['zcta5', 'state'], how='outer', indicator=True)

# Keep only the successfully merged rows
df_merged = df_merged[df_merged['_merge'] == 'both'].drop('_merge', axis=1)


# Function to calculate weighted average, ignoring NaN values
def weighted_average_ignore_nan(df, group_cols, value_cols, weight_col):
    # Ensure weight column is numeric
    df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')

    # Dictionary to store weighted average results
    weighted_avgs = {}

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create a mask to filter out rows where the column value is NaN
        mask = df[col].notna()

        # Multiply the non-NaN value columns by the weight column
        df[col + '_weighted'] = df.loc[mask, col] * df.loc[mask, weight_col]

        # Group by specified columns and sum the weighted values and the weights
        grouped_weighted = df.loc[mask].groupby(group_cols)[col + '_weighted'].sum()
        grouped_weights = df.loc[mask].groupby(group_cols)[weight_col].sum()

        # Divide the sum of weighted values by the sum of weights to get the weighted average
        weighted_avg = grouped_weighted / grouped_weights

        weighted_avgs[col] = weighted_avg

    # Combine all the weighted averages into a single DataFrame
    result = pd.DataFrame(weighted_avgs).reset_index()

    return result

# Ensure pop10 column is converted to numeric to avoid type issues during the weighted average calculation
df_merged['pop10'] = pd.to_numeric(df_merged['pop10'], errors='coerce')

#Make sure placefp is a five digit string
df_merged['placefp'] = df_merged['placefp'].astype(str).str.zfill(5)

# Calculate weighted averages for place data
place_data = weighted_average_ignore_nan(df_merged, ['state', 'placefp'], columns_to_aggregate, 'pop10')

# Merge with sample data
merged_place = pd.merge(sample[sample['UNIT_TYPE'] == '2 - MUNICIPAL'], place_data, how='left', left_on=['FIPS_PLACE', 'FIPS_STATE'], right_on=['placefp', 'state'])

# Read in geo bridge for county subdivisions
geo_bridge_cousub = pd.read_csv(os.path.join(raw_path, 'GEO Crosswalks', 'geocorr2018_zip_cousub.csv'), encoding='ISO-8859-1')

#Drop first row
geo_bridge_cousub = geo_bridge_cousub.drop(0)

# Ensure zcta5 is string type (it should already be zero-padded)
geo_bridge_cousub['zcta5'] = geo_bridge_cousub['zcta5'].astype(str)

#Make county a five digit zero padded string
geo_bridge_cousub['county'] = geo_bridge_cousub['county'].astype(str).str.zfill(5)

#Make state as first two digits
geo_bridge_cousub['state'] = geo_bridge_cousub['county'].str[:2]

# Merge social capital data with geo bridge for county subdivisions
df_merged_cousub = pd.merge(df, geo_bridge_cousub, left_on=['zip', 'state'], right_on=['zcta5', 'state'], how='outer', indicator=True)

# Identify rows that didn't merge for county subdivisions
left_only_cousub = df_merged_cousub[df_merged_cousub['_merge'] == 'left_only']
right_only_cousub = df_merged_cousub[df_merged_cousub['_merge'] == 'right_only']

print(f"\nNumber of rows in social capital data that didn't merge with county subdivisions: {len(left_only_cousub)}")
print(f"Number of rows in geo bridge data (county subdivisions) that didn't merge: {len(right_only_cousub)}")

# Keep only the successfully merged rows
df_merged_cousub = df_merged_cousub[df_merged_cousub['_merge'] == 'both'].drop('_merge', axis=1)

# Ensure pop10 column is converted to numeric to avoid type issues during the weighted average calculation
df_merged_cousub['pop10'] = pd.to_numeric(df_merged_cousub['pop10'], errors='coerce')

# Calculate weighted averages for county subdivision data
cousub_data = weighted_average_ignore_nan(df_merged_cousub, ['state','county_x', 'cousubfp'], columns_to_aggregate, 'pop10')

#make 'county' column by taking the last three digits of 'county_x'
cousub_data['county'] = cousub_data['county_x'].astype(int).astype(str).str[-3:].astype(int)

# Merge with sample data for county subdivisions
merged_cousub = pd.merge(sample[sample['UNIT_TYPE'] == '3 - TOWNSHIP'],
                         cousub_data,
                         how='left',
                         left_on=['FIPS_STATE', 'FIPS_COUNTY', 'FIPS_PLACE'],
                         right_on=['state', 'county', 'cousubfp'])

# Combine place and cousub data
final_merged = pd.concat([merged_place, merged_cousub], ignore_index=True)

#Find rows with missing ec_zip
missing_ec_zip = final_merged[final_merged['ec_zip'].isnull()]

#ort by POPUALATION
missing_ec_zip = missing_ec_zip.sort_values(by = 'POPULATION', ascending = False)


# Keep only the relevant columns
columns_to_keep = ['CENSUS_ID_PID6'] + columns_to_aggregate
final_data = final_merged[columns_to_keep]

#Drop pop2018
final_data = final_data.drop(columns = 'pop2018')

#Count number of non null rows fro ec_zip
print(f"Number of non-null rows for ec_zip: {final_data['ec_zip'].count()}")


# Export the final data
final_data.to_excel(os.path.join(data_path, 'interim_data', 'Social_Capital_Data.xlsx'), index=False)


print("\nData processing completed. Check the output for merge statistics.")