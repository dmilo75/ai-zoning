import pandas as pd
import os
import yaml
import helper_functions as hf
import requests
import numpy as np

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

# Read in sample data
sample = pd.read_excel(os.path.join(raw_path, "Sample Data.xlsx"))

# Initialize FIPS_COUNTY_ADJ with the same values as FIPS_COUNTY
sample['FIPS_COUNTY_ADJ'] = sample['FIPS_COUNTY']

# Apply the update function only to rows where 'State' == 'ct'
sample.loc[sample['State'] == 'ct'] = sample[sample['State'] == 'ct'].apply(hf.update_fips_adj, axis=1)

# Print adjusted rows
print(sample[sample['State'] == 'ct'][['Muni', 'FIPS_COUNTY', 'FIPS_COUNTY_ADJ']])


API_KEY = config['census_key']

state_fips = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56"
]

##

def make_more_vars(df):
    # Calculate Owner Occupied Share
    df['Owner_Occupied_Share_2022'] = df['Owner_Occupied_Housing_Units_Tenure_2022'] / df['Total_Housing_Units_Tenure_2022'] * 100

    '''
    Need to include male in the name of the variables
    '''

    # Calculate Population Aged 65 and Older
    df['Population_Aged_65_and_Over_2022'] = (
                                                     df['Population_Male_Aged_65_66_2022'] +
                                                     df['Population_Male_Aged_67_69_2022'] +
                                                     df['Population_Male_Aged_70_74_2022'] +
                                                     df['Population_Male_Aged_75_79_2022'] +
                                                     df['Population_Male_Aged_80_84_2022'] +
                                                     df['Population_Male_Aged_85_Over_2022'] +
                                                     df['Population_Female_65_66_2022'] +
                                                     df['Population_Female_67_69_2022'] +
                                                     df['Population_Female_70_74_2022'] +
                                                     df['Population_Female_75_79_2022'] +
                                                     df['Population_Female_80_84_2022'] +
                                                     df['Population_Female_85_Over_2022']
                                             ) / df['Total_Population_2022'] * 100

    # Calculate Population Under 18
    df['Population_Under_18_2022'] = (
                                             df['Population_Male_Under_5_2022'] +
                                             df['Population_Male_5_9_2022'] +
                                             df['Population_Male_10_14_2022'] +
                                             df['Population_Male_15_17_2022'] +
                                             df['Population_Female_Under_5_2022'] +
                                             df['Population_Female_5_9_2022'] +
                                             df['Population_Female_10_14_2022'] +
                                             df['Population_Female_15_17_2022']
                                     ) / df['Total_Population_2022'] * 100

    # Calculate White Share
    df['White_Share_2022'] = df['White_Population_2022'] / df['Race_Population_2022'] * 100

    #Calculate Black Share
    df['Black_Share_2022'] = df['Black_Population_2022'] / df['Race_Population_2022']*100

    # Calculate Poverty Rate
    df['Poverty_Rate_2022'] = df['Poverty_Population_2022'] / df['Poverty_Status_Population_2022'] * 100

    # Calculate Share Born in Another State
    df['Born_in_Another_State_Share_2022'] = df['Born_in_Another_State_2022'] / df['Where_Born_Population_2022'] * 100

    #Calculate Share Foreign Born
    df['Foreign_Born_Share_2022'] = df['Foreign_Born_2022'] / df['Where_Born_Population_2022'] * 100

    # Calculate Share with a 4-Year College Degree or More
    df['College_Share_2022'] = (
                                                     df['Bachelors_Degree_2022'] +
                                                     df['Master_Degree_2022'] +
                                                     df['Professional_School_Degree_2022'] +
                                                     df['Doctorate_Degree_2022']
                                             ) / df['Population_25_Over_2022'] * 100

    # Calculate Share of Structures Built Before 1970
    df['Share_Structures_Built_Before_1970_2022'] = (
                                                            df['Structures_Built_1960_to_1969_2022'] +
                                                            df['Structures_Built_1950_to_1959_2022'] +
                                                            df['Structures_Built_1940_to_1949_2022'] +
                                                            df['Structures_Built_1939_or_Earlier_2022']
                                                    ) / df['Total_Structures_Age_2022'] * 100

    # Calculate Share of Single Family Attached Units
    df['SF_Attached_Share_2022'] = df['SF_Attached_2022'] / df['Total_Structures_Units_2022'] * 100

    #Calculate share of structures with 2 or more units
    df['Structures_2_or_More_Share_2022'] = (
        df['Structures_2_2022'] +
        df['Structures_3_to_4_2022'] +
        df['Structures_5_to_9_2022'] +
        df['Structures_10_to_19_2022'] +
        df['Structures_20_to_49_2022'] +
        df['Structures_50_or_more_2022']
    ) / df['Total_Structures_Units_2022'] * 100

    #Calculate share of mobile/boat/rv/van units
    df['Mobile_Home_Boat_RV_Van_Share_2022'] = (
        df['Mobile_Home_2022'] +
        df['Boat_RV_Van_2022']
    ) / df['Total_Structures_Units_2022'] * 100

    #Calculate share that use car/truck/van to commute to work
    df['Car_Truck_Van_Share_2022'] = df['Car_Truck_Van_2022'] / df['Commute_Method_2022'] * 100

    #Calculate vacancy rate
    df['Vacancy_Rate_2022'] = df['Occupancy_Vacant_2022'] / df['Occupancy_Total_2022'] * 100

    # Calculate Share of Households Paying More Than 30% of Income on Rent
    df['Share_Paying_More_Than_30_Percent_Rent_2022'] = (
                                                                df['Rent_30_to_34_percent_2022'] +
                                                                df['Rent_35_to_39_percent_2022'] +
                                                                df['Rent_40_to_49_percent_2022'] +
                                                                df['Rent_50_percent_or_more_2022']
                                                        ) / df['Rent_Households_2022'] * 100

    # Calculate Share of Commuters with Travel Time Over 30 Minutes
    df['Share_Commute_Over_30_Minutes_2022'] = (
                                                       df['Commute_30_to_34_minutes_2022'] +
                                                       df['Commute_35_to_39_minutes_2022'] +
                                                       df['Commute_40_to_44_minutes_2022'] +
                                                       df['Commute_45_to_59_minutes_2022'] +
                                                       df['Commute_60_to_89_minutes_2022'] +
                                                       df['Commute_90_or_more_minutes_2022']
                                               ) / df['Total_Commuters_2022'] * 100

    #Make population
    df['Population_ACS_2022'] = df['Total_Population_2022']

    #Make housing units
    df['Units_Housing_2022'] = df['Housing_Units_Total_2022']

    # Calculate the percentage of housing units that are rented in cash
    df['Percent_Renting_2022'] = df['Renter_Occupied_Housing_Units_Tenure_2022'] / df['Total_Housing_Units_Tenure_2022']*df['With_Cash_Rent_2022']/df['Total_Gross_Rent_2022']

    # Calculate rent distribution percentages as a percentage of all housing units
    df['Rent_Less_Than_100_Pct_All_Units'] = (df['Rent_Less_Than_100_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_100_to_149_Pct_All_Units'] = (df['Rent_100_to_149_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_150_to_199_Pct_All_Units'] = (df['Rent_150_to_199_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_200_to_249_Pct_All_Units'] = (df['Rent_200_to_249_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_250_to_299_Pct_All_Units'] = (df['Rent_250_to_299_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_300_to_349_Pct_All_Units'] = (df['Rent_300_to_349_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_350_to_399_Pct_All_Units'] = (df['Rent_350_to_399_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_400_to_449_Pct_All_Units'] = (df['Rent_400_to_449_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_450_to_499_Pct_All_Units'] = (df['Rent_450_to_499_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_500_to_549_Pct_All_Units'] = (df['Rent_500_to_549_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_550_to_599_Pct_All_Units'] = (df['Rent_550_to_599_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_600_to_649_Pct_All_Units'] = (df['Rent_600_to_649_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_650_to_699_Pct_All_Units'] = (df['Rent_650_to_699_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_700_to_749_Pct_All_Units'] = (df['Rent_700_to_749_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_750_to_799_Pct_All_Units'] = (df['Rent_750_to_799_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_800_to_899_Pct_All_Units'] = (df['Rent_800_to_899_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_900_to_999_Pct_All_Units'] = (df['Rent_900_to_999_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_1000_to_1249_Pct_All_Units'] = (df['Rent_1000_to_1249_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_1250_to_1499_Pct_All_Units'] = (df['Rent_1250_to_1499_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_1500_to_1999_Pct_All_Units'] = (df['Rent_1500_to_1999_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_2000_to_2499_Pct_All_Units'] = (df['Rent_2000_to_2499_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_2500_to_2999_Pct_All_Units'] = (df['Rent_2500_to_2999_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_3000_to_3499_Pct_All_Units'] = (df['Rent_3000_to_3499_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100
    df['Rent_3500_or_more_Pct_All_Units'] = (df['Rent_3500_or_more_2022'] / df['With_Cash_Rent_2022']) * df[
        'Percent_Renting_2022'] * 100

    # Calculate percentages for home values
    for value_range in ['Less_Than_10000', '10000_to_14999', '15000_to_19999', '20000_to_24999', '25000_to_29999',
                        '30000_to_34999', '35000_to_39999', '40000_to_49999', '50000_to_59999', '60000_to_69999',
                        '70000_to_79999', '80000_to_89999', '90000_to_99999', '100000_to_124999', '125000_to_149999',
                        '150000_to_174999', '175000_to_199999', '200000_to_249999', '250000_to_299999',
                        '300000_to_399999',
                        '400000_to_499999', '500000_to_749999', '750000_to_999999', '1000000_to_1499999',
                        '1500000_to_1999999',
                        '2000000_or_more']:
        df[f'Value_{value_range}_Pct'] = df[f'Value_{value_range}_2022'] / df['Total_Value_2022'] * 100*df['Owner_Occupied_Housing_Units_Tenure_2022']/df['Total_Housing_Units_Tenure_2022']

    # Calculate percentages for mortgage payments
    for mortgage_range in ['Less_Than_200', '200_to_299', '300_to_399', '400_to_499', '500_to_599', '600_to_699',
                           '700_to_799', '800_to_899', '900_to_999', '1000_to_1249', '1250_to_1499', '1500_to_1999',
                           '2000_to_2499', '2500_to_2999', '3000_to_3499', '3500_to_3999', '4000_or_more']:
        df[f'Mortgage_{mortgage_range}_Pct'] = df[f'Mortgage_{mortgage_range}_2022'] / df[
            'Housing_Units_Total_2022'] * 100

    # Drop the original variables
    vars_to_drop = list(vars['2022'].keys())

    #Drop any vars in '2012'
    vars_2012 = list(vars['2012'].keys())
    vars_to_drop = [x for x in vars_to_drop if x not in vars_2012]

    vars_to_drop = [x+"_2022" for x in vars_to_drop]

    df = df.drop(columns = vars_to_drop)

    return df

def process_df(df, var_names,var_codes,year):

    # Add _year to end of var names
    var_names = [name + f"_{year}" for name in var_names]

    # Renaming the columns
    rename_dict = dict(zip(var_codes, var_names))
    df = df.rename(columns=rename_dict)

    #Drop rows with None
    df = df.dropna()

    # Cleaning the data
    df[var_names] = df[var_names].applymap(lambda x: int(x))

    df = df.replace(-666666666, np.nan)

    # Add _year to NAME
    df = df.rename(columns={'NAME': 'NAME_' + str(year)})

    #If year is 2022 then send to make more vars
    if year == '2022':
        df = make_more_vars(df)

    return df

def split_variables(variables, chunk_size=49):
    var_codes = list(variables.values())
    var_names = list(variables.keys())
    for i in range(0, len(var_codes), chunk_size):
        yield var_names[i:i + chunk_size], var_codes[i:i + chunk_size]


def get_ACS_place(variables, year):
    all_data = None

    for var_names_chunk, var_codes_chunk in split_variables(variables):
        API_ENDPOINT = f"https://api.census.gov/data/{year}/acs/acs5"
        if all_data is None:
            var_codes_str = ','.join(var_codes_chunk + ['NAME'])
        else:
            var_codes_str = ','.join(var_codes_chunk)
        params = {
            "get": var_codes_str,
            "for": "place:*",
            "key": API_KEY
        }
        response = requests.get(API_ENDPOINT, params=params)
        data = response.json()

        header = data[0]
        rows = data[1:]

        df_chunk = pd.DataFrame(rows, columns=header)

        if all_data is None:
            all_data = df_chunk
        else:
            all_data = all_data.merge(df_chunk, on=[ 'state', 'place'])

    var_names = list(variables.keys())
    var_codes = list(variables.values())
    all_data = process_df(all_data, var_names, var_codes, year)

    return all_data

def get_ACS_county_subdivision(variables, year):
    all_data = None

    for var_names_chunk, var_codes_chunk in split_variables(variables):
        if all_data is None:
            var_codes_str = ','.join(var_codes_chunk + ['NAME'])
        else:
            var_codes_str = ','.join(var_codes_chunk)
        all_states_data = []
        header_saved = False  # Flag to check if the header is saved

        for state in state_fips:
            API_ENDPOINT = f"https://api.census.gov/data/{year}/acs/acs5"
            params = {
                "get": var_codes_str,
                "for": "county subdivision:*",
                "in": f"state:{state}",
                "key": API_KEY
            }
            response = requests.get(API_ENDPOINT, params=params)
            data = response.json()

            # Save header only once
            if not header_saved:
                header = data[0]
                header_saved = True
                all_states_data.append(header)

            # Extend the main data list with data rows excluding the header
            all_states_data.extend(data[1:])

        df_chunk = pd.DataFrame(all_states_data[1:], columns=all_states_data[0])  # Use saved header for columns

        if all_data is None:
            all_data = df_chunk
        else:
            all_data = all_data.merge(df_chunk, on=[ 'state', 'county','county subdivision'])

    var_names = list(variables.keys())
    var_codes = list(variables.values())
    all_data = process_df(all_data, var_names, var_codes, year)

    return all_data

##

# %%ACS Merging code
# https://api.census.gov/data/2022/acs/acs1/variables.html



vars = {}

vars['2012'] = {
    'Median_Household_Income': 'B19013_001E',
    'Median_Gross_Rent': 'B25064_001E',
    'Median_Home_Value': 'B25077_001E'
}

vars['2017'] = {
    'Agg_Home_Value': 'B25082_001E',

}



vars['2022'] = {
    'Housing_Units_Total':'B25001_001E',
    'Median_Household_Income': 'B19013_001E',
    'Median_Gross_Rent': 'B25064_001E',
    'Median_Home_Value': 'B25077_001E',
    'Owner_Occupied_Housing_Units_Tenure': 'B25003_002E',
    'Total_Housing_Units_Tenure': 'B25003_001E',
    'Population_Male_Aged_65_66': 'B01001_020E',
    'Population_Male_Aged_67_69': 'B01001_021E',
    'Population_Male_Aged_70_74': 'B01001_022E',
    'Population_Male_Aged_75_79': 'B01001_023E',
    'Population_Male_Aged_80_84': 'B01001_024E',
    'Population_Male_Aged_85_Over': 'B01001_025E',
    'Population_Female_65_66': 'B01001_044E',
    'Population_Female_67_69': 'B01001_045E',
    'Population_Female_70_74': 'B01001_046E',
    'Population_Female_75_79': 'B01001_047E',
    'Population_Female_80_84': 'B01001_048E',
    'Population_Female_85_Over': 'B01001_049E',
    'Total_Population': 'B01001_001E',
    'Population_Male_Under_5': 'B01001_003E',
    'Population_Male_5_9': 'B01001_004E',
    'Population_Male_10_14': 'B01001_005E',
    'Population_Male_15_17': 'B01001_006E',
    'Population_Female_Under_5': 'B01001_027E',
    'Population_Female_5_9': 'B01001_028E',
    'Population_Female_10_14': 'B01001_029E',
    'Population_Female_15_17': 'B01001_030E',
    'Race_Population': 'B02001_001E',
    'White_Population': 'B02001_002E',
    'Black_Population': 'B02001_003E',
    'Poverty_Population': 'B17001_002E',
    'Poverty_Status_Population': 'B17001_001E',
    'Born_in_Another_State': 'B05002_003E',
    'Foreign_Born': 'B05002_013E',
    'Where_Born_Population': 'B05002_001E',
    'Population_25_Over': 'B15003_001E',
    'Bachelors_Degree': 'B15003_022E',
    'Master_Degree': 'B15003_023E',
    'Professional_School_Degree': 'B15003_024E',
    'Doctorate_Degree': 'B15003_025E',
    'Total_Structures_Age': 'B25034_001E',
    'Structures_Built_1960_to_1969': 'B25034_008E',
    'Structures_Built_1950_to_1959': 'B25034_009E',
    'Structures_Built_1940_to_1949': 'B25034_010E',
    'Structures_Built_1939_or_Earlier': 'B25034_011E',
    'Total_Structures_Units': 'B25024_001E',
    'SF_Attached': 'B25024_003E',
    'Structures_2': 'B25024_004E',
    'Structures_3_to_4': 'B25024_005E',
    'Structures_5_to_9': 'B25024_006E',
    'Structures_10_to_19': 'B25024_007E',
    'Structures_20_to_49': 'B25024_008E',
    'Structures_50_or_more': 'B25024_009E',
    'Mobile_Home': 'B25024_010E',
    'Boat_RV_Van': 'B25024_011E',
    'Commute_Method':'B08301_001E',
    'Car_Truck_Van':'B08301_002E',
    'Occupancy_Total':'B25002_001E',
    'Occupancy_Vacant':'B25002_003E',
    'Rent_Households': 'B25070_001E',
    'Rent_30_to_34_percent': 'B25070_007E',
    'Rent_35_to_39_percent': 'B25070_008E',
    'Rent_40_to_49_percent': 'B25070_009E',
    'Rent_50_percent_or_more': 'B25070_010E',
    'Total_Commuters': 'B08303_001E',
    'Commute_30_to_34_minutes': 'B08303_008E',
    'Commute_35_to_39_minutes': 'B08303_009E',
    'Commute_40_to_44_minutes': 'B08303_010E',
    'Commute_45_to_59_minutes': 'B08303_011E',
    'Commute_60_to_89_minutes': 'B08303_012E',
    'Commute_90_or_more_minutes': 'B08303_013E',
    'Total_Gross_Rent': 'B25063_001E',
    'With_Cash_Rent': 'B25063_002E',
    'Rent_Less_Than_100': 'B25063_003E',
    'Rent_100_to_149': 'B25063_004E',
    'Rent_150_to_199': 'B25063_005E',
    'Rent_200_to_249': 'B25063_006E',
    'Rent_250_to_299': 'B25063_007E',
    'Rent_300_to_349': 'B25063_008E',
    'Rent_350_to_399': 'B25063_009E',
    'Rent_400_to_449': 'B25063_010E',
    'Rent_450_to_499': 'B25063_011E',
    'Rent_500_to_549': 'B25063_012E',
    'Rent_550_to_599': 'B25063_013E',
    'Rent_600_to_649': 'B25063_014E',
    'Rent_650_to_699': 'B25063_015E',
    'Rent_700_to_749': 'B25063_016E',
    'Rent_750_to_799': 'B25063_017E',
    'Rent_800_to_899': 'B25063_018E',
    'Rent_900_to_999': 'B25063_019E',
    'Rent_1000_to_1249': 'B25063_020E',
    'Rent_1250_to_1499': 'B25063_021E',
    'Rent_1500_to_1999': 'B25063_022E',
    'Rent_2000_to_2499': 'B25063_023E',
    'Rent_2500_to_2999': 'B25063_024E',
    'Rent_3000_to_3499': 'B25063_025E',
    'Rent_3500_or_more': 'B25063_026E',
    'Renter_Occupied_Housing_Units_Tenure': 'B25003_003E',
'Total_Value': 'B25075_001E',
    'Value_Less_Than_10000': 'B25075_002E',
    'Value_10000_to_14999': 'B25075_003E',
    'Value_15000_to_19999': 'B25075_004E',
    'Value_20000_to_24999': 'B25075_005E',
    'Value_25000_to_29999': 'B25075_006E',
    'Value_30000_to_34999': 'B25075_007E',
    'Value_35000_to_39999': 'B25075_008E',
    'Value_40000_to_49999': 'B25075_009E',
    'Value_50000_to_59999': 'B25075_010E',
    'Value_60000_to_69999': 'B25075_011E',
    'Value_70000_to_79999': 'B25075_012E',
    'Value_80000_to_89999': 'B25075_013E',
    'Value_90000_to_99999': 'B25075_014E',
    'Value_100000_to_124999': 'B25075_015E',
    'Value_125000_to_149999': 'B25075_016E',
    'Value_150000_to_174999': 'B25075_017E',
    'Value_175000_to_199999': 'B25075_018E',
    'Value_200000_to_249999': 'B25075_019E',
    'Value_250000_to_299999': 'B25075_020E',
    'Value_300000_to_399999': 'B25075_021E',
    'Value_400000_to_499999': 'B25075_022E',
    'Value_500000_to_749999': 'B25075_023E',
    'Value_750000_to_999999': 'B25075_024E',
    'Value_1000000_to_1499999': 'B25075_025E',
    'Value_1500000_to_1999999': 'B25075_026E',
    'Value_2000000_or_more': 'B25075_027E',
    'Total_Mortgage_Status': 'B25087_001E',
    'Units_With_Mortgage': 'B25087_002E',
    'Mortgage_Less_Than_200': 'B25087_003E',
    'Mortgage_200_to_299': 'B25087_004E',
    'Mortgage_300_to_399': 'B25087_005E',
    'Mortgage_400_to_499': 'B25087_006E',
    'Mortgage_500_to_599': 'B25087_007E',
    'Mortgage_600_to_699': 'B25087_008E',
    'Mortgage_700_to_799': 'B25087_009E',
    'Mortgage_800_to_899': 'B25087_010E',
    'Mortgage_900_to_999': 'B25087_011E',
    'Mortgage_1000_to_1249': 'B25087_012E',
    'Mortgage_1250_to_1499': 'B25087_013E',
    'Mortgage_1500_to_1999': 'B25087_014E',
    'Mortgage_2000_to_2499': 'B25087_015E',
    'Mortgage_2500_to_2999': 'B25087_016E',
    'Mortgage_3000_to_3499': 'B25087_017E',
    'Mortgage_3500_to_3999': 'B25087_018E',
    'Mortgage_4000_or_more': 'B25087_019E',
}

places = []
cousubs = []


for year in vars.keys():

    place_df = get_ACS_place(vars[year], year)
    cousub_df = get_ACS_county_subdivision(vars[year], year)

    places.append(place_df)
    cousubs.append(cousub_df)





'''
I checked to see if fips places codes changing over time are a problem and they dont seem to be. 

When merging in place codes at least we always get the same name across 2012 to 2022

Sometimes places exist in 2012 and not in 2022 though and vice versa
'''

#Merge together places on state and place
# Initialize the merged_places with the first DataFrame in the list
merged_places = places[0]

# Loop through the remaining DataFrames and merge them
for i in range(1, len(places)):
    merged_places = pd.merge(merged_places, places[i], on=['state', 'place'], how='outer')

#Initialize merged_cousubs with the first DataFrame in the list
merged_cousubs = cousubs[0]

#Loop through the remaining DataFrames and merge them
for i in range(1, len(cousubs)):
    merged_cousubs = pd.merge(merged_cousubs, cousubs[i], on=['state', 'county', 'county subdivision'], how='outer')


#define vars_common as common vars between 2012 and 2022
vars_common = {k: v for k, v in vars['2012'].items() if k in vars['2022']}

#Make percent change for each variable
acs_vars = list(vars_common.keys())
for var in acs_vars:
    merged_places[var + "_Percent_Change"] = ((merged_places[var + '_2022'] - merged_places[var + '_2012']) / merged_places[var + '_2012']) * 100
    merged_cousubs[var + "_Percent_Change"] = ((merged_cousubs[var + '_2022'] - merged_cousubs[var + '_2012']) / merged_cousubs[var + '_2012']) * 100

#Make place an int for merging
merged_places['place'] = merged_places['place'].astype(int)

#Make state an int
merged_places['state'] = merged_places['state'].astype(int)

#Get places from sample
sample_places = sample[sample['UNIT_TYPE'] == '2 - MUNICIPAL']

#Merge merged_places into sample
merged_place_df = pd.merge(sample_places, merged_places, left_on = ['FIPS_STATE','FIPS_PLACE'], right_on=['state', 'place'], how='inner')

#Calc percent merged as length of final_df over length of sample_places
percent_merged = (len(merged_place_df) / len(sample_places)) * 100
print(f"Percent places merged: {percent_merged}")

#Get county subdivisions from sample
sample_cousubs = sample[sample['UNIT_TYPE'] == '3 - TOWNSHIP']

#Convert state, county, and county subdivision to ints
merged_cousubs['state'] = merged_cousubs['state'].astype(int)
merged_cousubs['county'] = merged_cousubs['county'].astype(int)
merged_cousubs['county subdivision'] = merged_cousubs['county subdivision'].astype(int)

#Merge merged_cousubs into sample
merged_cousub_df = pd.merge(sample_cousubs, merged_cousubs, left_on = ['FIPS_STATE','FIPS_COUNTY_ADJ','FIPS_PLACE'], right_on=['state', 'county', 'county subdivision'], how='inner')

#Calculate percent merged
percent_merged = (len(merged_cousub_df) / len(sample_cousubs)) * 100
print(f"Percent cousubs merged: {percent_merged}")

#Concatenate the two merged dataframes
final_df = pd.concat([merged_place_df, merged_cousub_df])

#Only keep relevant variables and CENSUS_ID_PID6
'''
Need to adjust once made sub_indices
'''

new_vars = [
    'Owner_Occupied_Share_2022',
    'Population_Aged_65_and_Over_2022',
    'Population_Under_18_2022',
    'White_Share_2022',
    'Black_Share_2022',
    'Poverty_Rate_2022',
    'Born_in_Another_State_Share_2022',
    'Foreign_Born_Share_2022',
    'College_Share_2022',
    'Share_Structures_Built_Before_1970_2022',
    'SF_Attached_Share_2022',
    'Structures_2_or_More_Share_2022',
    'Mobile_Home_Boat_RV_Van_Share_2022',
    'Car_Truck_Van_Share_2022',
    'Vacancy_Rate_2022',
    'Share_Paying_More_Than_30_Percent_Rent_2022',
    'Share_Commute_Over_30_Minutes_2022',
    'Population_ACS_2022',
    'Agg_Home_Value_2017',
'Rent_Less_Than_100_Pct_All_Units',
    'Rent_100_to_149_Pct_All_Units',
    'Rent_150_to_199_Pct_All_Units',
    'Rent_200_to_249_Pct_All_Units',
    'Rent_250_to_299_Pct_All_Units',
    'Rent_300_to_349_Pct_All_Units',
    'Rent_350_to_399_Pct_All_Units',
    'Rent_400_to_449_Pct_All_Units',
    'Rent_450_to_499_Pct_All_Units',
    'Rent_500_to_549_Pct_All_Units',
    'Rent_550_to_599_Pct_All_Units',
    'Rent_600_to_649_Pct_All_Units',
    'Rent_650_to_699_Pct_All_Units',
    'Rent_700_to_749_Pct_All_Units',
    'Rent_750_to_799_Pct_All_Units',
    'Rent_800_to_899_Pct_All_Units',
    'Rent_900_to_999_Pct_All_Units',
    'Rent_1000_to_1249_Pct_All_Units',
    'Rent_1250_to_1499_Pct_All_Units',
    'Rent_1500_to_1999_Pct_All_Units',
    'Rent_2000_to_2499_Pct_All_Units',
    'Rent_2500_to_2999_Pct_All_Units',
    'Rent_3000_to_3499_Pct_All_Units',
    'Rent_3500_or_more_Pct_All_Units',
'Value_Less_Than_10000_Pct', 'Value_10000_to_14999_Pct', 'Value_15000_to_19999_Pct',
    'Value_20000_to_24999_Pct', 'Value_25000_to_29999_Pct', 'Value_30000_to_34999_Pct',
    'Value_35000_to_39999_Pct', 'Value_40000_to_49999_Pct', 'Value_50000_to_59999_Pct',
    'Value_60000_to_69999_Pct', 'Value_70000_to_79999_Pct', 'Value_80000_to_89999_Pct',
    'Value_90000_to_99999_Pct', 'Value_100000_to_124999_Pct', 'Value_125000_to_149999_Pct',
    'Value_150000_to_174999_Pct', 'Value_175000_to_199999_Pct', 'Value_200000_to_249999_Pct',
    'Value_250000_to_299999_Pct', 'Value_300000_to_399999_Pct', 'Value_400000_to_499999_Pct',
    'Value_500000_to_749999_Pct', 'Value_750000_to_999999_Pct', 'Value_1000000_to_1499999_Pct',
    'Value_1500000_to_1999999_Pct', 'Value_2000000_or_more_Pct',
    'Mortgage_Less_Than_200_Pct', 'Mortgage_200_to_299_Pct', 'Mortgage_300_to_399_Pct',
    'Mortgage_400_to_499_Pct', 'Mortgage_500_to_599_Pct', 'Mortgage_600_to_699_Pct',
    'Mortgage_700_to_799_Pct', 'Mortgage_800_to_899_Pct', 'Mortgage_900_to_999_Pct',
    'Mortgage_1000_to_1249_Pct', 'Mortgage_1250_to_1499_Pct', 'Mortgage_1500_to_1999_Pct',
    'Mortgage_2000_to_2499_Pct', 'Mortgage_2500_to_2999_Pct', 'Mortgage_3000_to_3499_Pct',
    'Mortgage_3500_to_3999_Pct', 'Mortgage_4000_or_more_Pct',
]

rel_cols = [col for col in final_df.columns if any(var in col for var in acs_vars)] + ['CENSUS_ID_PID6']+new_vars

#Keep only relevant columns
final_df = final_df[rel_cols]

#Make dataframe of unmerged rows from sample
unmerged_rows = sample[~sample['CENSUS_ID_PID6'].isin(final_df['CENSUS_ID_PID6'])]

#Save to 'interim_data' under processed data folder
final_df.to_excel(os.path.join(data_path, 'interim_data', 'ACS Data.xlsx'))


