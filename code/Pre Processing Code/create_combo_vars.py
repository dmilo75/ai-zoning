import pandas as pd
import os
import yaml
import numpy as np
import requests

def process_gov_fin_vars(merged_df):
    # Define property tax rate
    merged_df['Property_Tax_Rate_2017'] = 100000 * merged_df['Property_Tax_2017'] / merged_df['Agg_Home_Value_2017']

    # Anything above 0.2, recode as null
    merged_df['Property_Tax_Rate_2017'] = np.where(merged_df['Property_Tax_Rate_2017'] > 20, np.nan, merged_df['Property_Tax_Rate_2017'])

    # Define expenditures of central staff per capita
    merged_df['Cen_Staff_Total_Expend_Per_Capita_2017'] = 1000 * merged_df['Cen_Staff_Total_Expend_2017'] / merged_df['Population_2017']

    # Values above the top 99th percentile recode as null
    top_99_cen_staff = merged_df['Cen_Staff_Total_Expend_Per_Capita_2017'].quantile(0.99)
    merged_df['Cen_Staff_Total_Expend_Per_Capita_2017'] = np.where(merged_df['Cen_Staff_Total_Expend_Per_Capita_2017'] > top_99_cen_staff, np.nan, merged_df['Cen_Staff_Total_Expend_Per_Capita_2017'])

    # Define total revenue per capita
    merged_df['Total_Revenue_Per_Capita_2017'] = 1000 * merged_df['Total_Revenue_2017'] / merged_df['Population_2017']

    # Winsorize at 99th percentile
    top_99_total_revenue = merged_df['Total_Revenue_Per_Capita_2017'].quantile(0.99)
    merged_df['Total_Revenue_Per_Capita_2017'] = np.where(merged_df['Total_Revenue_Per_Capita_2017'] > top_99_total_revenue, np.nan, merged_df['Total_Revenue_Per_Capita_2017'])

    # Define total expenditures per capita
    merged_df['Total_Expenditure_Per_Capita_2017'] = 1000 * merged_df['Total_Expenditure_2017'] / merged_df['Population_2017']

    # Winsorize at 99th percentile
    top_99_total_expenditure = merged_df['Total_Expenditure_Per_Capita_2017'].quantile(0.99)
    merged_df['Total_Expenditure_Per_Capita_2017'] = np.where(merged_df['Total_Expenditure_Per_Capita_2017'] > top_99_total_expenditure, np.nan, merged_df['Total_Expenditure_Per_Capita_2017'])

    # Calculate intergovernmental transfer per capita
    merged_df['Total_IGR_Hous_Com_Dev_Per_Capita_2017'] = 1000 * merged_df['Total_IGR_Hous_Com_Dev_2017'] / merged_df['Population_2017']

    # Winsorize at 99th percentile
    top_99_igr = merged_df['Total_IGR_Hous_Com_Dev_Per_Capita_2017'].quantile(0.99)
    merged_df['Total_IGR_Hous_Com_Dev_Per_Capita_2017'] = np.where(merged_df['Total_IGR_Hous_Com_Dev_Per_Capita_2017'] > top_99_igr, np.nan, merged_df['Total_IGR_Hous_Com_Dev_Per_Capita_2017'])

    # Drop raw variables from government finance
    merged_df = merged_df.drop(columns=['Total_Revenue_2017', 'Total_Expenditure_2017', 'Property_Tax_2017', 'Total_IGR_Hous_Com_Dev_2017', 'Cen_Staff_Total_Expend_2017', 'Population_2017'])

    return merged_df


# Constants and category dictionaries
RENT_CATEGORIES = {
    'Rent_Less_Than_100_Pct_All_Units': 100,
    'Rent_100_to_149_Pct_All_Units': 149,
    'Rent_150_to_199_Pct_All_Units': 199,
    'Rent_200_to_249_Pct_All_Units': 249,
    'Rent_250_to_299_Pct_All_Units': 299,
    'Rent_300_to_349_Pct_All_Units': 349,
    'Rent_350_to_399_Pct_All_Units': 399,
    'Rent_400_to_449_Pct_All_Units': 449,
    'Rent_450_to_499_Pct_All_Units': 499,
    'Rent_500_to_549_Pct_All_Units': 549,
    'Rent_550_to_599_Pct_All_Units': 599,
    'Rent_600_to_649_Pct_All_Units': 649,
    'Rent_650_to_699_Pct_All_Units': 699,
    'Rent_700_to_749_Pct_All_Units': 749,
    'Rent_750_to_799_Pct_All_Units': 799,
    'Rent_800_to_899_Pct_All_Units': 899,
    'Rent_900_to_999_Pct_All_Units': 999,
    'Rent_1000_to_1249_Pct_All_Units': 1249,
    'Rent_1250_to_1499_Pct_All_Units': 1499,
    'Rent_1500_to_1999_Pct_All_Units': 1999,
    'Rent_2000_to_2499_Pct_All_Units': 2499,
    'Rent_2500_to_2999_Pct_All_Units': 2999,
    'Rent_3000_to_3499_Pct_All_Units': 3499,
    'Rent_3500_or_more_Pct_All_Units': 3500
}

VALUE_CATEGORIES = {
    'Value_Less_Than_10000_Pct': 10000,
    'Value_10000_to_14999_Pct': 14999,
    'Value_15000_to_19999_Pct': 19999,
    'Value_20000_to_24999_Pct': 24999,
    'Value_25000_to_29999_Pct': 29999,
    'Value_30000_to_34999_Pct': 34999,
    'Value_35000_to_39999_Pct': 39999,
    'Value_40000_to_49999_Pct': 49999,
    'Value_50000_to_59999_Pct': 59999,
    'Value_60000_to_69999_Pct': 69999,
    'Value_70000_to_79999_Pct': 79999,
    'Value_80000_to_89999_Pct': 89999,
    'Value_90000_to_99999_Pct': 99999,
    'Value_100000_to_124999_Pct': 124999,
    'Value_125000_to_149999_Pct': 149999,
    'Value_150000_to_174999_Pct': 174999,
    'Value_175000_to_199999_Pct': 199999,
    'Value_200000_to_249999_Pct': 249999,
    'Value_250000_to_299999_Pct': 299999,
    'Value_300000_to_399999_Pct': 399999,
    'Value_400000_to_499999_Pct': 499999,
    'Value_500000_to_749999_Pct': 749999,
    'Value_750000_to_999999_Pct': 999999,
    'Value_1000000_to_1499999_Pct': 1499999,
    'Value_1500000_to_1999999_Pct': 1999999,
    'Value_2000000_or_more_Pct': 2000000
}

MORTGAGE_CATEGORIES = {
    'Mortgage_Less_Than_200_Pct': 200,
    'Mortgage_200_to_299_Pct': 299,
    'Mortgage_300_to_399_Pct': 399,
    'Mortgage_400_to_499_Pct': 499,
    'Mortgage_500_to_599_Pct': 599,
    'Mortgage_600_to_699_Pct': 699,
    'Mortgage_700_to_799_Pct': 799,
    'Mortgage_800_to_899_Pct': 899,
    'Mortgage_900_to_999_Pct': 999,
    'Mortgage_1000_to_1249_Pct': 1249,
    'Mortgage_1250_to_1499_Pct': 1499,
    'Mortgage_1500_to_1999_Pct': 1999,
    'Mortgage_2000_to_2499_Pct': 2499,
    'Mortgage_2500_to_2999_Pct': 2999,
    'Mortgage_3000_to_3499_Pct': 3499,
    'Mortgage_3500_to_3999_Pct': 3999,
    'Mortgage_4000_or_more_Pct': 4000
}

def get_state_median_income(api_key, year):
    acs_vars = {
        'State_Median_Household_Income': 'B19013_001E'
    }

    var_codes_str = ','.join(list(acs_vars.values()) + ['NAME'])
    api_endpoint = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": var_codes_str,
        "for": "state:*",
        "key": api_key
    }

    response = requests.get(api_endpoint, params=params)
    data = response.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns={v: k for k, v in acs_vars.items()})
    df = df.apply(pd.to_numeric, errors='ignore')

    df[f"State_Median_Household_Income_{year}"] = df['State_Median_Household_Income']
    df = df.drop(columns=['State_Median_Household_Income'])

    return df


def calculate_affordable_units(row, categories, income, multiplier):
    affordable_limit = income * multiplier
    affordable_columns = [col for col, value in categories.items() if value <= affordable_limit]
    return row[affordable_columns].sum()


def add_affordability_measures(df, income_column):
    # Calculate total units for each category
    df['Total_Rental_Units'] = df[RENT_CATEGORIES.keys()].sum(axis=1)
    df['Total_Value_Units'] = df[VALUE_CATEGORIES.keys()].sum(axis=1)
    df['Total_Mortgage_Units'] = df[MORTGAGE_CATEGORIES.keys()].sum(axis=1)

    # Rental affordability
    df['Affordable_Rental_Units'] = df.apply(
        lambda row: calculate_affordable_units(row, RENT_CATEGORIES, row[income_column], 0.3 / 12), axis=1)
    df['Percent_Affordable_Rental'] = df['Affordable_Rental_Units'] / df['Total_Rental_Units'] * 100

    # Home value affordability
    df['Affordable_Value_Units'] = df.apply(
        lambda row: calculate_affordable_units(row, VALUE_CATEGORIES, row[income_column], 3), axis=1)
    df['Percent_Affordable_Value'] = df['Affordable_Value_Units'] / df['Total_Value_Units'] * 100

    # Mortgage affordability
    df['Affordable_Mortgage_Units'] = df.apply(
        lambda row: calculate_affordable_units(row, MORTGAGE_CATEGORIES, row[income_column], 0.3 / 12), axis=1)
    df['Percent_Affordable_Mortgage'] = df['Affordable_Mortgage_Units'] / df['Total_Mortgage_Units'] * 100

    # Combined affordability
    df['Combined_Affordable_Units'] = (
            df['Affordable_Rental_Units'] +
            df['Affordable_Value_Units']
    )

    return df


def create_affordability_measures(df, api_key):
    # Get state median income data
    state_income_data = get_state_median_income(api_key, 2022)

    # Merge state income data with main dataframe
    df = pd.merge(df, state_income_data, left_on='FIPS_STATE', right_on='state', how='left')

    # Add affordability measures
    income_column = f'State_Median_Household_Income_2022'
    df = add_affordability_measures(df, income_column)

    return df