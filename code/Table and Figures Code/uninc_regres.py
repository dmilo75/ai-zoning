import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns


os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

##

def load_and_merge_block_group_data():
    # Load ACS data
    acs_data = pd.read_csv(os.path.join(config['raw_data'], '2022_acs_blockgroup_data', "acs_blockgroup_data_all_states.csv"))

    # Replace any values of -666666666 with np.nan
    acs_data = acs_data.replace(-666666666, np.nan)

    # Load shape data
    shape_data = pd.read_csv(os.path.join(config['processed_data'], 'Block Group Analysis Results', 'block_group_analysis_results.csv'))

    # Merge shape_data with acs_data
    merged_data = shape_data.merge(acs_data, left_on=['STATEFP', 'COUNTYFP', 'TRACTCE', 'BLKGRPCE'], right_on=['state', 'county', 'tract', 'block group'])

    # If percent incorporated > 100, make it 100
    merged_data['Percent_Incorporated'] = np.where(merged_data['Percent_Incorporated'] > 100, 100, merged_data['Percent_Incorporated'])

    # If percent incorporated < 0, make it 0
    merged_data['Percent_Incorporated'] = np.where(merged_data['Percent_Incorporated'] < 0, 0, merged_data['Percent_Incorporated'])

    # Make vacancy rate as percent of Vacant_Housing_Units over Total_Housing_Units
    merged_data['Vacancy_Rate'] = 100 * merged_data['Vacant_Housing_Units'] / merged_data['Total_Housing_Units']

    # Make rental rate as 1 - Owner_Occupied_Units / Total_Occupied_Units
    merged_data['Rental_Rate'] = 100 - 100 * merged_data['Owner_Occupied_Units'] / merged_data['Total_Occupied_Units']

    # Make percent commute over 60 mins as Workers_60_to_89_min_commute + Workers_90_or_more_min_commute / Total_Workers
    merged_data['Percent_Commute_Over_60'] = 100 * (merged_data['Workers_60_to_89_min_commute'] + merged_data['Workers_90_plus_min_commute']) / merged_data['Total_Workers']

    # Make percent over 65 as sum of relevant age groups / total population * 100
    age_groups_over_65 = [
        'Males_65_to_66', 'Males_67_to_69', 'Males_70_to_74', 'Males_75_to_79', 'Males_80_to_84', 'Males_85_plus',
        'Females_65_to_66', 'Females_67_to_69', 'Females_70_to_74', 'Females_75_to_79', 'Females_80_to_84', 'Females_85_plus'
    ]
    merged_data['Percent_Over_65'] = 100 * merged_data[age_groups_over_65].sum(axis=1) / merged_data['Total_Population']

    # Create incorporated and unincorporated dataframes
    incorporated = merged_data.copy()
    unincorporated = merged_data.copy()

    # Set the weight column
    incorporated['Weight'] = merged_data['Percent_Incorporated'] * merged_data['Total_Population']
    unincorporated['Weight'] = (100 - merged_data['Percent_Incorporated']) * merged_data['Total_Population']

    # Add binary indicator for incorporation
    incorporated['Is_Incorporated'] = 1
    unincorporated['Is_Incorporated'] = 0

    # Drop rows with weight 0
    incorporated = incorporated[incorporated['Weight'] != 0]
    unincorporated = unincorporated[unincorporated['Weight'] != 0]

    # Concatenate the dataframes
    final_data = pd.concat([incorporated, unincorporated], ignore_index=True)

    return final_data



##Unincorporated regression

orig_bg_data = load_and_merge_block_group_data()

##

'''
Stress that does this in all states now 
'''

bg_data = orig_bg_data.copy()

# Define the number of quantiles
num_quantiles = 5

# Create a new variable with quantiles of 'Distance_to_Metro'
bg_data['Distance_to_Metro_Quantiles'] = pd.qcut(bg_data['Distance_to_Metro'], num_quantiles, labels=False)

#Make Is_Unincorporated
bg_data['Is_Unincorporated'] = 1 - bg_data['Is_Incorporated']

# Update the formulas dictionary to include the flexible distance control
formulas = {
    'Bivariate': '{lhs_var} ~ Is_Unincorporated',
    'Metro FE': '{lhs_var} ~ Is_Unincorporated + C(Nearest_Metro)',
    'Distance FE': '{lhs_var} ~ Is_Unincorporated + C(Distance_to_Metro_Quantiles)',
    'Metro and Distance FE': '{lhs_var} ~ Is_Unincorporated + C(Nearest_Metro) + C(Distance_to_Metro_Quantiles)',
}

# Dictionary to store the results
results_dict = {}

# Define the dependent variables
dependent_vars = [
    'Median_Home_Value',
    'Median_Year_Built',
    'Median_Household_Income',
    'Median_Gross_Rent',
    'Vacancy_Rate',
    'Rental_Rate',
    'Percent_Commute_Over_60',
    'Percent_Over_65',
]

#Divide median home value by 10,000
bg_data['Median_Home_Value'] = bg_data['Median_Home_Value'] / 10000

#Divide median household income by 1,000
bg_data['Median_Household_Income'] = bg_data['Median_Household_Income'] / 1000

# Function to add significance stars
def significance_stars(p_value):
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    else:
        return ''


# List to store series for creating DataFrame
results_list = []
for var in dependent_vars:
    # Define series for storing results for the current formula
    coef_series = pd.Series(name=var)
    stderr_series = pd.Series(name='')
    for key, formula in formulas.items():



        # Run the regression with weights
        model = smf.wls(formula=formula.format(lhs_var=var), data=bg_data, weights=bg_data['Weight'])
        results = model.fit()

        # Extract the coefficient, p-value, and standard error for 'Is_Incorporated'
        coefficient_is_incorporated = results.params['Is_Unincorporated']
        p_value_is_incorporated = results.pvalues['Is_Unincorporated']
        std_err_is_incorporated = results.bse['Is_Unincorporated']
        stars = significance_stars(p_value_is_incorporated)

        # Store the results in the dictionary
        if key not in results_dict:
            results_dict[key] = {}
        results_dict[key][var] = {
            'Coefficient': coefficient_is_incorporated,
            'P-Value': p_value_is_incorporated,
            'StdErr': std_err_is_incorporated,
            'Stars': stars
        }

        # Append results to the series
        coef_series[key] = f'{coefficient_is_incorporated:.1f}{stars}'
        stderr_series[key] = f'({std_err_is_incorporated:.1f})'

    # Append the series to the results list
    results_list.append(coef_series)
    results_list.append(stderr_series)

# Create DataFrame from results list
latex_table = pd.DataFrame(results_list)

# Replace underscores with spaces in variable name in index
latex_table.index = latex_table.index.str.replace('_', ' ')

# Convert the DataFrame to a LaTeX table
latex_output = latex_table.to_latex(index=True, na_rep='')

# Print the LaTeX table
print(latex_output)

