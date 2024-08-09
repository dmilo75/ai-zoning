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
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

##

def variable_map(str):

    if str in variable_mapping:
        return variable_mapping[str]

    if 'Question' in str:
        return short_questions[str.split('_')[1]]
    else:
        return str

variable_mapping = {
    'Single_Unit_Permits': 'Building Permits Single Units 2021',
    'Multi_Unit_Permits': 'Building Permits Multi Units 2021',
    'All_Unit_Permits': 'Building Permits All Units 2021',
    'Housing_Elasticity': 'Housing Elasticity',
    'Population_Aged_65_and_Over_2022': 'Share Population 65 and Over',
    'Population_Under_18_2022': 'Share Population Under 18',
    'White_Share_2022': 'White Share',
    'College_Share_2022': 'College Share',
    'job_density_2013': 'Job Density',
    'jobs_highpay_5mi_2015': 'Nearby High Paying Jobs',
    'kfr_weighted': 'Opportunity Index',
    'cs_mn_avg_mth_ol': 'Average Math Test Scores',
'cs_mn_avg_rla_ol': 'Average Reading Test Scores',
'cs_mn_grd_mth_ol': 'Math Learning Rate',
'cs_mn_grd_rla_ol': 'Reading Learning Rate',
    'perflu': 'Percent Eligible for Free Lunch',
    'Property_Tax_Rate_2017': 'Property Tax Rate',
    'yr_incorp': 'Year of Incorporation',
    'percent_democrat': 'Percent Democrat',
    'Percent_Urban': 'Percent Urban Area',
    'Land_Area_Acres_q2': 'Land Area Acres Q2',
    'Land_Area_Acres_q3': 'Land Area Acres Q3',
    'Land_Area_Acres_q4': 'Land Area Acres Q4',
    'Neighbors_25_q2': 'Munis within 25 Miles Q2',
    'Neighbors_25_q3': 'Munis within 25 Miles Q3',
    'Neighbors_25_q4': 'Munis within 25 Miles Q4',
    'Population_Density': 'Population Density',
    'Housing_Unit_Density': 'Housing Unit Density',
    'Combined_Affordable_Units': 'Share Units Affordable',
    'First_PC': 'First Principal Component (Regulatory Complexity)',
    'Second_PC': 'Second Principal Component (Strictness)',
    'Miles_to_Metro_Center': 'Log Miles to Metro Center',
    'Median_Household_Income_2022': 'Median Household Income',
    'Median_Gross_Rent_2022': 'Median Gross Rent',
    'Median_Home_Value_2022': 'Median Home Value',
    'Owner_Occupied_Share_2022': 'Share Units Owner Occupied',
    'Black_Share_2022': 'Black Share',
    'Poverty_Rate_2022': 'Poverty Rate',
    'Born_in_Another_State_Share_2022': 'Born in Another State Share',
    'Foreign_Born_Share_2022': 'Foreign Born Share',
    'Share_Structures_Built_Before_1970_2022': 'Share Structures Built Before 1970',
    'SF_Attached_Share_2022': 'Single-Family Attached Share',
    'Structures_2_or_More_Share_2022': 'Share Structures with 2 or More Units',
    'Mobile_Home_Boat_RV_Van_Share_2022': 'Mobile Home/Boat/RV/Van Share',
    'Car_Truck_Van_Share_2022': 'Auto Commute Share',
    'Vacancy_Rate_2022': 'Vacancy Rate',
    'Share_Paying_More_Than_30_Percent_Rent_2022': 'Share Paying More Than 30% Rent',
    'Share_Commute_Over_30_Minutes_2022': 'Share with Commute Over 30 Minutes',
    'Cen_Staff_Total_Expend_Per_Capita_2017': 'Central Staff Total Expenditure Per Capita (2017)',
    'Total_Revenue_Per_Capita_2017': 'Total Revenue Per Capita (2017)',
    'Total_Expenditure_Per_Capita_2017': 'Total Expenditure Per Capita (2017)',
    'Total_IGR_Hous_Com_Dev_Per_Capita_2017': 'Total IGR Housing Community Development Per Capita (2017)',
'gamma01b_units_FMM': 'Housing Unit Elasticity',
    'Trct_FrcDev_01': 'Share Land Developed (2001)',
    'Trct_FrcDev_01_squared': 'Squared Share Land Developed (2001)',
    'lsf_1_flat_plains': 'Share Land Flat Plains',
    'gamma01b_newunits_FMM': 'New Housing Unit Elasticity',
    'Neighbors_25':'Log Neighbors within 25 Miles',
    'Land_Area_Acres':'Log Land Area',
    'Local_Revenue_Per_Student':'Local Revenue Per Student',
}
# Updated short questions mapping
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
    '17w': 'Affordable Mandate',
    '17': 'Affordable Incentive',
    '18': 'Inclusionary Zoning Adopted',
    '19': 'Affordable Units Built',
    '20': 'Permit Cap Or Phasing',
    '21': 'Wetlands Restricted in Lot Size Calc',
    '22': 'Longest Frontage Requirement',
    '23': 'Frontage Measurement',
    '24': 'Lot Shape Requirements',
    '25': 'Height Measurement',
    '26': 'Additional Zoning Notes',
    '28Mean': 'Mean Res Min Lot Size',
    '28Min': 'Minimum Res Min Lot Size',
    '28Max': 'Maximum Res Min Lot Size',
    '30': 'Mandatory Approval Steps',
    '31': 'Distinct Approval Bodies',
    '32': 'Public Hearing Requirements',
    '34': 'Max Review Waiting Time',
    '36': 'Parking Requirement',
    '37': 'Garage Requirement',
    '38': 'Single Family Parking Spots',
    '39': 'Multi 600sqft Parking Spots',
    '40': 'Multi 1000sqft Parking Spots'
}


def add_lasso(summary_table, df, lhs_var, rhs_vars, name):
    # Impute missing values using KNN
    imputer = KNNImputer(n_neighbors=50)
    X_imputed = imputer.fit_transform(df[rhs_vars])
    y = df[lhs_var].values

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Fit Lasso model with cross-validation to find the best alpha
    lasso = LassoCV(cv=10).fit(X_scaled, y)

    # Extract coefficients and standard errors (Lasso doesn't provide standard errors directly)
    lasso_coefs = lasso.coef_
    lasso_intercept = lasso.intercept_

    # Add Lasso results to summary table
    summary_table[name] = ''

    # Reorder columns so the new column is first
    cols = summary_table.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    summary_table = summary_table[cols]

    # Populate the summary table
    counter = 0
    for var, coef in zip(rhs_vars, lasso_coefs):
        if coef != 0:  # Only add non-zero coefficients
            summary_table.iloc[counter, 0] = f"{coef:.2f}"
            counter += 2

    # Add intercept at the bottom
    summary_table.iloc[counter, 0] = f"{lasso_intercept:.2f}"

    return summary_table


def clean_summary(summary_table, order=None):
    # Split DataFrame into sub-DataFrames, each containing a coefficient and its standard error (if present)
    sub_dfs = []  # List to hold sub-DataFrames
    current_sub_df = []  # Temporary storage for the current sub-DataFrame
    for index, row in summary_table.iterrows():
        if index.strip():  # Non-blank index, start of a new sub-DataFrame
            if current_sub_df:  # If there's an ongoing sub-DataFrame, save it
                sub_dfs.append(pd.DataFrame(current_sub_df))
                current_sub_df = []  # Reset for the next sub-DataFrame
            current_sub_df.append(row)  # Add the non-blank row to the current sub-DataFrame
        else:  # Blank index, part of the current sub-DataFrame
            current_sub_df.append(row)  # Add the blank row to the current sub-DataFrame
    if current_sub_df:  # Ensure the last sub-DataFrame is added
        sub_dfs.append(pd.DataFrame(current_sub_df))

    # Filter out fixed effects coefficients
    filtered_sub_dfs = [df for df in sub_dfs if not df.index[0].startswith('C(')]

    # Reorder if necessary
    if order:
        # Assuming 'order' is a list of indices in the desired order
        # Create a mapping of indices to sub-DataFrames
        index_to_df = {df.index[0]: df for df in filtered_sub_dfs}
        # Reorder according to 'order', filtering out any indices not present
        reordered_sub_dfs = [index_to_df[idx] for idx in order if idx in index_to_df]
    else:
        reordered_sub_dfs = filtered_sub_dfs

    # Concatenate the sub-DataFrames back into a single DataFrame
    cleaned_summary_table = pd.concat(reordered_sub_dfs)

    return cleaned_summary_table


def add_bivariate(summary_table, formula_structure, df,independents,name):



    # Step 1: Run Bivariate Regressions
    bivariate_results = {}
    for var in independents:
        # Define the formula
        formula = formula_structure.format(bivariate_var = var)

        # Fit the model
        model = smf.ols(formula, data=df).fit()
        bivariate_results[var] = model

    # Step 2: Extract the coefficients and standard errors
    bivariate_coefs = {var: model.params[var] for var, model in bivariate_results.items()}
    bivariate_se = {var: model.bse[var] for var, model in bivariate_results.items()}
    bivariate_pvalues = {var: model.pvalues[var] for var, model in bivariate_results.items()}

    # Step 3: Add a new column and locate it as the first column
    summary_table[name] = ''

    # Reorder columns so the new column is first
    cols = summary_table.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    summary_table = summary_table[cols]

    # Step 4: Populate the summary table
    counter = 0
    for var in independents:
        # Determine how many stars to add to the coefficient
        stars = ''
        if bivariate_pvalues[var] < 0.01:
            stars = '***'
        elif bivariate_pvalues[var] < 0.05:
            stars = '**'
        elif bivariate_pvalues[var] < 0.1:
            stars = '*'
        summary_table.iloc[counter, 0] = f"{bivariate_coefs[var]:.2f}{stars}"
        summary_table.iloc[counter + 1, 0] = f"({bivariate_se[var]:.2f})"
        counter += 2

    return summary_table


# Custom function to extract the number of observations
def get_nobs(model):
    return f"{int(model.nobs)}"

def get_latex(summary, no_se):

    #First, gen the regular latex
    latex_summary = summary.as_latex()

    #Now, isolate the cells between the second '\hline' and last '\hline'
    latex_cells = latex_summary.split('\hline')[2].strip()

    #Now loop over each line, if the line is followed by another line that has a first non-whitespace character of & then we increment the counter by 2, otherwise we add a midrule before the line
    lines = latex_cells.split('\n')
    added_mid = False
    i = 0
    while i + 1 < len(lines):
        if lines[i+1].strip()[0] == '&':
            i = i + 2
        else:
            lines[i] = '\\midrule\n'+lines[i]
            added_mid = True
            break

    if not added_mid:
        if lines[-1].strip()[0] != '&':
            lines[-1] ='\\midrule\n'+ lines[-1]

    if no_se:
        #Remove lines whose first non-space char is '&'
        lines = [line for line in lines if line.strip()[0] != '&']


    return '\n'.join(lines)+'\n\\bottomrule'



def run_reg(formulas, lhs_var, rhs_vars, no_se = False):
    # Fit models based on the formulas
    models = []
    model_names = []



    for name, formula_data in formulas.items():
        if 'Bivariate' in name:
            continue

        model = smf.ols(formula=formula_data['formula'], data=formula_data['data']).fit()
        models.append(model)
        model_names.append(name)

    if len(models) == 0:
        fake_formula = f"{lhs_var} ~ " + ' + '.join(rhs_vars)
        fake_model = smf.ols(formula=fake_formula, data=formulas[name]['data']).fit()
        models.append(fake_model)
        model_names.append(name + '_fake')

    # Create summary table with custom rows indicating the presence of fixed effects
    summary = summary_col(models, stars=True, float_format='%0.2f',
                          model_names=model_names, info_dict={
            'N': get_nobs,  # Add the number of observations using the custom function
        })

    # Adjust the table to only display the R-squared and fixed effects inclusion
    # This removes the coefficients from the summary table, focusing on your specific requirements
    summary.tables[0] = clean_summary(summary.tables[0], order=rhs_vars + ['Intercept', 'R-squared', 'N'])

    # Loop over again and add bivariates
    for name, formula_data in formulas.items():
        if 'Bivariate' in name:
            summary.tables[0] = add_bivariate(summary.tables[0], formula_data['formula'], formula_data['data'], rhs_vars, name)

    # Replace the index with the variable names but if not in dictionary then keep the index
    summary.tables[0].index = [variable_map(index) for index in summary.tables[0].index]

    #Remove columns with _fake from the summary table
    summary.tables[0] = summary.tables[0].loc[:,~summary.tables[0].columns.str.contains('_fake')]

    # Print the summary table
    print(summary)

    # If you need the summary as LaTeX
    latex_summary = get_latex(summary,no_se)
    print(latex_summary)

    # Print heatmap
    clean_and_plot_heatmap(summary.tables[0], lhs_var)

    return summary.tables[0]

    # Save the summary table to a file
    # with open(os.path.join(config['tables_path'],'latex', output_file_name), 'w') as f:
    #     f.write(latex_summary)



def clean_and_plot_heatmap(df,lhs_var):
    # Separate 'R-squared' and 'N' rows
    r_squared = df.loc['R-squared']
    n_values = df.loc['N']

    # Remove 'Intercept', 'R-squared', and 'N' rows
    keep_indices = ['Intercept', 'R-squared', 'N']
    df_cleaned = df[~df.index.isin(keep_indices) & df.index.notnull() & (df.index != '')]

    # Function to extract numeric value and check for three stars
    def extract_value(value):
        if isinstance(value, str):
            stars = value.count('*')
            if stars >=2:
                numeric_value = value.split('*')[0].strip('()')
                try:
                    return float(numeric_value)
                except ValueError:
                    return np.nan
        return np.nan

    # Function to remove stars from labels
    def remove_stars(value):
        if isinstance(value, str):
            return value.replace('*', '')
        return value

    # Apply the functions to the dataframe
    value_df = df_cleaned.applymap(extract_value)
    label_df = df_cleaned.applymap(remove_stars)

    # Plotting heatmap with shades of red based on value
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure height
    sns.heatmap(value_df.abs(), annot=label_df, fmt="", cmap="Reds",
                cbar_kws={'label': 'Value Intensity'}, linewidths=.5, ax=ax)

    plt.title(f'Heatmap of Coefficients Siginficant at the 95% Level for {lhs_var}\nCoefficients on Normalized Variables')

    # Adjusting the plot to add padding for labels and ensure all labels are shown
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', va = 'top')
    ax.tick_params(axis='y', pad=10)

    # Ensure all y-axis labels are shown
    ax.set_yticks(range(len(df_cleaned.index)))
    ax.set_yticklabels(df_cleaned.index)

    # Modify x-axis labels to include R-squared and N for each column
    current_xlabels = ax.get_xticklabels()
    new_xlabels = [f"{label.get_text()}\nR²: {r_squared[i]}, N: {n_values[i]}"
                   for i, label in enumerate(current_xlabels)]
    ax.set_xticklabels(new_xlabels)
    ax.tick_params(axis='x', rotation=45, labelsize=8)  # Adjust rotation and font size as needed

    plt.tight_layout()
    plt.show()

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


def get_variables(df):

    #Replace all spaces with _ in column names
    df.columns = df.columns.str.replace(' ', '_').str.replace('-','_')

    #Rename %_Urban to Percent_Urban
    df.rename(columns={'%_Urban': 'Percent_Urban'}, inplace=True)

    # Identify the position of the 'CSA Title' column
    csa_title_index = df.columns.get_loc('CSA_Title')

    # Identify the columns to the right of 'CSA title'
    right_columns = df.columns[(csa_title_index + 1):]

    # Remove any columns starts with Q or ending in _PC
    rhs_vars = [col for col in right_columns if not col.startswith('Q') and not col.endswith('_PC')]

    #Rest of the variables are on the left
    lhs_vars = [col for col in right_columns if col not in rhs_vars]

    # Also drop 'Nearest Metro Name'
    rhs_vars = [col for col in rhs_vars if col != 'Nearest_Metro_Name']

    return rhs_vars, lhs_vars

def prep_df(df,rhs_vars,lhs_vars):

    #Make certain variables logs first
    for var in ['Neighbors_25','Land_Area_Acres','Miles_to_Metro_Center']:
        df[var] = np.log(df[var]+1)

    #Now turn every column into z-scores
    for col in rhs_vars + lhs_vars:
        try:
            #Calculate mean and std
            mean = df[col].mean()
            std = df[col].std()
            #Calculate z-score
            df[col] = (df[col] - mean) / std
        except:
            #drop var
            df = df.drop(columns = col)

    #Make some new variables
    def create_quartile_indicators(df, var_name):
        for q in range(1, 5):
            df[f'{var_name}_q{q}'] = ((df[var_name] >= df[var_name].quantile((q - 1) / 4))  &
                                      (df[var_name] <= df[var_name].quantile(q / 4))).astype(int)
        return df

    #Make indicators for num neighbors quartiles and for land area
    df = create_quartile_indicators(df,'Land_Area_Acres')
    df = create_quartile_indicators(df,'Neighbors_25')

    #Make unique county ids from FIPS_STATE and FIPS_COUNTY
    df['County_ID'] = df['FIPS_STATE'].astype(str).str.zfill(2)+'#'+df['FIPS_COUNTY'].astype(str).str.zfill(3)

    return df


def fips_to_region(fips_state):
    northeast = {9, 23, 25, 33, 34, 36, 42, 44, 50}
    midwest = {17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55}
    south = {1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54}
    west = {2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56}

    if fips_state in northeast:
        return "Northeast"
    elif fips_state in midwest:
        return "Midwest"
    elif fips_state in south:
        return "South"
    elif fips_state in west:
        return "West"
    else:
        return "Unknown"



##


data = pd.read_excel(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\Model Output\augest_latest\Comprehensive Data.xlsx")


##Define all variables
rhs_vars, lhs_vars = get_variables(data)

#Prep dataframe into z-scores
df = prep_df(data.copy(),rhs_vars,lhs_vars)

#Make a MSA filledna with state column
df['MSA_State'] = df['CSA_Title'].fillna(df['FIPS_STATE'])


# Variables dictionary
variables_dict = {
    'Chty': ['job_density_2013','kfr_weighted'],
    'Educ':['cs_mn_avg_mth_ol','cs_mn_grd_mth_ol','perflu','Local_Revenue_Per_Student'],
    'Gov': ['Property_Tax_Rate_2017','Total_Expenditure_Per_Capita_2017'],
    'Perm': ['All_Unit_Permits'],
    'Date': ['yr_incorp'],
    'Vote': ['percent_democrat'],
    'Spat': ['Land_Area_Acres','Neighbors_25','Housing_Unit_Density'],
    'Dist': ['Miles_to_Metro_Center'],
    'Afford':['Combined_Affordable_Units'],
}


# ACS variables
acs_vars = [
'Car_Truck_Van_Share_2022',
'Foreign_Born_Share_2022',
    'Median_Household_Income_2022',
'Population_Aged_65_and_Over_2022',
    'Median_Gross_Rent_2022',
    'Median_Home_Value_2022',
    'Owner_Occupied_Share_2022',
    'Population_Under_18_2022',
    'White_Share_2022',
    'Poverty_Rate_2022',
    'College_Share_2022',
    'Share_Structures_Built_Before_1970_2022',
    'Structures_2_or_More_Share_2022',
    'Vacancy_Rate_2022',
    'Share_Commute_Over_30_Minutes_2022'
]

lhs_var = 'First_PC'

# Dictionary for formulas
# Dictionary for formulas
formulas = {
    'Bivariate': {'formula':f'{lhs_var} ~ {{bivariate_var}}' ,'data':df},
    'Bivariate FE': {'formula':f'{lhs_var} ~ {{bivariate_var}}+C(MSA_State)' ,'data':df},
    'All': {'formula':f'{lhs_var} ~ ' + ' + '.join(acs_vars + sum(variables_dict.values(), [])),'data':df},
'All FE': {'formula':f'{lhs_var} ~ C(MSA_State)+' + ' + '.join(acs_vars + sum(variables_dict.values(), [])),'data':df},
}
rhs_vars = []



rhs_vars.extend(acs_vars + sum(variables_dict.values(), []))

reg_df = run_reg(formulas, lhs_var,rhs_vars, no_se = False)



##Second PC

#Define all variables
rhs_vars, lhs_vars = get_variables(data)

#Prep dataframe into z-scores
df = prep_df(data.copy(),rhs_vars,lhs_vars)

#Make a MSA filledna with state column
df['MSA_State'] = df['CSA_Title'].fillna(df['FIPS_STATE'])

# Variables dictionary
variables_dict = {
    'Chty': ['job_density_2013','kfr_weighted'],
    'Educ':['cs_mn_avg_mth_ol','cs_mn_grd_mth_ol','perflu','Local_Revenue_Per_Student'],
    'Gov': ['Property_Tax_Rate_2017','Total_Expenditure_Per_Capita_2017'],
    'Perm': ['All_Unit_Permits'],
    'Date': ['yr_incorp'],
    'Vote': ['percent_democrat'],
    'Spat': ['Land_Area_Acres','Neighbors_25','Housing_Unit_Density'],
    'Dist': ['Miles_to_Metro_Center'],
    'Afford':['Combined_Affordable_Units'],
}


# ACS variables
acs_vars = [
'Car_Truck_Van_Share_2022',
'Foreign_Born_Share_2022',
    'Median_Household_Income_2022',
'Population_Aged_65_and_Over_2022',
    'Median_Gross_Rent_2022',
    'Median_Home_Value_2022',
    'Owner_Occupied_Share_2022',
    'Population_Under_18_2022',
    'White_Share_2022',
    'Poverty_Rate_2022',
    'College_Share_2022',
    'Share_Structures_Built_Before_1970_2022',
    'Structures_2_or_More_Share_2022',
    'Vacancy_Rate_2022',
    'Share_Commute_Over_30_Minutes_2022'
]

lhs_var = 'Second_PC'


#['gamma01b_units_FMM', 'Trct_FrcDev_01','Trct_FrcDev_01_squared', 'lsf_1_flat_plains','gamma01b_newunits_FMM']

# Dictionary for formulas
formulas = {
    'Bivariate': {'formula':f'{lhs_var} ~ {{bivariate_var}}' ,'data':df},
    'Bivariate FE': {'formula':f'{lhs_var} ~ {{bivariate_var}}+C(MSA_State)' ,'data':df},
    'All': {'formula':f'{lhs_var} ~ ' + ' + '.join(acs_vars + sum(variables_dict.values(), [])),'data':df},
'All FE': {'formula':f'{lhs_var} ~ C(MSA_State)+' + ' + '.join(acs_vars + sum(variables_dict.values(), [])),'data':df},
    #'elast': {'formula':f'{lhs_var} ~ gamma01b_newunits_FMM+Trct_FrcDev_01+Trct_FrcDev_01_squared+lsf_1_flat_plains+Miles_to_Metro_Center','data':df},
#'elast MSA FE': {'formula':f'{lhs_var} ~ gamma01b_newunits_FMM+C(MSA_State)+Trct_FrcDev_01+Trct_FrcDev_01_squared+lsf_1_flat_plains+Miles_to_Metro_Center','data':df},
}

rhs_vars = []



rhs_vars.extend(acs_vars + sum(variables_dict.values(), []))

reg_df = run_reg(formulas, lhs_var,rhs_vars)

##Elasticity measures

# Dictionary for formulas
formulas = {
    'first elast': {'formula':f'First_PC ~ gamma01b_newunits_FMM+Trct_FrcDev_01+Trct_FrcDev_01_squared+lsf_1_flat_plains+Miles_to_Metro_Center','data':df},
'first elast FE': {'formula':f'First_PC ~ gamma01b_newunits_FMM+C(MSA_State)+Trct_FrcDev_01+Trct_FrcDev_01_squared+lsf_1_flat_plains+Miles_to_Metro_Center','data':df},
    'second elast': {'formula':f'Second_PC ~ gamma01b_newunits_FMM+Trct_FrcDev_01+Trct_FrcDev_01_squared+lsf_1_flat_plains+Miles_to_Metro_Center','data':df},
'second elast FE': {'formula':f'Second_PC ~ gamma01b_newunits_FMM+C(MSA_State)+Trct_FrcDev_01+Trct_FrcDev_01_squared+lsf_1_flat_plains+Miles_to_Metro_Center','data':df},
}
rhs_vars = ['gamma01b_newunits_FMM','Trct_FrcDev_01','Trct_FrcDev_01_squared','lsf_1_flat_plains','Miles_to_Metro_Center']

#First we run for First
lhs_var = 'First_PC'
reg_df = run_reg(formulas, lhs_var,rhs_vars)


##

import matplotlib.pyplot as plt

def plot_regression_bar_chart(reg_df,title):
    # Drop specified indices
    df = reg_df.copy().drop(index=['', 'Intercept', 'R-squared', 'N', 'New Housing Unit Elasticity'], errors='ignore')

    # Remove asterisks from the 'Bivariate FE' column values
    df['Bivariate FE'] = df['Bivariate FE'].str.replace('*', '')

    # Make numeric
    df['Bivariate FE'] = df['Bivariate FE'].astype(float)

    # Keep only the 'Bivariate FE' column
    df = df[['Bivariate FE']]

    # Sort the dataframe
    df = df.sort_values(by='Bivariate FE', ascending=False)

    # Plot a horizontal bar chart
    plt.figure(dpi = 300)

    #Set figsize
    plt.figure(figsize=(7, 6))

    df['Bivariate FE'].astype(float).plot(kind='barh')
    plt.xlabel('Coefficient on Normalized Variable')
    plt.ylabel('Independent Variable')
    plt.title(title)
    plt.tight_layout()

    #Save fig
    plt.savefig(os.path.join(config['figures_path'],'Slides - Reg Bar PC1'), dpi = 300)

    plt.show()

plot_regression_bar_chart(reg_df,  'Bivariate Regression w/ MSA FE')


##Distance to Center
from copy import deepcopy

#All units permits
'''
'All_Units_Permits'
'Median_Household_Income_2022',
'Median_Gross_Rent_2022',
'Median_Home_Value_2022',
'Miles_to_Metro_Center',
'''

rhs_vars, lhs_vars = get_variables(data)

#Prep dataframe into z-scores
df = prep_df(data.copy(),rhs_vars,lhs_vars)


df['Miles_to_Metro_Center'] = data['Miles_to_Metro_Center']

#Filter out those above 50
#df = df[df['Miles_to_Metro_Center'] < 50]

#Make distance in log terms
df['Miles_to_Metro_Center'] = np.log(df['Miles_to_Metro_Center'])

#Make region var
df['Region'] = df['FIPS_STATE'].apply(fips_to_region)


lhs_var = 'Miles_to_Metro_Center'

#Drop question 35, and any with 27 in it
df = df.drop(columns = ['Question_35'])
df = df[[col for col in df.columns if 'Question_27' not in col]]

# Variables dictionary
variables_dict_quesiton = {
'Questions':[x for x in list(df.columns) if 'Question' in x],
'PCAs':['First_PC','Second_PC']
}


new_vars = variables_dict_quesiton['PCAs']+variables_dict_quesiton['Questions']

# Dictionary for formulas
formulas = {
    'Bivariate': {'formula':f'{lhs_var} ~ {{bivariate_var}} + C(Nearest_Metro_Name)','data':deepcopy(df)},
    'Bivariate Northeast': {'formula':f'{lhs_var} ~ {{bivariate_var}} + C(Nearest_Metro_Name)','data':df[df['Region'] == 'Northeast']},
    'Bivariate Midwest': {'formula':f'{lhs_var} ~ {{bivariate_var}} + C(Nearest_Metro_Name)','data':df[df['Region'] == 'Midwest']},
    'Bivariate South': {'formula':f'{lhs_var} ~ {{bivariate_var}} + C(Nearest_Metro_Name)','data':df[df['Region'] == 'South']},
    'Bivariate West': {'formula':f'{lhs_var} ~ {{bivariate_var}} + C(Nearest_Metro_Name)','data':df[df['Region'] == 'West']},
}
new_rhs_vars = sum(variables_dict_quesiton.values(), [])


run_reg(formulas, lhs_var,new_rhs_vars)

##Unincorporated regression

