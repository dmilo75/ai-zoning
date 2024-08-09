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
import xgboost as xgb
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np
from itertools import combinations

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

##

def run_xgboost(df, lhs_var, rhs_vars):
    # Drop rows with missing values in the dependent and independent variables
    df_clean = df.dropna(subset=[lhs_var] + rhs_vars)

    # Extract the dependent and independent variables
    X = df_clean[rhs_vars].values
    y = df_clean[lhs_var].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set parameters for XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # Train the model and track the evaluation results
    evals_result = {}
    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], early_stopping_rounds=10,
                      evals_result=evals_result)

    return model, evals_result


def add_xgboost_to_summary(summary_table, df, lhs_var, rhs_vars, name):
    '''

    add_xgboost_to_summary(summary.tables[0], formula_data['data'], lhs_var, boost_vars, 'XGBoost')
    '''

    # Run the XGBoost model
    model, eval_results = run_xgboost(df, lhs_var, rhs_vars)

    # Get feature importance
    importance_gain = model.get_score(importance_type='gain')

    # Find the maximum importance score
    if importance_gain:
        max_score = max(importance_gain.values())
    else:
        max_score = 1  # Avoid division by zero if no features are important

    # Normalize the scores to a maximum of 100, round, and format as integers
    importance_mapped = {
        variable_map(var): int(round((importance_gain.get(f'f{i}', 0) / max_score) * 100))
        for i, var in enumerate(rhs_vars)
    }

    # Add XGBoost results to summary table
    summary_table[name] = ''

    # Reorder columns so the new column is first
    cols = summary_table.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    summary_table = summary_table[cols]

    # Populate the summary table
    counter = 0
    for var in rhs_vars:
        mapped_var = variable_map(var)
        score = importance_mapped[mapped_var]
        print(mapped_var + " " + str(score))
        summary_table.iloc[counter, 0] = f"{score}"
        counter += 2

    return summary_table



def variable_map(str):

    if 'Imputed' in str:
        return short_questions[str.split('_')[1]]
    elif 'Question' in str:
        return short_questions[str.split('_')[1]]
    else:
        return str

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


def run_lasso(df, lhs_var, rhs_vars, return_rmse=False):
    # Drop rows with missing values in the dependent and independent variables
    df_clean = df.dropna(subset=[lhs_var] + rhs_vars)

    # Extract the dependent and independent variables
    X = df_clean[rhs_vars].values
    y = df_clean[lhs_var].values

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Lasso model with cross-validation to find the best alpha
    lasso = LassoCV(cv=10).fit(X_scaled, y)

    if return_rmse:
        # Predict using the Lasso model
        y_pred = lasso.predict(X)

        # Calculate Mean Squared Error
        mse = mean_squared_error(y, y_pred)

        # Calculate Root Mean Squared Error
        rmse = np.sqrt(mse)

        return lasso, rmse

    return lasso

def run_ols_with_rmse(df, lhs_var, rhs_vars):
    # Drop rows with missing values in the dependent and independent variables
    df_clean = df.dropna(subset=[lhs_var] + rhs_vars)

    # Extract the dependent and independent variables
    X = df_clean[rhs_vars]
    y = df_clean[lhs_var]

    # Add a constant term for the intercept
    X = sm.add_constant(X)

    # Fit OLS model
    ols_model = sm.OLS(y, X).fit()

    # Calculate predictions
    y_pred = ols_model.predict(X)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y, y_pred)

    # Calculate Root Mean Squared Error
    rmse = np.sqrt(mse)

    return rmse

def add_lasso(summary_table, df, lhs_var, rhs_vars, name):
    '''

    add_lasso(summary.tables[0], formula_data['data'], lhs_var, rhs_vars, 'LASSO')
    '''
    # Run the Lasso model
    lasso = run_lasso(df, lhs_var, rhs_vars)

    # Extract coefficients and intercept (Lasso doesn't provide standard errors directly)
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



def run_reg(formulas, lhs_var, rhs_vars,boost_vars, no_se = False):
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

    #Add LASSO
    summary.tables[0] = add_lasso(summary.tables[0], formula_data['data'], lhs_var, rhs_vars, 'LASSO')

    #Add XGBoost
    summary.tables[0] = add_xgboost_to_summary(summary.tables[0], formula_data['data'], lhs_var, boost_vars, 'XGBoost')

    # Print the summary table
    print(summary)

    # If you need the summary as LaTeX
    latex_summary = get_latex(summary,no_se)
    print(latex_summary)


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
    rhs_vars = [col for col in right_columns]

    # Also drop 'Nearest Metro Name'
    rhs_vars = [col for col in rhs_vars if col not in ['Nearest_Metro_Name','state','NAME','Region']]

    return rhs_vars

def prep_df(df,rhs_vars, demean = True):
    # Make a MSA filledna with state column
    df['MSA_State'] = df['CSA_Title'].fillna(df['FIPS_STATE'])

    #Demean by MSA_State
    if demean:
        df = demean_by_msa_state(df,rhs_vars)

    #Now turn every column into z-scores
    for col in rhs_vars:
        try:
            #Calculate mean and std
            mean = df[col].mean()
            std = df[col].std()
            #Calculate z-score
            df[col] = (df[col] - mean) / std
        except:
            #drop var
            df = df.drop(columns = col)

    return df




def demean_by_msa_state(df, vars):
    """
    Demean the variables by MSA_State.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    lhs_var (str): The dependent variable.
    rhs_vars (list): The list of independent variables.

    Returns:
    pd.DataFrame: The dataframe with demeaned variables.
    """
    # Combine lhs_var and rhs_vars
    all_vars = vars

    # Demean the variables by MSA_State
    df_demeaned = df.copy()
    df_demeaned[all_vars] = df.groupby('MSA_State')[all_vars].transform(lambda x: x - x.mean())

    return df_demeaned


##


data = pd.read_excel(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\Model Output\augest_latest\Comprehensive Data.xlsx")

#Draw in imputed data
imputed_data = pd.read_excel(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\Model Output\augest_latest\Overall_Index.xlsx")

#Get CENSUS_ID_PID6 from column Muni between two #'s
imputed_data['CENSUS_ID_PID6'] = imputed_data['Muni'].str.extract(r'#(.*?)#')

#Now drop column Muni and columns with _Component in them
imputed_data = imputed_data.drop(columns = ['Muni']+[col for col in imputed_data.columns if '_Component' in col])

#Add prefix 'Imputed ' to all columns except CENSUS_ID_PID6
imputed_data.columns = ['Imputed '+col if col != 'CENSUS_ID_PID6' else col for col in imputed_data.columns]

#Make census id an int
imputed_data['CENSUS_ID_PID6'] = imputed_data['CENSUS_ID_PID6'].astype(int)

#Merge in
data = data.merge(imputed_data, on = 'CENSUS_ID_PID6')

for question in ['35','27Max','27Mean','27Min']:
    try:
        data = data.drop(columns = f'Question_{question}')
        data = data.drop(columns = f'Imputed_{question}')
    except:
        pass

##Define all variables
rhs_vars = get_variables(data)

#Prep dataframe into z-scores
df = prep_df(data.copy(),rhs_vars, demean = True)


#'Median_Home_Value_2022'
#'All_Unit_Permits'
#'Median_Gross_Rent_2022'
#Housing_Unit_Density
#Combined_Affordable_Units
lhs_vars = ['Median_Home_Value_2022', 'All_Unit_Permits','Median_Gross_Rent_2022','Housing_Unit_Density','Combined_Affordable_Units']


rhs_vars = [x for x in list(df.columns) if 'Imputed_' in x]
boost_vars = [x for x in list(df.columns) if 'Question_' in x]

#Drop 28Mean from boost_vars
boost_vars = [x for x in boost_vars if x != 'Question_28Mean']

# List to store all regression dataframes
all_reg_dfs = []

for lhs_var in lhs_vars:
    # Dictionary for formulas
    formulas = {
        'Bivariate': {'formula': f'{lhs_var} ~ {{bivariate_var}}', 'data': df},
    }

    reg_df = run_reg(formulas, lhs_var, rhs_vars, boost_vars, no_se=False)

    # Add the lhs_var as the top level of a MultiIndex
    reg_df.columns = pd.MultiIndex.from_product([[lhs_var], reg_df.columns])

    all_reg_dfs.append(reg_df)

# Concatenate all dataframes horizontally
final_df = pd.concat(all_reg_dfs, axis=1)

#Drop rows with empty string index or index of Intercept, R-squared or N
final_df = final_df[~final_df.index.isin(['Intercept', 'R-squared', 'N']) & (final_df.index != '')]

##
xgboost_columns = [col for col in final_df.columns.get_level_values(1) if 'XGBoost' in col]

# Function to convert to numeric, replacing any non-numeric values with NaN
def safe_numeric_convert(x):
    try:
        return pd.to_numeric(x)
    except ValueError:
        return np.nan

# Convert XGBoost columns to numeric
for col in final_df.columns:
    if 'XGBoost' in col[1]:
        final_df[col] = final_df[col].apply(safe_numeric_convert)

# Calculate the average of XGBoost columns for each row
final_df['XGBoost_Average'] = final_df.xs('XGBoost', level=1, axis=1).mean(axis=1)

# Sort the dataframe by the XGBoost average in descending order
final_df_sorted = final_df.sort_values('XGBoost_Average', ascending=False)

# Remove the temporary XGBoost_Average column
final_df_sorted = final_df_sorted.drop('XGBoost_Average', axis=1)

#Get latex string
latex_summary = final_df_sorted.to_latex()


##

def find_rmse_with_without_controls(df, lhs_var, rhs_var_dict):
    results = []

    for spec_name, rhs_vars in rhs_var_dict.items():
        # Run XGBoost
        model, evals = run_xgboost(df, lhs_var, rhs_vars)
        rmse_xgb = evals['test']['rmse'][-1]

        # Run Lasso
        lasso, rmse_lasso = run_lasso(df, lhs_var, rhs_vars, return_rmse=True)

        # Run OLS
        rmse_ols = run_ols_with_rmse(df, lhs_var, rhs_vars)

        # Append results
        results.append({
            'Specification': spec_name,
            'Model': 'XGBoost',
            'RMSE': rmse_xgb
        })
        results.append({
            'Specification': spec_name,
            'Model': 'Lasso',
            'RMSE': rmse_lasso
        })
        results.append({
            'Specification': spec_name,
            'Model': 'OLS',
            'RMSE': rmse_ols
        })

    # Create DataFrame
    rmse_df = pd.DataFrame(results)

    #Reshape so that Specification is the columns Model is the row and RMSE are the cells
    rmse_df = rmse_df.pivot(index = 'Model', columns = 'Specification', values = 'RMSE')

    #Order columns by order of specification
    rmse_df = rmse_df[rhs_var_dict.keys()]

    return rmse_df


# List of LHS variables
lhs_vars = ['Median_Home_Value_2022', 'All_Unit_Permits', 'Median_Gross_Rent_2022', 'Housing_Unit_Density',
            'Combined_Affordable_Units']

# Initialize an empty list to store all results
all_results = []

for lhs_var in lhs_vars:
    boost_vars = [col for col in df.columns if 'Imputed_' in col]
    land_controls = ['lsf_1_flat_plains']
    income_control = ['Median_Household_Income_2022']

    # Generate all permutations of the three lists
    control_combinations = {
        'Zoning Regulations': boost_vars,
        'Land Flat Plains': land_controls,
        'Median Household Income': income_control
    }

    # Drop any rows with any missing control data
    df_clean = df.dropna(subset=boost_vars + land_controls + income_control)

    # Create rhs_var_dict with all permutations
    rhs_var_dict = {}
    for n in range(1, len(control_combinations) + 1):
        for combo in combinations(control_combinations.keys(), n):
            name = ' + '.join(combo)
            vars_combination = []
            for key in combo:
                vars_combination += control_combinations[key]
            rhs_var_dict[name] = vars_combination

    rmse_df = find_rmse_with_without_controls(df_clean, lhs_var, rhs_var_dict)

    # Add LHS variable as a column
    rmse_df['LHS Variable'] = lhs_var

    # Append to all_results
    all_results.append(rmse_df.reset_index())

# Combine all results into a single DataFrame
final_df = pd.concat(all_results, ignore_index=True)

# Set up the multi-index
final_df.set_index(['LHS Variable', 'Model'], inplace=True)

# Rename the columns to match the desired output
column_rename = {
    'Regulations': 'Zoning Regulations',
    'Land Controls': 'Land Flat Plains',
    'Income Control': 'Median Household Income',
    'Regulations + Land Controls': 'Zoning Regulations + Land Flat Plains',
    'Regulations + Income Control': 'Zoning Regulations + Median Household Income',
    'Land Controls + Income Control': 'Land Flat Plains + Median Household Income',
    'Regulations + Land Controls + Income Control': 'All Controls'
}
final_df = final_df.rename(columns=column_rename)

# Generate LaTeX table
latex_table = final_df.to_latex(
    float_format="%.2f",
    multirow=True,
)

# Print the LaTeX table
print(latex_table)