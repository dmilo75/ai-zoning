import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os
import yaml

model = 'national_run_may'

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

variable_mapping = {
'Median_Household_Income': 'Median Household Income (2021)',
'Median_Household_Income_Percent_Change': '% Change, 2021-2010 Median Household Income',
'Urbanization': '% Urban',
'Median_Home_Value': 'Median Home Value (2021)',
'Median_Gross_Rent': 'Median Gross Rent (2021)',
'Median_Home_Value_Percent_Change': 'Median Home Value % Change, 2021-2010',
'Median_Gross_Rent_Percent_Change': 'Median Gross Rent % Change, 2021-2010 ',
'Single_Unit_Permits': 'Building Permits Single Units 2021',
'Multi_Unit_Permits': 'Building Permits Multi Units 2021',
'All_Unit_Permits': 'Building Permits All Units 2021',
'Housing_Elasticity': 'Housing Elasticity',
}


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

def add_bivariate(summary_table,independents, dependent,df):

    # Step 1: Run Bivariate Regressions
    bivariate_results = {}
    for var in independents:
        model = smf.ols(f"{dependent} ~ {var}", data=df).fit()
        bivariate_results[var] = model

    # Step 2: Extract the coefficients and standard errors
    bivariate_coefs = {var: model.params[var] for var, model in bivariate_results.items()}
    bivariate_se = {var: model.bse[var] for var, model in bivariate_results.items()}
    bivariate_pvalues = {var: model.pvalues[var] for var, model in bivariate_results.items()}

    #Now, add a new column called 'Bivariate' and locate it as the first column
    summary_table['Bivariate'] = ''

    #Reorder columns so its first
    cols = summary_table.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    summary_table = summary_table[cols]


    counter = 0
    for var in independents:
        #Determine how many stars to add to coefficient
        stars = ''
        if bivariate_pvalues[var] < 0.01:
            stars = '***'
        elif bivariate_pvalues[var] < 0.05:
            stars = '**'
        elif bivariate_pvalues[var] < 0.1:
            stars = '*'
        summary_table.iloc[counter,0] = f"{bivariate_coefs[var]:.2f}{stars}"
        summary_table.iloc[counter+1,0] = f"({bivariate_se[var]:.2f})"
        counter= counter+2

    return summary_table

# Custom function to extract the number of observations
def get_nobs(model):
    return f"{int(model.nobs)}"

def get_latex(summary):

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

    return '\n'.join(lines)+'\n\\bottomrule'

def run_reg(variables, rename_dict, formulas, df, sample_enriched,output_file_name):
    # Rename the columns
    sample_enriched = sample_enriched.rename(columns=rename_dict)

    # Rename variables list too
    variables = [rename_dict[var] for var in variables]

    # Turn all variables into z-scores
    for var in variables:
        sample_enriched[var] = (sample_enriched[var] - sample_enriched[var].mean()) / sample_enriched[var].std()

    # Merge in the variables
    df = df.merge(sample_enriched[['CENSUS_ID_PID6'] + variables], on='CENSUS_ID_PID6')

    # Fit models based on the formulas
    models = []
    model_names = []

    for name, formula in formulas.items():
        model = smf.ols(formula=formula, data=df).fit()
        models.append(model)
        model_names.append(name)

    # Create summary table with custom rows indicating the presence of fixed effects
    summary = summary_col(models, stars=True, float_format='%0.2f',
                          model_names=model_names, info_dict={
            'N': get_nobs,  # Add the number of observations using the custom function
        })

    # Adjust the table to only display the R-squared and fixed effects inclusion
    # This removes the coefficients from the summary table, focusing on your specific requirements
    summary.tables[0] = clean_summary(summary.tables[0], order=variables + ['Intercept', 'R-squared', 'N'])

    summary.tables[0] = add_bivariate(summary.tables[0], variables, 'Index',df)

    #Replace the index with the variable names but if not in dictionary then keep the index
    summary.tables[0].index = [variable_mapping.get(index,index) for index in summary.tables[0].index]

    # Print the summary table
    print(summary)

    # If you need the summary as LaTeX
    latex_summary = get_latex(summary)
    print(latex_summary)

    # Save the summary table to a file
    with open(os.path.join(config['tables_path'],'latex', output_file_name), 'w') as f:
        f.write(latex_summary)
##


#First, bring in the index
df = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,'Overall_Index.xlsx'))

#Define CENSUS_ID_PID6 by splitting 'Muni' on '#' and taking the second entry
df['CENSUS_ID_PID6'] = df['Muni'].apply(lambda x: int(x.split('#')[1]))

#Second, bring in sample enriched
sample_enriched = pd.read_excel(os.path.join(config['processed_data'],'Sample_Enriched.xlsx'))

#Rename first principal component to index
df = df.rename(columns = {'First_Principal_Component':'Index'})

Table_num = '6'

'''
Have to run twice, once for first and once for second component 
'''

##Supply side housing variables

#First, define variable list that we want to merge in
variables = ['Median Home Value (Owner Occupied)_2021','Median Gross Rent_2021','Median Home Value (Owner Occupied)_Percent_Change','Median Gross Rent_Percent_Change','Single Unit Permits','Multi-Unit Permits','All Unit Permits','weighted_avg_gamma']

#Rename variables to be more concise

#Define a dictionary mapping where each new name doesn't have spaces and is more concise
rename_dict = {
    'Median Home Value (Owner Occupied)_2021':'Median_Home_Value',
    'Median Gross Rent_2021':'Median_Gross_Rent',
    'Median Home Value (Owner Occupied)_Percent_Change':'Median_Home_Value_Percent_Change',
    'Median Gross Rent_Percent_Change':'Median_Gross_Rent_Percent_Change',
    'Single Unit Permits':'Single_Unit_Permits',
    'Multi-Unit Permits':'Multi_Unit_Permits',
    'All Unit Permits':'All_Unit_Permits',
'weighted_avg_gamma':'Housing_Elasticity',
}

# Dictionary to store model formulas
formulas = {
'Price': 'Index ~ Median_Home_Value + Median_Gross_Rent + Median_Home_Value_Percent_Change + Median_Gross_Rent_Percent_Change',
'Permits': 'Index ~ Single_Unit_Permits + Multi_Unit_Permits + All_Unit_Permits',
'All': 'Index ~ Median_Home_Value + Median_Gross_Rent + Median_Home_Value_Percent_Change + Median_Gross_Rent_Percent_Change + Single_Unit_Permits + Multi_Unit_Permits + All_Unit_Permits',
'All and Elasticity': 'Index ~ Median_Home_Value + Median_Gross_Rent + Median_Home_Value_Percent_Change + Median_Gross_Rent_Percent_Change + Single_Unit_Permits + Multi_Unit_Permits + All_Unit_Permits+Housing_Elasticity'

}

run_reg(variables, rename_dict, formulas, df, sample_enriched,f'Table {Table_num}A - Correlation of Zoning Index With Outcomes.tex')


##Now demand side

#First, define variable list that we want to merge in
variables = ['Median Household Income_2021','Median Household Income_Percent_Change','% Urban']


#Rename variables to be more concise
rename_dict = {
    'Median Household Income_2021':'Median_Household_Income',
    'Median Household Income_Percent_Change':'Median_Household_Income_Percent_Change',
    '% Urban':'Urbanization'
}

# Dictionary to store model formulas
formulas = {
'Income': 'Index ~ Median_Household_Income_Percent_Change + Median_Household_Income',
'All': 'Index ~ Median_Household_Income + Median_Household_Income_Percent_Change + Urbanization'

}

#Run the regression
run_reg(variables, rename_dict, formulas, df, sample_enriched, f'Table {Table_num}B - Correlation of Zoning Index With Outcomes.tex')






