import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

binary_qs = config['yes_no_qs']

#make binary_qs list of strings
binary_qs = [str(q) for q in binary_qs]

#Make list of questions with compound clauses
compound_qs = ['13','14','20','21']

#Draw in sample data
sample_data = pd.read_excel(os.path.join(config['processed_data'],'Sample_Enriched.xlsx'))

#Read in the dataframe
dataset = 'Testing Sample'
df = pd.read_excel(os.path.join(config['processed_data'],'Model Output',dataset,'Light Data Merged Adjusted.xlsx'))

#Merge in 'POPULATION' on CENSUS_ID_PID6
df = df.merge(sample_data[['CENSUS_ID_PID6','POPULATION']],on = 'CENSUS_ID_PID6')

#Bring in reading stats
reading_stats = pd.read_excel(os.path.join(config['raw_data'],'Muni Reading Stats.xlsx'))

'''
Merge is failing here
'''

#Merge in length, mean_reading_level and std_dev_reading_level on CENSUS_ID_PID6
df = df.merge(reading_stats,on = 'CENSUS_ID_PID6')

#Define columns for whether binary question as int
df['Binary'] = df['Question'].isin(binary_qs).astype(int)

#Define columns for whether question is compound
df['Compound'] = df['Question'].isin(compound_qs).astype(int)

#Need to adjust correct column
df['Correct'] = df['Answer'].astype(str) == df['Answer_pioneer'].astype(str)
df['Correct'] = df['Correct'].astype(int)

#Normalize population by 10 thousands
df['POPULATION'] = df['POPULATION']/10000

#Drop null answers
df = df.dropna(subset = ['Answer'])

# Dictionary to store model formulas
formulas = {
'Ques Char 1': 'Correct ~ Binary',
'Ques Char 3': 'Correct ~ Binary+Compound',
    'Muni Char 1': 'Correct ~ Binary+POPULATION',
'Muni Char 2': 'Correct ~ Binary+length',
'Muni Char 3': 'Correct ~ Binary+mean_reading_level',
'Muni Char 4': 'Correct ~ Binary+mean_reading_level + std_dev_reading_level',
'Muni Char 5': 'Correct ~ Binary+POPULATION + mean_reading_level + std_dev_reading_level + length',
    'Question FE': 'Correct ~ C(Question)',
    'Muni FE': 'Correct ~ C(Muni)',
    'Question + Muni FE': 'Correct ~ C(Question) + C(Muni)'
}

# Fit models based on the formulas
models = []
model_names = []

for name, formula in formulas.items():
    model = smf.ols(formula=formula, data=df).fit()
    models.append(model)
    model_names.append(name)

# Define a function to check if FE is included in the model
def fe_included(model, fe):
    return 'Y' if f'C({fe})' in model.model.formula else 'N'

# Create summary table with custom rows indicating the presence of fixed effects
summary = summary_col(models, stars=True, float_format='%0.2f',
                      model_names=model_names,
                      info_dict={
                          'Question FE': lambda x: fe_included(x, 'Question'),
                          'Muni FE': lambda x: fe_included(x, 'Muni')
                      })

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

# Adjust the table to only display the R-squared and fixed effects inclusion
# This removes the coefficients from the summary table, focusing on your specific requirements
summary.tables[0] = clean_summary(summary.tables[0], order = ['Binary','Compound','POPULATION','length','mean_reading_level','std_dev_reading_level','Question FE','Muni FE','R-squared'])

#Create an index map mapping each RHS variable to a more intuitive name
index_map = {
    'Binary': 'Binary Question',
    'Compound': 'Compound Question',
    'POPULATION': 'Population',
    'length': 'Ordinance Length',
    'mean_reading_level': 'Mean Reading Level',
    'std_dev_reading_level': 'Std Dev of Reading Level',
}

#Rename the indices in summary.tables[0] with function
def rename_indices(df, index_map):
    #Try to rename with index_map, if not found, keep the original index
    df.index = [index_map.get(idx, idx) for idx in df.index]
    return df
summary.tables[0] = rename_indices(summary.tables[0], index_map)

# Print the summary table
print(summary)

# If you need the summary as LaTeX
latex_summary = summary.as_latex()

# Split the LaTeX summary into lines
lines = latex_summary.split('\n')

# Find the index of the second line containing "\hline"
hline_index = [i for i, line in enumerate(lines) if '\\hline' in line][1]

# Extract the cell contents (excluding the header and footer)
cell_lines = lines[hline_index+1:-2]

# Join the cell lines with newline characters
latex_cells = '\n'.join(cell_lines)

# Replace the default significance stars with escaped LaTeX stars
latex_cells = latex_cells.replace('*', '^{\\ast}')

# Add the modified table header
table_header = '''\\begin{tabular}{@{} l *{10}{d{2.5}} @{}} 
\\toprule
& \\multicolumn{2}{c}{\\begin{tabular}[c]{@{}c@{}}Question \\\\ Characteristics\\end{tabular}} & \\multicolumn{5}{c}{\\begin{tabular}[c]{@{}c@{}}Municipality \\\\ Characteristics\\end{tabular}} & \\multicolumn{3}{c}{\\begin{tabular}[c]{@{}c@{}}Fixed \\\\ Effects\\end{tabular}} \\\\ 
\\cmidrule(lr){2-3} \\cmidrule(lr){4-8} \\cmidrule(lr){9-11}
& \\mc{(1)} & \\mc{(2)} & \\mc{(3)} & \\mc{(4)} & \\mc{(5)} & \\mc{(6)} & \\mc{(7)} & \\mc{(8)} & \\mc{(9)} & \\mc{(10)}\\\\ 
\\midrule'''

#Add '\midline' right before 'Question FE'
latex_cells = latex_cells.replace('Question FE', '\\midrule\nQuestion FE')

# Combine the table header and cell contents
latex_table = table_header + '\n' + latex_cells

# Export the LaTeX table to a file
with open(os.path.join(config['tables_path'], 'latex', 'Table 3 LPM of Errors.tex'), 'w') as f:
    f.write(latex_table)




