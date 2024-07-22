# -*- coding: utf-8 -*-

import pandas as pd
import os
import yaml
import numpy as np

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
import cbsa_correlation as cbsa  # cbsa.py

# Load merged data and PCA index
merged_data = pd.read_excel(os.path.join(config['processed_data'], 'Wharton Comparison', 'edited_full_run_with_wharton_index_2018.xlsx'))
pca_index = pd.read_excel(os.path.join(config['processed_data'], 'Model Output', 'latest_combined', 'Overall_Index.xlsx'))

# Extract geoid from Muni column and convert to integer
pca_index["geoid"] = pca_index["Muni"].str[-9:-3].astype(int)

# Merge data and drop unnecessary columns
data = pd.merge(merged_data, pca_index[["geoid", "First_Principal_Component", "Second_Principal_Component"]], on="geoid")
data = data.drop(["Unnamed: 0"] + data.columns[data.columns.str.startswith('Q')].tolist(), axis=1)

# Function to calculate mean of non-null values in a column
def nan_mean(col):
    return col.dropna().mean()

# Create a DataFrame to store correlation results
Correlation = pd.DataFrame(index=['Affordable Housing', 'Less than 1/2 acre', '1/2 to 1 acre', '1 to under 2 acres', '2 acres or more'], columns=['Wharton', 'Our Results', 'Correlation'])

# Process affordable housing data
affordable_housing = data[["A17w", "q9a18"]].replace(2, 0)
Correlation.loc['Affordable Housing', 'Wharton'] = nan_mean(affordable_housing["q9a18"])
Correlation.loc['Affordable Housing', 'Our Results'] = nan_mean(affordable_housing["A17w"])
Correlation.loc['Affordable Housing', 'Correlation'] = affordable_housing.dropna().corr()["A17w"][1]

# Process minimum lot size data
max_min_lot = data[["A28Max", "q7b18"]]
max_min_lot["A28Max"] = pd.to_numeric(max_min_lot["A28Max"], errors='coerce')
max_min_lot["A28Max"] = max_min_lot["A28Max"].apply(lambda x: x * 43560 if pd.notnull(x) and x <= 10 else x)
max_min_lot["A28Max"] = pd.cut(max_min_lot["A28Max"], bins=[0, 0.5 * 43560, 1 * 43560, 2 * 43560, float('inf')], labels=[1, 2, 3, 4], right=False)
max_min_lot["A28Max"] = max_min_lot["A28Max"].cat.codes + 1

# Drop na in max_min_lot
max_min_lot = max_min_lot.dropna()

# Map columns to indicators and calculate correlations
lot_size_labels = ['Less than 1/2 acre', '1/2 to 1 acre', '1 to under 2 acres', '2 acres or more']
for i, label in enumerate(lot_size_labels, start=1):
    max_min_lot[f"Wharton_{i}"] = (max_min_lot["q7b18"] == i).astype(int)
    max_min_lot[f"Our_Results_{i}"] = (max_min_lot["A28Max"] == i).astype(int)

    Correlation.loc[label, 'Wharton'] = nan_mean(max_min_lot[f"Wharton_{i}"])
    Correlation.loc[label, 'Our Results'] = nan_mean(max_min_lot[f"Our_Results_{i}"])

    corr_data = max_min_lot[[f"Wharton_{i}", f"Our_Results_{i}"]].dropna()
    if not corr_data.empty:
        Correlation.loc[label, 'Correlation'] = corr_data.corr().iloc[0, 1]
    else:
        Correlation.loc[label, 'Correlation'] = np.nan

# Create a multi-index for the rows
row_index = pd.MultiIndex.from_tuples([('Affordable Housing', ''),
                                        ('Minimum Lot Size', 'Less than 1/2 acre'),
                                        ('Minimum Lot Size', '1/2 to 1 acre'),
                                        ('Minimum Lot Size', '1 to under 2 acres'),
                                        ('Minimum Lot Size', '2 acres or more')])
Correlation.index = row_index

latex_table1 = Correlation.style.format(na_rep='-', precision=2).to_latex(column_format='llccc', hrules=True, multirow_align='t')

# Remove extra backslashes from the generated LaTeX code
latex_table1 = latex_table1.replace('\\\\', '\\')

with open(os.path.join(config['tables_path'], 'latex', 'Table 4a - Comparison With Wharton.tex'), 'w') as f:
    f.write(latex_table1)

# Create a correlation matrix for the second LaTeX table
corr_matrix = data[["WRLURI18", "First_Principal_Component", "Second_Principal_Component"]].corr()
corr_matrix = corr_matrix.rename(columns={"WRLURI18": "Wharton Index", "First_Principal_Component": "PC 1", "Second_Principal_Component": "PC 2"}, index={"WRLURI18": "Wharton Index", "First_Principal_Component": "PC 1", "Second_Principal_Component": "PC 2"})

# Export the second LaTeX table
latex_table2 = corr_matrix.style.format(na_rep='-', precision=2).to_latex(column_format='p{3cm}p{3cm}p{3cm}p{3cm}', hrules = True)
with open(os.path.join(config['tables_path'], 'latex', 'Table 4b - Correlation Matrix.tex'), 'w') as f:
    f.write(latex_table2)