#Load config file
import os
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

import pandas as pd
import numpy as np

#%%First, let's draw in all of the data
model = 'latest_combined'

#Questions
questions = pd.read_excel(os.path.join( config['raw_data'],"Questions.xlsx"))

comprehensive_data_path = os.path.join(config['processed_data'], 'Model Output', model, 'Comprehensive Data.xlsx')
# Load comprehensive data
df = pd.read_excel(comprehensive_data_path)



##Get means

# Initialize the sums DataFrame
question_columns = df.filter(regex='^Question').columns
sums = pd.DataFrame(index=question_columns)

# Compute Mean
sums['Mean'] = df[question_columns].mean()

# Compute Population Weighted Mean
population_weighted_mean = (df[question_columns].multiply(df['POPULATION'], axis=0).sum() /
                            df['POPULATION'].sum())
sums['Population Weighted Mean'] = population_weighted_mean

# Get sample size
sums['Sample Size'] = df[question_columns].notna().sum()

# Income categories
df['Income Category'] = pd.qcut(df['Median_Household_Income_2022'], q=3,
                                labels=['Low Income', 'Middle Income', 'High Income'])

# Urban categories
df['Urban Category'] = pd.cut(df['% Urban'], bins=[-np.inf, 0, 99.99, np.inf],
                              labels=['Rural', 'Suburban', 'Urban'])

# Calculate means for each category
categories = ['Low Income', 'Middle Income', 'High Income', 'Rural', 'Suburban', 'Urban']
for category in categories:
    category_col = 'Income Category' if 'Income' in category else 'Urban Category'
    category_mean = df[df[category_col] == category][question_columns].mean()
    sums[f'Mean ({category})'] = category_mean

#%%Get the latex formatted table

# Create a dictionary mapping 'Question X' to the actual question text
question_dict = {f'Question {k}': v for k, v in questions.set_index('ID')['Pioneer Question'].to_dict().items()}

# Add specific mappings for min lot size questions
question_dict.update({
    'Question 28Max': 'Max of Residential Min Lot Sizes',
    'Question 28Mean': 'Mean of Residential Min Lot Sizes',
    'Question 28Min': 'Min of Residential Min Lot Sizes'
})

# Apply the function to adjust the index
sums.index = sums.index.map(lambda x: question_dict.get(x, x))



# Setting max_colwidth to avoid truncation
pd.set_option('display.max_colwidth', None)

yes_no_indices = [question_dict[f'Question {id}'] for id in questions[questions['Question Type'] == 'Binary']['ID']]

# Find the rows that correspond to questions in yes_no_qs
mask = sums.index.isin(yes_no_indices)

# Multiply all columns (excluding 'Sample Size') by 100 for those rows
sums.loc[mask, sums.columns.difference(['Sample Size'])] = sums.loc[mask, sums.columns.difference(['Sample Size'])] * 100


# Adjusting custom_format function to handle new column
def format_value(value, column_name, index):
    """
    Formats the given value based on the column name and index criteria.
    """

    # Check if the column is "Sample Size" or if the value is numeric and at a specific index.
    if column_name == "Sample Size" or (index in yes_no_indices and isinstance(value, (int, float))):
        return '{:,.0f}'.format(value)

    if index.startswith('For a typical new multi-family') or index.startswith('What is the maxium potential'):
        return '{:,.1f}'.format(value)

    # For all other cases, round the value, convert to an integer, and return as a string.
    # This handles non-numeric types by attempting to round, convert to int, and then to string.
    return str(int(round(value)))




# Re-apply custom formatting for each column
for col in sums.columns:
    sums[col] = sums.apply(lambda row: format_value(row[col], col, row.name), axis=1)

sum_dic = {
    'Continuous': sums.drop(index=[idx for idx in yes_no_indices if idx in sums.index]).drop(index=['Max of Residential Min Lot Sizes'] if 'Max of Residential Min Lot Sizes' in sums.index else []),
    'Binary': sums.loc[[idx for idx in yes_no_indices if idx in sums.index]]
}


#Export to Excel
with pd.ExcelWriter(os.path.join(config['tables_path'],'Table 2 - National Means.xlsx')) as writer:
    for name, data in sum_dic.items():
        data.to_excel(writer, sheet_name=name)

# Generate LaTeX and trim for both
sum_dic['Continuous_latex'] = sum_dic['Continuous'].to_latex().split('midrule')[1].split('\\bottomrule')[0].strip()+'\\bottomrule'
sum_dic['Binary_latex'] = sum_dic['Binary'].to_latex().split('midrule')[1].split('\\bottomrule')[0].strip()+'\\bottomrule'


#Export latex to latex files
with open(os.path.join(config['tables_path'],'latex','cts_means.tex'),'w') as file:
    file.write(sum_dic['Continuous_latex'])

with open(os.path.join(config['tables_path'],'latex','binary_means.tex'),'w') as file:
    file.write(sum_dic['Binary_latex'])


##

