import pandas as pd
import os
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define the model to use
model = 'latest_combined'

# Read the questions DataFrame
questions = pd.read_excel(os.path.join(config['raw_data'], 'Questions.xlsx'))

def adjust_index(index):
    try:
        if index != '17w':
            index = int(index)
        return questions.loc[questions['ID'] == index, 'Pioneer Question'].values[0]
    except:
        if index == '27Mean':
            return 'Mean of Min Lot Sizes (Square Feet)'
        elif index == '27Min':
            return 'Minimum of Min Lot Sizes (Square Feet)'
        elif index == '28Mean':
            return 'Mean of Residential Min Lot Sizes'
        elif index == '28Min':
            return 'Minimum of Residential Min Lot Sizes'
        elif index == '28Max':
            return 'Maximum of Residential Min Lot Sizes'
        if index == 'Variance Explained':
            return 'Variance Explained'
        return index

# Import loadings
loadings = pd.read_excel(os.path.join(config['processed_data'], 'Model Output', model, 'Loadings.xlsx'), index_col=0)

# Transpose
loadings = loadings.T

# Adjust index
loadings.index = [adjust_index(index) for index in loadings.index]

# Sort by PC1
loadings = loadings.sort_values('PC1', ascending=False)


# Rename columns
loadings.columns = ['First', 'Second','Third','Fourth','Fifth']

# Set the max_colwidth option to None
pd.set_option('display.max_colwidth', None)

# Translate to LaTeX
latex = loadings.to_latex(index=True, float_format = '%.2f', column_format = 'p{13cm}ccccc', caption = 'Principal Component Analysis Loadings', label = 'tab:loadings')

#Export latex file
with open(os.path.join(config['tables_path'],'latex', 'Appendix - PCA Loadings.tex'), 'w') as f:
    f.write(latex)