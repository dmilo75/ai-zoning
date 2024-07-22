#Load config file
import os
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

root = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning"
import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the default DPI for all figures
plt.rcParams['figure.dpi'] = 150

# Fips to state abbrev mapping
fips_to_state = {
    2: 'AK', 1: 'AL', 5: 'AR', 60: 'AS', 4: 'AZ',
    6: 'CA', 8: 'CO', 9: 'CT', 11: 'DC', 10: 'DE',
    12: 'FL', 13: 'GA', 66: 'GU', 15: 'HI', 19: 'IA',
    16: 'ID', 17: 'IL', 18: 'IN', 20: 'KS', 21: 'KY',
    22: 'LA', 25: 'MA', 24: 'MD', 23: 'ME', 26: 'MI',
    27: 'MN', 29: 'MO', 28: 'MS', 30: 'MT', 37: 'NC',
    38: 'ND', 31: 'NE', 33: 'NH', 34: 'NJ', 35: 'NM',
    32: 'NV', 36: 'NY', 39: 'OH', 40: 'OK', 41: 'OR',
    42: 'PA', 72: 'PR', 44: 'RI', 45: 'SC', 46: 'SD',
    47: 'TN', 48: 'TX', 49: 'UT', 51: 'VA', 78: 'VI',
    50: 'VT', 53: 'WA', 55: 'WI', 54: 'WV', 56: 'WY'
}

state_to_region = {
    'me': 'Northeast', 'nh': 'Northeast', 'vt': 'Northeast', 'ma': 'Northeast',
    'ri': 'Northeast', 'ct': 'Northeast', 'ny': 'Northeast', 'nj': 'Northeast',
    'pa': 'Northeast',

    'wi': 'Midwest', 'mi': 'Midwest', 'il': 'Midwest', 'in': 'Midwest',
    'oh': 'Midwest', 'nd': 'Midwest', 'sd': 'Midwest', 'ne': 'Midwest',
    'ks': 'Midwest', 'mn': 'Midwest', 'ia': 'Midwest', 'mo': 'Midwest',

    'de': 'South', 'md': 'South', 'va': 'South', 'wv': 'South',
    'ky': 'South', 'nc': 'South', 'sc': 'South', 'tn': 'South',
    'ga': 'South', 'fl': 'South', 'al': 'South', 'ms': 'South',
    'ar': 'South', 'la': 'South', 'tx': 'South', 'ok': 'South',

    'mt': 'West', 'id': 'West', 'wy': 'West', 'co': 'West',
    'nm': 'West', 'az': 'West', 'ut': 'West', 'nv': 'West',
    'ca': 'West', 'or': 'West', 'wa': 'West', 'ak': 'West', 'hi': 'West', 'dc': 'South',
}


# The path to the directory containing the pickle files
directory_path = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\processed data\pop_coverage"

# %%Load in data

# Dictionary to hold the dataframes for each dataset
dataframes = {}

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.pkl'):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        # Read the pickle file into a pandas Series
        data_series = pd.read_pickle(file_path)

        print(data_series)

        # Extract the dataset and state from the filename
        dataset, state = filename.split('#')
        state = state.split('.')[0]  # Remove the '.pkl' extension

        #Correct the local gov pop variable
        data_series['Local Gov Pop'] = data_series['% Pop']*data_series['State Pop']/100

        state_str = fips_to_state[int(state)]
        region = state_to_region[state_str.lower()]

        data_series.loc['State'] = state_str
        data_series.loc['Region'] = region

        # If the dataset is not yet in the dataframes dictionary, add it
        if dataset not in dataframes:
            dataframes[dataset] = pd.DataFrame(columns=data_series.index)

        # Append the series to the DataFrame, setting the state as a new column
        dataframes[dataset].loc[state] = data_series

#For each dataframe we want to groupby region and sum
for dataset in dataframes:
    dataframes[dataset] = dataframes[dataset].groupby('Region').sum()

    #Add a total row
    dataframes[dataset].loc['Total'] = dataframes[dataset].sum()

panelA = pd.DataFrame(columns = ['Total','Northeast','Midwest','South','West'])
panelA.loc['Total Munis'] = dataframes['cog_sample']['Num Munis']
panelA.loc['% of Munis In Sample'] = 100*dataframes['our_sample']['Num Munis']/dataframes['cog_sample']['Num Munis']
panelA.loc['Total Townships'] = dataframes['cog_sample']['Num Townships']
panelA.loc['% of Townships In Sample'] = 100*dataframes['our_sample']['Num Townships']/dataframes['cog_sample']['Num Townships']
panelA.loc['Total Pop. (Millions)'] = dataframes['cog_sample']['State Pop']/1000000
panelA.loc['% of Pop. Under Local Gov.'] = 100*dataframes['cog_sample']['Local Gov Pop']/dataframes['cog_sample']['State Pop']

#Now panel B breaks it out by source the % of population living under a local government
panelB = pd.DataFrame(columns = ['Total','Northeast','Midwest','South','West'])
panelB.loc['American Legal Publishing'] =100*dataframes['alp']['Local Gov Pop']/dataframes['cog_sample']['Local Gov Pop']
panelB.loc['Municode'] = 100*dataframes['mc']['Local Gov Pop']/dataframes['cog_sample']['Local Gov Pop']
panelB.loc['Ordinance.com'] = 100*dataframes['ord']['Local Gov Pop']/dataframes['cog_sample']['Local Gov Pop']
panelB.loc['Total'] = 100*dataframes['our_sample']['Local Gov Pop']/dataframes['cog_sample']['Local Gov Pop']

#Now let's define a function to process the text and export as latex
def get_latex(df):
    #First, round the dataframe
    df = df.round(0)

    #Now fill na with '-'
    df = df.fillna('-')

    # Generate latex and format as integers wtih commas for thousands
    latex = df.to_latex(float_format=lambda x: '{:,.0f}'.format(x), index=True, index_names=True)
    # Isolate the cell entries, formal formatting found in Overleaf
    latex_rows = latex.split('midrule')[1].split('\\bottomrule')[0].strip()+'\\bottomrule'

    return latex_rows

panelAlatex = get_latex(panelA)
panelBlatex = get_latex(panelB)

#Now export to latex

# Export latex to latex files
with open(os.path.join(config['tables_path'], 'latex', 'table1a.tex'), 'w') as file:
    file.write(panelAlatex)

with open(os.path.join(config['tables_path'], 'latex', 'table1b.tex'), 'w') as file:
    file.write(panelBlatex)
