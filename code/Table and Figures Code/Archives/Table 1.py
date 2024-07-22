import pandas as pd
import yaml
import os

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

#%%

#Draw in data on sample
sample = pd.read_excel(os.path.join(data_path,"Sample_Enriched.xlsx"),index_col = 0)

#Draw in population data for states
pop22 = pd.read_csv(os.path.join(raw_path,"2022 Population Data","NST-EST2022-ALLDATA.csv"))

#Draw in county population
msapop22 = pd.read_csv(os.path.join(raw_path, "2022 Population Data","msa population.csv"), encoding='ISO-8859-1')

#%%Make table 1

#Get population by source
pop_source = sample.groupby('Source')['POPULATION'].sum()

#Total population in sample
pop_source.loc['Total'] = pop_source.sum()

#Now make percent of population
pop_source = 100*pop_source/pop22[pop22['NAME'] == 'United States']['POPESTIMATE2022'].iloc[0]

# Convert to DataFrame
pop_source_df = pd.DataFrame(pop_source)
pop_source_df.columns = ["National"]

#Now get by census region
# Define the mapping from states (in lowercase) to census regions
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
    'ca': 'West', 'or': 'West', 'wa': 'West', 'ak': 'West', 'hi': 'West'
}

sample['Region'] = sample['State'].map(state_to_region)

result = sample.pivot_table(index='Source', columns='Region', values='POPULATION', aggfunc='sum').fillna(0)
result.loc['Total'] = result.sum()

#Need to normalize by population 
result['Northeast'] = 100*result['Northeast']/pop22[pop22['NAME'] == 'Northeast Region']['POPESTIMATE2022'].iloc[0]
result['Midwest'] = 100*result['Midwest']/pop22[pop22['NAME'] == 'Midwest Region']['POPESTIMATE2022'].iloc[0]
result['South'] = 100*result['South']/pop22[pop22['NAME'] == 'South Region']['POPESTIMATE2022'].iloc[0]
result['West'] = 100*result['West']/pop22[pop22['NAME'] == 'West Region']['POPESTIMATE2022'].iloc[0]

result = pd.concat([pop_source_df,result], axis = 1)

#%%Get share of MSA coverage

#First, get total MSA population in 2022
msapop = msapop22[msapop22['LSAD'] == 'Metropolitan Statistical Area']['POPESTIMATE2022'].sum()

#Now let's get MSA pop by source in our sample
msa_sample = sample[sample['MSA'] == 1].groupby('Source')['POPULATION'].sum()

#Get a total
msa_sample.loc['Total'] = msa_sample.sum()

#Normalize by population
msa_sample = 100*msa_sample/msapop

#Add in now
result = pd.concat([result,msa_sample], axis = 1)

#%%Now get % of population that lives in muni or township

#First, draw in the data

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

        # If the dataset is not yet in the dataframes dictionary, add it
        if dataset not in dataframes:
            dataframes[dataset] = pd.DataFrame(columns=data_series.index)

        # Append the series to the DataFrame, setting the state as a new column
        dataframes[dataset].loc[state] = data_series

#%%Make the actual table

# Set index name to None (essentially removing the index name)
result.index.name = 'Sample'

#Set column headers
result.columns = ['National','Midwest','Northeast','South','West','MSA']

#Generate latex
latex = result.to_latex(float_format=lambda x: '{:.1f}'.format(x), index=True, index_names=True)

#Isolate the cell entries, formal formatting found in Overleaf
latex = latex.split('midrule')[1].split('\\bottomrule')[0]

#panel B could be % of town/muni population, % of townships, % of munis




