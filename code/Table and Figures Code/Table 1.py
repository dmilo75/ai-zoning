import pandas as pd
import yaml
import os

# Load filepaths
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

#%%

#Draw in data on sample
sample = pd.read_excel(os.path.join(data_path,"Sample_Enriched.xlsx"),index_col = 0)

#Draw in population data for states
pop22 = pd.read_csv(os.path.join(raw_path,r"2022 Population Data\NST-EST2022-ALLDATA.csv"))

#Draw in county population
msapop22 = pd.read_csv(os.path.join(raw_path, r"2022 Population Data\msa population.csv"), encoding='ISO-8859-1')

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


#%%Make the actual table

# Set index name to None (essentially removing the index name)
result.index.name = 'Sample'

#Set column headers
result.columns = ['National','Midwest','Northeast','South','West','MSA']

#Generate latex
latex = result.to_latex(float_format=lambda x: '{:.1f}'.format(x), index=True, index_names=True)

#Isolate the cell entries, formal formatting found in Overleaf
latex = latex.split('midrule')[1].split('\\bottomrule')[0]

#Export to Excel
result.to_excel(os.path.join(config['tables_path'],"Table 1 - Population Coverage By Source.xlsx"))
