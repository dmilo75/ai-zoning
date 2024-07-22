# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:11:28 2023

@author: Dan's Laptop
"""
root = r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning"
import os
import pandas as pd
import us
import plotly.express as px
import plotly.io as pio
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import matplotlib.pyplot as plt

#Load config file
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set the default DPI for all figures
plt.rcParams['figure.dpi'] = 100

#Fips to state abbrev mapping
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

#%%Load in data

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



#%%Analyze COG local gov coverage

df = dataframes['cog_sample'].copy()

df['FIPS'] = df.index

df['State'] = df['FIPS'].astype(int).map(fips_to_state)
'''
Want ISO-2 
'''

fig = px.choropleth(df.fillna(0), 
                    locations='State', 
                    color='% Pop', 
                    locationmode="USA-states", 
                    scope="usa",
                    color_continuous_scale=["white", "red"],
                    labels={'% Pop':'Percent of Population'},
                    title="Percent of Population Living in a Muni and/or Township By State")

pio.write_image(fig, r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Embedding_Project\Web_Scraping\Sample Coverage\muni_coverage.png", format='png', scale=3)

#bar chart

df = df.sort_values(by = '% Pop')

# Create a bar chart
plt.figure(figsize=(12, 6))  # Adjust the size of the figure as needed
plt.bar(df['State'], df['% Pop'], color='blue')  # You can change the color

# Adding titles and labels
plt.title('Percentage of Population in Town or Muni by State')
plt.xlabel('State')
plt.ylabel('% of Population')
plt.ylim(0,100)

# Show the plot
plt.show()

# Calculate the total sum of 'Num Munis' and 'Num Townships' for sorting
df['Total'] = df['Num Munis'] + df['Num Townships']

# Sort the DataFrame by this total sum
df = df.sort_values(by='Total')

# Create a stacked bar chart
plt.figure(figsize=(12, 6))  # Adjust the size of the figure as needed

# Plot 'Num Munis' and 'Num Townships' as stacked bars
plt.bar(df['State'], df['Num Munis'], color='#008080', label='Num Munis')
plt.bar(df['State'], df['Num Townships'], color='#FF6F61', bottom=df['Num Munis'], label='Num Townships')

# Adding titles and labels
plt.title('Number of Municipalities and Townships per State')
plt.xlabel('State')
plt.ylabel('Count')
plt.legend()  # Add a legend

# Show the plot
plt.show()



#%%Now analyze our own sample coverage


df = 100*dataframes['our_sample']/dataframes['cog_sample']

df['FIPS'] = df.index

df['State'] = df['FIPS'].astype(int).map(fips_to_state)

#Pop Coverage

df = df.sort_values(by = '% Pop')

plt.figure(figsize=(12, 6))  # Adjust the size of the figure as needed
plt.bar(df['State'], df['% Pop'], color='blue')  # You can change the color

# Adding titles and labels
plt.title('Our Sample Percent of Local Government Population Coverage')
plt.xlabel('State')
plt.ylabel('% of Coverage')
plt.ylim(0,100)

# Show the plot
plt.show()

fig = px.choropleth(df.fillna(0), 
                    locations='State', 
                    color='% Pop', 
                    locationmode="USA-states", 
                    scope="usa",
                    color_continuous_scale=["white", "red"],
                    labels={'% Pop':'Percent of Muni/Township Population'},
                    title='Our Sample Percent of Muni/Township Population Covered By State')

pio.write_image(fig, r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Embedding_Project\Web_Scraping\Sample Coverage\our_coverage.png", format='png', scale=3)


#Muni Coverage
df = df.sort_values(by = 'Num Munis')

plt.figure(figsize=(12, 6))  # Adjust the size of the figure as needed
plt.bar(df['State'], df['Num Munis'], color='blue')  # You can change the color

# Adding titles and labels
plt.title('Our Sample Percent of Total Munis Covered by State')
plt.xlabel('State')
plt.ylabel('% of Coverage')
plt.ylim(0,100)

# Show the plot
plt.show()


#Muni Coverage
df = df.sort_values(by = 'Num Townships')

plt.figure(figsize=(12, 6))  # Adjust the size of the figure as needed
plt.bar(df['State'], df['Num Townships'], color='blue')  # You can change the color

# Adding titles and labels
plt.title('Our Sample Percent of Total Townships Covered By State')
plt.xlabel('State')
plt.ylabel('% of Coverage')
plt.ylim(0,100)

# Show the plot
plt.show()

#%%What is the national coverage?


def calculate_population_coverage(dataframes, state_pop):
    def get_state_population(fips):
        if int(fips) in state_pop['STATE'].values:
            return state_pop[state_pop['STATE'] == int(fips)].iloc[0]['POPESTIMATE2021']
        else:
            # Handle the case where FIPS code is not found. You can also choose to raise an error here.
            return 0


    def calculate_total_population(dataframe):
        total_population = 0
        for fips in dataframe.index:
            state_population = get_state_population(fips)
            total_population += dataframe.loc[fips, '% Pop'] * state_population
        return total_population

    total_population_cog = calculate_total_population(dataframes['cog_sample'])
    total_population_our = calculate_total_population(dataframes['our_sample'])

    ratio = total_population_our / total_population_cog if total_population_cog != 0 else 0
    
    ratio2 = total_population_cog / state_pop[state_pop['SUMLEV'] == 10]['POPESTIMATE2021'].sum()

    return total_population_cog/100000000, total_population_our/100000000, 100*ratio, ratio2

#Load in state population
state_pop = pd.read_csv(os.path.join(config['raw_data'],'2022 Population Data','state_population.csv'))


nat_cov = calculate_population_coverage(dataframes,state_pop)
print(nat_cov)

#%%What are the most populous munis that we don't cover?

cog = pd.read_excel(os.path.join(config['raw_data'],'census_of_gov.xlsx'))
cog = cog[cog['UNIT_TYPE'] != '1 - COUNTY']
sample = pd.read_excel(os.path.join(config['raw_data'],'Sample Data.xlsx'))

need_to_cover = cog[~cog['CENSUS_ID_PID6'].isin(sample['CENSUS_ID_PID6'].to_list())]

# Sort the DataFrame by 'POPULATION' in descending order and take the top 25
sorted_df = need_to_cover.sort_values(by='POPULATION', ascending=False).head(25)

# Rename the columns
sorted_df = sorted_df.rename(columns={'UNIT_NAME': 'Local Gov Name', 'STATE': 'State', 'POPULATION': 'Population'})

# Print the desired columns
print(sorted_df[['Local Gov Name', 'State', 'Population']])




