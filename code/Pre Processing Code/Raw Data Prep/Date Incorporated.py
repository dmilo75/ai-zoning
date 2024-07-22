import pandas as pd
import os
import yaml
import helper_functions as hf

# Load filepaths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_path = config['processed_data']
raw_path = config['raw_data']

##First, read in goodman Github data in on year of incorporation munis

# URL of the CSV file
url = "https://raw.githubusercontent.com/cbgoodman/muni-incorporation/master/rawdata/muni-incorporation.csv"

# Read the CSV file into a DataFrame
goodman_df = pd.read_csv(url)

#Rename census_id_pid6 to CENSUS_ID_PID6
goodman_df = goodman_df.rename(columns = {'census_id_pid6':'CENSUS_ID_PID6'})

#Keep only CENSUS_ID_PID6 and year_of_incorporation
goodman_df = goodman_df[['CENSUS_ID_PID6','yr_incorp']]
goodman_df = goodman_df.dropna()

#Load in sample_data
sample = pd.read_excel(os.path.join(raw_path, "Sample Data.xlsx"), index_col=0)

#Only keep munis in sample
goodman_df = goodman_df[goodman_df['CENSUS_ID_PID6'].isin(sample['CENSUS_ID_PID6'])]

#Export
goodman_df.to_excel(os.path.join(data_path,'interim_data','Year_Incorporated.xlsx'), index = False)

##Scraping in Massachussetts townships date of incorporation
'''
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_table_to_dataframe(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table_data = []

    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) == 3:
            town_data = {
                'Town': cells[0].get_text(strip=True),
                'Founded': cells[1].get_text(strip=True),
                'Incorporated': cells[2].get_text(strip=True)
            }
            table_data.append(town_data)

    df = pd.DataFrame(table_data)
    return df

url = 'https://massachusettsalmanac.com/incorporation-dates-of-mass-towns/'
df = scrape_table_to_dataframe(url)

#Next, we fuzzy merge into census of governments
'''