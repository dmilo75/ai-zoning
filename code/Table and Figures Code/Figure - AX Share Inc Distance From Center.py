import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import os
import yaml
from binsreg import binsreg
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load configuration
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load ACS data
acs_data = pd.read_csv(os.path.join(config['raw_data'], '2022_acs_blockgroup_data', "acs_blockgroup_data_all_states.csv"))

#Replace any values of -666666666666 with np.nan
acs_data = acs_data.replace(-666666666, np.nan)

#Load shape data
shape_data = pd.read_csv(os.path.join(config['processed_data'], 'Block Group Analysis Results', 'block_group_analysis_results.csv'))


##
# Mapping of STATEFP to census region
state_to_region = {
    '09': 'Northeast', '23': 'Northeast', '25': 'Northeast', '33': 'Northeast', '44': 'Northeast', '50': 'Northeast',
    '34': 'Northeast', '36': 'Northeast', '42': 'Northeast', '18': 'Midwest', '17': 'Midwest', '26': 'Midwest',
    '39': 'Midwest', '55': 'Midwest', '19': 'Midwest', '20': 'Midwest', '27': 'Midwest', '29': 'Midwest', '31': 'Midwest',
    '38': 'Midwest', '46': 'Midwest', '10': 'South', '11': 'South', '12': 'South', '13': 'South', '24': 'South',
    '37': 'South', '45': 'South', '51': 'South', '54': 'South', '01': 'South', '21': 'South', '28': 'South',
    '47': 'South', '05': 'South', '22': 'South', '40': 'South', '48': 'South', '04': 'West', '08': 'West', '16': 'West',
    '30': 'West', '32': 'West', '35': 'West', '49': 'West', '56': 'West', '02': 'West', '06': 'West', '15': 'West',
    '41': 'West', '53': 'West'
}

# Add census region column to shape_data
shape_data['Census_Region'] = shape_data['STATEFP'].astype(str).str.zfill(2).map(state_to_region)


##
from binsreg import binsreg

# Filter data for South and West regions
south_west_data = shape_data[shape_data['Census_Region'].isin(['South', 'West'])]

# Filter data for Distance_to_Metro <= 100
south_west_data = south_west_data[south_west_data['Distance_to_Metro'] <= 100]

# Perform binscatter for 'Distance_to_Metro' and 'Percent_Incorporated'
est = binsreg(y='Percent_Incorporated', x='Distance_to_Metro', data=south_west_data, binsmethod='dpi', noplot=True)

##

# Plot the results
plt.figure(figsize=(1.4*5.8, 1.4*3.3))
result = est.data_plot[0]
dots = pd.DataFrame(result.dots)
plt.scatter(dots['x'], dots['fit'], color='blue', s=20, alpha=0.5)
plt.xlabel('Distance to Metro')
plt.ylabel('Percent Incorporated')
plt.title('Bin Scatter: Distance to Metro vs Percent Incorporated (South and West Regions)')
plt.tight_layout()
plt.show()

#Save figure to figures folder
plt.savefig(os.path.join(config['figures_path'], 'Figure - AX Share Inc Distance From Center.png'))

##
