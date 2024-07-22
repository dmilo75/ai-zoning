import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model = 'Testing Sample'

#Draw in sample data
results = pd.read_excel(os.path.join(config['processed_data'],'Model Output',model,'Light Data Merged.xlsx'))



