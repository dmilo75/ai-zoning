# -*- coding: utf-8 -*-

import pandas as pd
import math
import os
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def nan_sum(lst):
    total = 0
    for num in lst:
        if not math.isnan(num):
            total += num
    return total


wharton = pd.read_stata(os.path.join(config['raw_data'], 'WRLURI_01_15_2020.dta'))
wharton = wharton[wharton["metro18"] == 1]
wharton = wharton[["WRLURI18", "cbsacode18", "cbsatitle18"]]
wharton = wharton.rename(columns = {"cbsacode18":"CBSA Code", "cbsatitle18":"CBSA Title"})
wharton["CBSA Code"] = [str(i)[:-2] for i in wharton["CBSA Code"]]

wharton_grouped = wharton.groupby(['CBSA Code','CBSA Title']).agg(['mean', 'count'])
wharton_grouped = wharton_grouped.sort_values(('WRLURI18', 'mean'), ascending=False)
wharton_mean = nan_sum(wharton_grouped[('WRLURI18', 'mean')])/len(wharton_grouped[('WRLURI18', 'mean')])


ourindex = pd.read_excel(os.path.join(config['processed_data'],'Model Output','Full Run','Overall_Index.xlsx'))
geoid = [i[-9:-3] for i in ourindex["Muni"]]
ourindex["geoid"] = geoid

census = pd.read_excel(os.path.join(config['raw_data'], 'census_of_gov.xlsx'))
census = census[["CENSUS_ID_PID6", "FIPS_STATE", "FIPS_COUNTY"]]
census = census.rename(columns = {"CENSUS_ID_PID6":"geoid"})
census["geoid"] = [str(i)for i in census["geoid"]]
our_merged_fips = pd.merge(ourindex, census, on = "geoid")

county_code = []
our_county_code = list(our_merged_fips["FIPS_COUNTY"])
for i in our_county_code:
  code = str(i)
  if len(code) == 2:
    code = "0"+code
  elif len(code) == 1:
    code = "00"+code
  county_code.append(code)
our_merged_fips["FIPS_COUNTY"] = county_code

state_code = []
our_state_code = our_merged_fips["FIPS_STATE"]
for i in our_state_code:
  code = str(i)
  if len(code) == 1:
    code = "0"+code
  state_code.append(code)
our_merged_fips["FIPS_STATE"] = state_code
our_merged_fips = our_merged_fips[["FIPS_STATE","FIPS_COUNTY","First_Principal_Component","Second_Principal_Component"]]

cbsa2017 = pd.read_csv(os.path.join(config['raw_data'], 'cbsa_2017.csv'))
cbsa2017.columns = cbsa2017.iloc[1]
cbsa2017 = cbsa2017[2:].reset_index(drop=True)
cbsa2017 = cbsa2017.rename(columns = {"FIPS State Code":"FIPS_STATE", "FIPS County Code":"FIPS_COUNTY"})
cbsa2017 = cbsa2017[["CBSA Code","FIPS_STATE","FIPS_COUNTY","CBSA Title"]]

our_merged = pd.merge(our_merged_fips, cbsa2017, on = ["FIPS_STATE","FIPS_COUNTY"])
our_merged = our_merged[["CBSA Code","CBSA Title","First_Principal_Component","Second_Principal_Component"]]
our_grouped = our_merged.dropna().groupby(['CBSA Code','CBSA Title']).agg(['mean', 'count'])
our_grouped = our_grouped.sort_values(('First_Principal_Component', 'mean'), ascending=False)
PCA1_mean = nan_sum(our_grouped[('First_Principal_Component', 'mean')])/len(our_grouped[('First_Principal_Component', 'mean')])
PCA2_mean = nan_sum(our_grouped[('Second_Principal_Component', 'mean')])/len(our_grouped[('Second_Principal_Component', 'mean')])

corr = pd.merge(wharton_grouped, our_grouped, on = ["CBSA Code","CBSA Title"])

correlation_pca1 = corr[[('First_Principal_Component', 'mean'), ('WRLURI18', 'mean')]].corr().loc[('First_Principal_Component', 'mean'),('WRLURI18','mean')]
correlation_pca2 = corr[[('Second_Principal_Component', 'mean'), ('WRLURI18', 'mean')]].corr().loc[('Second_Principal_Component', 'mean'),('WRLURI18','mean')]
