# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:44:31 2023

@author: Dan's Laptop

Source: https://www.census.gov/data/datasets/2022/econ/gus/public-use-files.html

Data Dictionary: file:///C:/Users/Dan's%20Laptop/AppData/Local/Temp/Tempa1a42b96-58be-4d48-9bb2-f38c8d59729c_govt_units_2022.ZIP/Government_Units_List_Documentation_2022.pdf


"""
#%%
#root = r"/Users/gaocixin/Dropbox/Inclusionary Zoning"
import pandas as pd
#from bs4 import BeautifulSoup, NavigableString
import requests
#import pickle
#from matplotlib_venn import venn3
#import matplotlib.pyplot as plt
import re
#import us
import math
from thefuzz import fuzz
#import sys
#import os
#os.chdir( root+r"/Embedding_Project/Compare Results/Chart Formatter")
#sys.path.append('/Users/gaocixin/Dropbox/Inclusionary Zoning/Embedding_Project/Compare Results/Chart Formatter')
#import ChartFormatter as cf
 
#%%Setup list to hold all data entries

coverage = []

#%%Get all muni names from municode

def remove_county_info(input_string):
    # Remove the trailing (xxx co.) pattern
    modified_string = re.sub(r'\s\((.*) co\.\)', '', input_string)
    
    # Remove the trailing , xxx co pattern
    modified_string = re.sub(r',\s(.*) co$', '', modified_string)
    
    return modified_string

#Clean zip code
def process_zip(zip_code):
    zip_code = str(zip_code)
    if "-" in zip_code:
        zip_code = zip_code.split('-')[0]
    try:
        return int(zip_code)
    except:
        return None

#Get all states/munis
for state in requests.get("https://api.municode.com/States/").json():
    state_abrev = state['StateAbbreviation'].lower()
    print(state_abrev)
    url = "https://api.municode.com/Clients/stateAbbr?stateAbbr="+state_abrev
    response = requests.get(url)
    state_data = response.json()
    for town in state_data:
        specs = {}
        specs['Muni'] = remove_county_info(town['ClientName'].lower())
        specs['State'] = state_abrev
        specs['Source'] = 'Municode'
        specs['Zip Code'] = town['ZipCode']
        specs['Website'] = town['Website']
        specs['Address'] = town['Address']+" "+town['Address2']
        specs['Link'] = f"https://library.municode.com/{state_abrev}/{town['ClientName'].lower().replace(' ','_')}/"
        coverage.append(specs)
        
#%%Get all muni names from American Legal Publishing

data = requests.get("https://codelibrary.amlegal.com/api/client-regions/").json()

for entry in data:
    state = entry['slug']
    print(state)
    new_url = f"https://codelibrary.amlegal.com/api/client-regions/{state}/"
    towns = requests.get(new_url).json()['clients']
    for town in towns:
        
        #Basic info first
        specs = {}
        specs['Muni'] = town['name'].lower()
        specs['State'] = state
        specs['Source'] = 'American Legal Publishing'
        specs['Link'] = f"https://codelibrary.amlegal.com/codes/{town['slug']}/latest/overview"
        
        #Now we do an extra API call for extra info
        extra_info = requests.get(f"https://codelibrary.amlegal.com/api/clients/{town['slug']}/").json()
        specs['Zip Code'] = process_zip(extra_info['address_zip_code'])
        specs['Website'] = extra_info['website_url']
        specs['Address'] = extra_info['address_line_1']+" "+extra_info['address_line_2']
        
        #Append the data
        coverage.append(specs)

#%% Now pull in all munis from ordinance.com for Massachussetts

'''
Metadata for Ordinance.com is collected in the ord_dot_com scraper
'''

#Read in from Excel
df = pd.read_excel(r"/Users/gaocixin/Dropbox/Inclusionary Zoning/Embedding_Project/Web_Scraping/Data/ord_dot_com_metadata.xlsx", index_col = 0, engine = "openpyxl")

#Save a copy so that we can merge back the info
ord_dot_com_metadata = df.copy()

# Renaming 'Muni Name' to 'Muni'
df.rename(columns={'Muni Name': 'Muni'}, inplace=True)

#make link
df['Link'] = df.apply(lambda row: f"https://ordinance.com/working/view_new.asp?ordID={row['UID']}&ordUID={row['UID']}", axis=1)

# Dropping columns 'UID' and 'State ID'
df.drop(['UID', 'State ID'], axis=1, inplace=True)

#Add source
df['Source'] = 'Ordinance.com'

#turn to list format
ord_dot_com_list = df.to_dict(orient='records')

# Converting the dataframe to a list of dictionaries
coverage.extend(ord_dot_com_list)


#%% Now, we'll try to match stuff in the list of entries to the census data

#Bring in list of all munis

df = pd.read_excel(r"/Users/gaocixin/Dropbox/Inclusionary Zoning/Embedding_Project/Web_Scraping/All_US_Gov.xlsx", engine = "openpyxl")

munis = df[df['UNIT_TYPE'].isin(['2 - MUNICIPAL','3 - TOWNSHIP'])]

# List of phrases to replace
phrases_to_replace = [
    'CITY OF ', 'TOWN OF ', 'TOWNSHIP OF ', 'PLANTATION OF ', 
    'VILLAGE OF ', 'MUNICIPALITY OF ', 'METROPOLITAN GOVERNMENT OF ', 
    'CITY AND BOROUGH OF ', 'BOROUGH OF ', 'CITY AND COUNTY OF ', 
    'COUNTY OF ', 'CONSOLIDATED GOVERNMENT OF ', 'UNIFIED GOVERNMENT OF ',
    'METRO GOVERNMENT OF ', 'URBAN COUNTY GOVERNMENT OF ',
    'CITY PARISH OF ', 'CITY-PARISH OF ', 'CORPORATION OF '
]

# Loop through the list and replace each phrase with an empty string
for phrase in phrases_to_replace:
    munis['UNIT_NAME'] = munis['UNIT_NAME'].str.replace(phrase, '')

# Filter and print rows that still contain the word 'OF'
remaining_of = munis[munis['UNIT_NAME'].str.contains(' OF ')]
print(remaining_of['UNIT_NAME'])

#Make muni names and state abbrevs lowercase
munis['UNIT_NAME'] = munis['UNIT_NAME'].str.lower()
munis['STATE'] = munis['STATE'].str.lower()

#%%Functions for the matching code

def clean_string(string,old,new):
    clean_string = re.sub(old, new,string)
    if clean_string.strip() == "":
        return string
    return clean_string.strip()

def match(df,series,filters):
    
    #First, do exact filtering
    for filt in filters:
        source_var = filt
        dest_var = name_map[filt]
        if filters[filt] == 'Exact':
            df = df[df[dest_var] == series[source_var]]
        else:
            # Fuzzy matching
            matched_values = []
            for val in df[dest_var].dropna().unique():
                candidate_val = str(val).lower()
                source_val = str(series[source_var]).lower()

                # Clean the strings before fuzzy matching
                for old, new in replacements[source_var].items():
                    candidate_val = clean_string(candidate_val, old, new)
                    source_val = clean_string(source_val, old, new)

                ratio = fuzz.ratio(candidate_val.strip(), source_val.strip())
                if ratio > 90:
                    matched_values.append(val)
                    print(f"Fuzzy Matched: {val} with {series[source_var]}")

            # Keep rows that match any of the fuzzy matched values
            df = df[df[dest_var].isin(matched_values)]
            
    return df
            

def town_or_muni(muni_name,state):
    '''
    Based on muni name and state whether entity is township or municipality
    '''
    muni_name = muni_name.lower()

    
    #Check if muni 
    muni_keywords = ['city','village','borough','boro']
    for keyword in muni_keywords:
        if keyword in muni_name:
            return 'Muni'
        
    #Else, check if town
    town_keywords = ['township','twp']
    for keyword in town_keywords:
        if keyword in muni_name:
            return 'Township'
        
    #The word 'town' in new england, minnesota, new york and wisconsin is a township too, but a muni elsewhere
    if 'town' in muni_name:
        if state in ['mn','ny','wi','ct','ri','ma','nh','vt','me']:
            return 'Township'
        else:
            return 'Muni'
    
    #Else, we are unsure of the type
    return 'Unsure'

def filter_type(df,muni_name):
    '''
    Filter remaining results based on whether we have a town or muni
    '''
    
    #If we only have one type of government then dont use this to filer
    if len(df['UNIT_TYPE'].unique()) == 1:
        return df
    
    which = town_or_muni(muni_name,df['STATE'].iloc[0])
    
    if which == 'Unsure':
        return df
    
    if which == 'Muni':
        unit_type = '2 - MUNICIPAL'
    else:
        unit_type = '3 - TOWNSHIP'
    
    return df[df['UNIT_TYPE'] == unit_type]

#%%Parameters

#States where munis and townships can geographically overlap
overlap_states = ['ct','il','ks','mi','mn','mo','ne','ny','oh','vt']

#Here is some manual data cleaning before fuzzy matching. 
replacements = {}

replacements['Website'] = {
    "http://": "",
    "https:/": "",
    "www.":"",
    r'(\.com/|\.org/).*':r'\1', #Remove anything subpage information of the site, info trailing .com
}

replacements['Address'] = {
    "East": "E",
    "Street": "ST",
    "West": "W",
    "North": "N",
    "South": "S",
    "Lane":"LN",
    "Road":"RD",
    r' #[0-9]+':'', #This gets rid of a # followed by numbers
    r'PO Box [0-9]+ ':'', #Get rid of PO box info
    'Third':'3RD',
    'First':'1ST',
    'Second':'2ND',
    'Fourth':'4TH',
}

replacements['County'] = {}

replacements['Muni'] = {  
    "chrtr township":"",
    "town of":"",
    "charter township": "",
    "township": "",
    "twp.":"",
    "city of":"",
    "city":"",
    "(village of)":"",
    "village of ":"",
    "village":"",
    "borough":"",
    "boro":"",
    "town":"",
    "saint":"st",
    "charter":"",
    " pk ":" park ",
    "mount":"mt",
    "-":" ",
    "metro":"",
    ",":"",
    "\(":"",
    "\)":""
}

name_map = {
    'Muni':'UNIT_NAME',
    'State':'STATE',
    'County':'COUNTY_AREA_NAME',
    'Zip Code':'ZIP',
    }


#%%Actual merging code

# Assuming you've already created 'coverage_df' and 'munis' DataFrames
coverage_df = pd.DataFrame(coverage)

#First filter out any counties or parishes (in the case of Louisiana)
coverage_df = coverage_df[~coverage_df['Muni'].str.contains('county')]
coverage_df = coverage_df[~coverage_df['Muni'].str.contains('parish')]

# Create an empty list to store each DataFrame row
final_rows = []

# Loop through each row in 'coverage_df'
for index, row in coverage_df.iterrows():

    # Initialize an empty dictionary to store the final row
    final_row = {}

    # Populate initial keys from the row in 'coverage_df'
    for col in coverage_df.columns:
        final_row[col] = row[col]

    # Step 1: Try to do an exact match based on 'Muni' and 'State'
    exact_matches = match(munis,row,{'Muni':'Exact','State':'Exact'})


    if len(exact_matches) == 1:
        # Update final_row dictionary with values from the exact match
        for col in munis.columns:
            final_row[col] = exact_matches.iloc[0][col]
        final_row['Match'] = 'Matched (Exact Match on State and Muni Name)'

    elif row['Source'] == 'Ordinance.com':
        
        #If multiple towns with same name in the state then we tie break on county
        if len(exact_matches) > 1:
            #First, try to exact match on county
            res = match(exact_matches,row,{'County':'Exact'})
            
            #Then, try to fuzzy match on county
            if len(res) != 1:
                res = match(exact_matches,row,{'County':'Fuzzy'})
                
            if len(res) == 1:
                match_reason = 'Matched on State, Muni, and County'
        
        #If we didn't match at all then we need to try fuzzy matching
        else:
            fuzzy_name = match(munis,row,{'State':'Exact','Muni':'Fuzzy'})
            
            #If multiple matches then we need to use county info 
            if len(fuzzy_name) > 1:
                
                #First, exact county matches
                res = match(fuzzy_name,row,{'County':'Exact'})
                
                if len(res) != 1:
                    res = match(fuzzy_name,row,{'County':'Fuzzy'})
                    
                    #We now check to see if we can determine whether we have a muni or township to tie 
                    if len(res) > 1:
                        res = filter_type(res,row['Muni'])
                    
            else:
                res = fuzzy_name
        
            if len(res) == 1:
                match_reason = 'Fuzzy match on name and State/County info'
        if len(res) == 0:
            match_reason = 'No matches found'
        if len(res) > 1:
            match_reason = 'Multiple matches found'
        
        #Store the processed data
        if len(res) != 1:
            for col in munis.columns:
                final_row[col] = None
        else:
            for col in munis.columns:
                final_row[col] = res.iloc[0][col]
        final_row['Match'] = match_reason
    

    #Now try using zip code info
    else:
    
        if len(exact_matches) > 1:
            # Step 2: Exact match for ZIP
            exact_zip = match(exact_matches,row,{'Zip Code':'Exact'})
        
            #The zip code may have changed or we may not have had a valid zipcode from the web scraped source. So let's just use state then
            if len(exact_zip.index) == 0:
                exact_zip = exact_matches 
                
        #Else if there were no matches        
        else:
            
            #Match on zip and state
            exact_zip = match(munis,row,{'Zip Code':'Exact','State':'Exact'})
            
            #Now fuzzy match on muni name
            max_ratio = 0
            best_match = None
            for val1 in exact_zip['UNIT_NAME'].dropna().unique():
                
                #Clean the strings before fuzzy matching
                candidate_muni = str(val1).lower()
                source_muni = str(row['Muni']).lower()
                
                for old, new in replacements['Muni'].items():
                    candidate_muni = clean_string(candidate_muni,old,new)
                    source_muni = clean_string(source_muni,old,new)
                ratio = fuzz.ratio(candidate_muni.strip(), source_muni.strip())
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_match = val1
    
            if max_ratio > 90:
                exact_zip = exact_zip[exact_zip['UNIT_NAME'] == best_match]
                print(f"Fuzzy Matched: {best_match} with {row['Muni']}")
                
            #Otherwise try allowing for different zip codes and then fuzzy matching
            else:
                exact_zip = munis[(munis['STATE'] == row['State'])] 
                
                #Now fuzzy match on muni name
                max_ratio = 0
                best_match = None
                for val1 in exact_zip['UNIT_NAME'].dropna().unique():
                    
                    #Clean the strings before fuzzy matching
                    candidate_muni = str(val1).lower()
                    source_muni = str(row['Muni']).lower()
                    
                    for old, new in replacements['Muni'].items():
                        candidate_muni = clean_string(candidate_muni,old,new)
                        source_muni = clean_string(source_muni,old,new)
                    ratio = fuzz.ratio(candidate_muni.strip(), source_muni.strip())
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_match = val1
        
                if max_ratio > 90:
                    exact_zip = exact_zip[exact_zip['UNIT_NAME'] == best_match]
                    print(f"Fuzzy Matched: {best_match} with {row['Muni']}")
                
                #Otherwise we've found no matches on muni name
                else:
                    #Delete all rows then
                    exact_zip = exact_zip.drop(exact_zip.index)
                    
        #If we've found a unique match        
        if len(exact_zip) == 1:
            for col in munis.columns:
                final_row[col] = exact_zip.iloc[0][col]
            final_row['Match'] = 'Matched (Zip Code Tiebreaker and/or Fuzzy Name Match)'
                
                
        #Otherwise use website and address info
        else:

            found_match = False
            
            for col, muni_col in zip(['Website', 'Address'], ['WEB_ADDRESS', 'ADDRESS1']):
                max_ratio = 0
                best_match = None
                
                for val1 in exact_zip[muni_col].dropna().unique():
                    clean_val1 = str(val1)
                    clean_row_col = str(row[col])
                    
                    for old, new in replacements[col].items():
                        clean_val1 = clean_string(clean_val1,old,new)
                        clean_row_col = clean_string(clean_row_col,old,new)
                    
                    #Now ensure lowercase
                    clean_val1 = clean_val1.lower().strip().rstrip('/')
                    clean_row_col = clean_row_col.lower().strip().rstrip('/')
        
                    ratio = fuzz.ratio(clean_val1, clean_row_col)
    
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_match = val1
                        
                if max_ratio > 90:  # or any other threshold you consider appropriate
                    matched_row = exact_zip[exact_zip[muni_col] == best_match].iloc[0]
                    print(f"Fuzzy Matched: {best_match} with {row[col]}")
                    
                    for colm in munis.columns:
                        final_row[colm] = matched_row[colm]
                    final_row['Match'] = f'Matched (Fuzzy match on {col})'
                    found_match = True
                    break
    
            if not found_match:
                
                # If the state is in overlap_states, prefer the municipality over the township
                if row['State'] in overlap_states:
                    # Check if there are exactly two unique rows in exact_zip
                    if len(exact_zip.index) == 2 and len(exact_zip['UNIT_TYPE'].unique()) == 2:
                        # Filter exact_zip to find the row with 'UNIT_TYPE' as '2 - MUNICIPAL'
                        matched_row = exact_zip[exact_zip['UNIT_TYPE'] == '2 - MUNICIPAL'].iloc[0]
                        for col in munis.columns:
                            final_row[col] = matched_row[col]
                        final_row['Match'] = 'Matched (Took Muni over Township in Overlap State)'
                        found_match = True
                if not found_match:
                    for col in munis.columns:
                        final_row[col] = None
                    if len(exact_zip) > 0:
                        final_row['Match'] = 'Multiple Matches'
                    else:
                        final_row['Match'] = 'No Matches'

    final_rows.append(final_row)

# Create the final DataFrame from the list of dictionaries
final_df = pd.DataFrame(final_rows)

# Find duplicate rows based on 'Muni', 'State', 'Source'
duplicates = final_df[final_df['CENSUS_ID_PID6'].notnull() & final_df.duplicated(subset=['CENSUS_ID_PID6', 'Source'], keep=False)]

# If you want to display the duplicates
print("Duplicate Rows based on 'Muni', 'State', 'Source':")
print(duplicates)

non_null_count = final_df['CENSUS_ID_PID6'].count()
total_rows = final_df['CENSUS_ID_PID6'].shape[0]
percentage_non_null = (non_null_count / total_rows) * 100

print(f"Percentage of non-null entries in column CENSUS_ID_PID6: {percentage_non_null:.2f}%")

#%%

test = munis[munis['STATE'] == 'nj']

test = test[test['UNIT_TYPE'] == '3 - TOWNSHIP']

#%%Now let's manually analyze the unmatched rows and duplicate rows

#Munis that matched
matched = final_df[~final_df['CENSUS_ID_PID6'].isna() & ~final_df[['CENSUS_ID_PID6', 'Source']].apply(tuple, axis=1).isin(duplicates[['CENSUS_ID_PID6', 'Source']].apply(tuple, axis=1))]




#Munis that didn't match
unmatched = final_df[~final_df.index.isin(matched.index)]

#Update the manual merge section
manual_merge = pd.read_excel(r"/Users/gaocixin/Dropbox/Inclusionary Zoning/Embedding_Project/Web_Scraping/Merging With Census/Manual Merges.xlsx",index_col = 0, engine = "openpyxl")

selected_columns = ['Muni','State','County','Source','Zip Code','Website','Address','Match','Link']

select_unmatched = unmatched[selected_columns]

# First, set the index to the columns that uniquely identify each row
manual_merge.set_index(['Muni', 'State', 'Source'], inplace=True)
select_unmatched.set_index(['Muni', 'State', 'Source'], inplace=True)

# Drop the rows in 'select_unmatched' that are already in 'manual_merge'
select_unmatched = select_unmatched[~select_unmatched.index.isin(manual_merge.index)]

# Reset index to make the merge easier
manual_merge.reset_index(inplace=True)
select_unmatched.reset_index(inplace=True)

# Add the additional columns in 'select_unmatched' with null values
for col in ['ID', 'Manual Match Reason', 'Reason-Detail']: # replace with your actual column names
    select_unmatched[col] = None

# Concatenate the two DataFrames
merged_df = pd.concat([manual_merge, select_unmatched])

merged_df.to_excel(r"/Users/gaocixin/Dropbox/Inclusionary Zoning/Embedding_Project/Web_Scraping/Merging With Census/Manual Merges.xlsx", engine = "openpyxl")




#%%Once manual merge has been updated draw in corrections

#Import the manual merge excel
manual_merge = pd.read_excel(r"/Users/gaocixin/Dropbox/Inclusionary Zoning/Embedding_Project/Web_Scraping/Merging With Census/Manual Merges.xlsx", engine = "openpyxl")

if len(manual_merge['Manual Match Reason'].dropna()) != len(manual_merge.index):
    raise ValueError("Need to finish manually matching before merge is finished")

#Ensure we have all PID6 codes 
def map_id(num):
    
    if pd.isna(num):
        return num
    
    if num < 500000:
        return num
    
    row = munis[munis['CENSUS_ID_GIDID'] == num].iloc[0]
    
    return row['CENSUS_ID_PID6']


manual_merge['ID'] = manual_merge['ID'].apply(lambda x: map_id(x))

#Incorporate manual merge information
for index, row in manual_merge.iterrows():
    #If there is now a valid ID
    if not math.isnan(row['ID']) and not ((matched.apply(lambda x: x['CENSUS_ID_PID6'] == row['ID'] and x['Source'] == row['Source'], axis=1)).any()):
        
        full_row = unmatched[(unmatched['Muni'] == row['Muni']) & (unmatched['State'] == row['State']) & (unmatched['Source'] == row['Source']) & (unmatched['County'] == row['County'])]  

        census_row = munis[munis['CENSUS_ID_PID6'] == int(row['ID'])].iloc[0]
        for col in munis.columns:
            full_row[col] = census_row[col]
        full_row['Match'] = 'Manual'
        matched = pd.concat([matched,full_row])
        
#%%Now double check once more that we have no duplicates

duplicates_check = matched[matched['CENSUS_ID_PID6'].notnull() & matched.duplicated(subset=['CENSUS_ID_PID6', 'Source'], keep=False)]

if len(duplicates_check.index) != 0:
    raise ValueError("There are still duplicate ids")

        
#%%Merge ordinance.com metadata into the final matched dataframe

# Filter the 'matched' DataFrame for rows where 'Source' is 'Ordinance.com'
matched_filtered = matched[matched['Source'] == 'Ordinance.com']

# Perform an outer merge with the filtered 'matched' DataFrame
merged_df = pd.merge(
    matched_filtered,
    ord_dot_com_metadata.drop(columns = 'State ID'),
    how='left',
    left_on=['County', 'Muni', 'State'],
    right_on=['County', 'Muni Name', 'State']
)

# Append the non-merged rows from 'matched' (where 'Source' is not 'Ordinance.com')
non_merged_rows = matched[matched['Source'] != 'Ordinance.com']
merged_df = merged_df.append(non_merged_rows, ignore_index=True)

merged_df = merged_df.drop(columns = 'Muni Name')



#%%Export here

#Let's export matched so that we can use it to loop through munis in web scraping
merged_df.to_excel(r"/Users/gaocixin/Dropbox/Inclusionary Zoning/Embedding_Project/Web_Scraping/Data/matched_munis.xlsx", engine = "openpyxl")



