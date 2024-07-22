###Main Code
# %%Imports
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import yaml
import sys
import pandas as pd
import json
import helper_functionsV7 as hf
import context_buildingv4 as cb
import gpt_functions as gpt

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

##Cambridge to start

def zoning_analysis_function(map_info, muni, temperature=0, seed=42):
    first_col = map_info.columns[0]
    districts = map_info[first_col].to_list()
    state = 'semantic'
    district_list = ', '.join(districts)

    # Setup question
    question = {
        'Question Detail': f'For each zoning district in {district_list}, determine the allowed housing types',
        'Question Rephrase': f'For each zoning district in {district_list}, determine the allowed housing types',
    }

    # Get context
    context = cb.context_builder(question, state, muni)

    # Prepare the prompt
    prompt = f"""
    ###Instructions:
    As a zoning expert, analyze the following zoning districts in {muni['Muni'].split('#')[0]} {muni['Muni'].split('#')[2]}:

    {', '.join(districts)}

    For each district, determine:
    1) Does it exclusively allow single-family homes (either attached or detached)?
    2) Does it exclusively allow either single-family homes (attached or detached) or duplex housing (2 units)?
    3) Does it allow multi-family housing (three or more units)?

    ##Response Format:
    Use the following format for your response in proper JSON:
    {{
        "DISTRICT_NAME": {{
            "single_family_only": true/false/null,
            "single_family_or_duplex_only": true/false/null,
            "allows_multi_family": true/false/null
        }},
        ...
    }}

    If a district isn't mentioned or you're unsure, use null. Try to categorize each district to the best of your ability.

    ###Context:
    {context}
    """

    # Make the LLM call
    model = 'gpt-4o'
    messages = [
        {"role": "system", "content": "You are a municipal zoning analyst who follows instructions, references the context, and responds in JSON."},
        {"role": "user", "content": prompt}
    ]
    request = {
        "body": {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed
        }
    }
    response = gpt.generate_completion(model, messages, None, request, response_format={"type": "json_object"})

    # Process the LLM response
    try:
        zoning_analysis = json.loads(response.choices[0].message.content)
        print("LLM Response Parsed Successfully:", zoning_analysis)

        # Add new columns to map_info
        map_info['Single_Family_Only'] = map_info[first_col].map(
            lambda x: zoning_analysis.get(x, {}).get('single_family_only'))
        map_info['Single_Family_Or_Duplex_Only'] = map_info[first_col].map(
            lambda x: zoning_analysis.get(x, {}).get('single_family_or_duplex_only'))
        map_info['Allows_Multi_Family'] = map_info[first_col].map(
            lambda x: zoning_analysis.get(x, {}).get('allows_multi_family'))

        # Verify that columns are added
        print("Columns added to map_info:")
        print(map_info.head())

        # Print the results
        print("Map Info with new columns added:")
        print(map_info[[first_col, 'Single_Family_Only', 'Single_Family_Or_Duplex_Only', 'Allows_Multi_Family']])

        return map_info

    except json.JSONDecodeError:
        print("Error: LLM response was not in valid JSON format")
        print("Raw response:", response.choices[0].message.content)
        return None

##

def get_filenames(filepath):

    # Get all filenames in the directory
    filenames = os.listdir(filepath)

    # Filter out any non-file items (like subdirectories)
    filenames = [f for f in filenames if os.path.isfile(os.path.join(filepath, f))]

    return filenames


def process_filenames(cog, filenames, manual_review_file):
    results = []
    unmatched = []

    # Check if manual review file exists and load it if it does
    if os.path.exists(manual_review_file):
        manual_review_df = pd.read_excel(manual_review_file)
        # Check if all manual reviews are completed
        if manual_review_df['manual_match'].isnull().any():
            raise ValueError("Manual review is not complete. Please finish the manual review in the Excel file.")
    else:
        manual_review_df = pd.DataFrame()

    for filename in filenames:
        parts = filename.replace('.xlsx', '').split('_')

        if len(parts) >= 3:
            state = parts[-1]
            county = parts[-2]
            city = '_'.join(parts[:-2])
        else:
            print(f"Warning: Filename {filename} does not have expected format.")
            continue

        # Check if this file is in the manual review
        if not manual_review_df.empty:
            manual_match = manual_review_df[manual_review_df['filename'] == filename]

            if not manual_match.empty:
                match_result = manual_match['manual_match'].iloc[0]
            else:
                match_result = match_to_cog(cog, city, county, state)
        else:
            match_result = match_to_cog(cog, city, county, state)

        result = {
            'filename': filename,
            'city': city,
            'county': county,
            'state': state,
            'match_result': match_result
        }

        results.append(result)

        if match_result == "No match found" or match_result == "Multiple matches found":
            unmatched.append(result)

    # Create manual review file if it doesn't exist and there are unmatched entries
    if not os.path.exists(manual_review_file) and unmatched:
        df = pd.DataFrame(unmatched)
        df['manual_match'] = ''  # Add column for manual matching
        df.to_excel(manual_review_file, index=False)
        print(f"Created manual review file: {manual_review_file}")
        raise ValueError("Manual review file created. Please complete the manual review before running again.")

    # Merge automated and manual results
    final_results = pd.DataFrame(results)

    #Drop any rows with 'Failed Manual' as the match result, this means we couldn't manually match to cog
    final_results = final_results[final_results['match_result'] != 'Failed Manual']

    return final_results


import re


def clean_unit_name(name):
    """
    Clean and standardize the UNIT_NAME field.

    Args:
    name (str): The original UNIT_NAME to clean.

    Returns:
    str: The cleaned UNIT_NAME.
    """
    # Convert to lowercase
    name = name.lower()

    # List of phrases to remove
    phrases_to_remove = [
        'city of', 'town of', 'township of', 'plantation of',
        'village of', 'municipality of', 'metropolitan government of',
        'city and borough of', 'borough of', 'city and county of',
        'county of', 'consolidated government of', 'unified government of',
        'metro government of', 'urban county government of',
        'city parish of', 'city-parish of', 'corporation of'
    ]

    # Remove the phrases
    for phrase in phrases_to_remove:
        name = name.replace(phrase, '')

    # Additional cleaning
    name = re.sub(r'\s+', ' ', name)  # Remove extra spaces
    name = name.strip()  # Remove leading/trailing spaces

    # Additional replacements
    replacements = {
        'saint': 'st',
        'mount': 'mt',
        '-': ' ',
        ',': '',
        '(': '',
        ')': ''
    }
    for old, new in replacements.items():
        name = name.replace(old, new)

    return name

# The match_to_cog function will be defined next
def match_to_cog(cog, city, county, state):
    # Clean the city name
    clean_city = clean_unit_name(city)

    # Subset cog for the state
    cog_state = cog[cog['STATE'] == state.upper()]

    # Subset cog_state for the county
    cog_county = cog_state[cog_state['COUNTY_AREA_NAME'] == county.upper()]

    # Clean the UNIT_NAME in the cog_county DataFrame
    cog_county.loc[:, 'CLEAN_UNIT_NAME'] = cog_county['UNIT_NAME'].apply(clean_unit_name)


    # Try to find an exact match
    exact_match = cog_county[cog_county['CLEAN_UNIT_NAME'] == clean_city]

    if len(exact_match) == 1:

        #get muni id name
        muni_id = exact_match['CENSUS_ID_PID6'].iloc[0]

        return muni_id
    elif len(exact_match) > 1:
        # Handle multiple matches (you might want to implement additional logic here)
        return "Multiple matches found"
    else:
        # No exact match found, you might want to implement fuzzy matching here
        return "No match found"

def get_muni_dict(census_id, sample):
    row = sample[sample['CENSUS_ID_PID6'] == census_id].iloc[0]
    return {
        'Muni': row['UNIT_NAME'].lower() + '#' + str(row['CENSUS_ID_PID6']) + '#' + row['State'].lower(),
    }


def process_batch(batch_map_info, muni):
    # Initialize variables
    processed_rows = pd.DataFrame()
    remaining_rows = batch_map_info.copy()
    threshold = 5
    previous_remaining_count = float('inf')

    while not remaining_rows.empty:
        # Process the remaining rows in the current batch
        newly_processed = zoning_analysis_function(remaining_rows, muni)

        if newly_processed is not None:
            # Separate processed and unprocessed rows
            processed = newly_processed[~newly_processed['Single_Family_Only'].isnull() &
                                        ~newly_processed['Single_Family_Or_Duplex_Only'].isnull() &
                                        ~newly_processed['Allows_Multi_Family'].isnull()]
            unprocessed = newly_processed[newly_processed['Single_Family_Only'].isnull() |
                                          newly_processed['Single_Family_Or_Duplex_Only'].isnull() |
                                          newly_processed['Allows_Multi_Family'].isnull()]

            # Add newly processed rows to the processed_rows dataframe
            processed_rows = pd.concat([processed_rows, processed], ignore_index=True)

            # Update remaining_rows
            remaining_rows = unprocessed

            # Check if we made progress
            if len(remaining_rows) >= previous_remaining_count:
                print("No progress made in the last iteration. Exiting loop.")
                break

            previous_remaining_count = len(remaining_rows)

        else:
            print("Error in processing. Exiting loop.")
            break

    return processed_rows, remaining_rows

def process_zoning_map(muni, filename):
    # Construct file paths
    map_info_filepath = os.path.join(config['raw_data'], 'Zoning Map Data', 'with_county_state', filename)
    output_folder = os.path.join(config['processed_data'], 'Zoning Map Analysis')
    os.makedirs(output_folder, exist_ok=True)

    output_filename = f"{muni['Muni']}_zoning_analysis.xlsx"
    output_path = os.path.join(output_folder, output_filename)

    #If file already exists then return
    if os.path.exists(output_path):
        print(f"Zoning map analysis already exists for {filename}. Skipping.")
        return

    # Read in excel
    map_info = pd.read_excel(map_info_filepath)
    #Sort by column perc descending
    map_info = map_info.sort_values(by = 'perc', ascending = False)
    first_col = map_info.columns[0]
    districts = map_info[first_col].to_list()

    # Split the districts into batches of at most 15
    batch_size = 15
    district_batches = [districts[i:i + batch_size] for i in range(0, len(districts), batch_size)]

    # Initialize variables
    final_processed_rows = pd.DataFrame()
    final_remaining_rows = pd.DataFrame()

    for batch in district_batches:
        # Create a sub-dataframe for the current batch
        batch_map_info = map_info[map_info[first_col].isin(batch)].copy()

        # Process the current batch
        processed_rows, remaining_rows = process_batch(batch_map_info, muni)

        # Combine results
        final_processed_rows = pd.concat([final_processed_rows, processed_rows], ignore_index=True)
        final_remaining_rows = pd.concat([final_remaining_rows, remaining_rows], ignore_index=True)

    # Combine processed rows with any remaining unprocessed rows
    final_map_info = pd.concat([final_processed_rows, final_remaining_rows], ignore_index=True)

    # Check if the required columns exist in final_map_info
    required_columns = ['Single_Family_Only', 'Single_Family_Or_Duplex_Only', 'Allows_Multi_Family']
    for column in required_columns:
        if column not in final_map_info.columns:
            print(f"Warning: {column} not found in final_map_info columns")

    # Print the results
    print("\nFinal results:")
    print(final_map_info.columns)
    print(final_map_info[[first_col] + required_columns])

    # Print statistics
    print(f"\nTotal districts: {len(final_map_info)}")
    print(f"Processed districts: {len(final_processed_rows)}")
    print(f"Unprocessed districts: {len(final_remaining_rows)}")
    if 'perc' in final_remaining_rows.columns:
        print(f"Percentage of area in unprocessed districts: {final_remaining_rows['perc'].sum():.2f}%")

    # Save results
    final_map_info.to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


##Code to bridge filename with COG

# Read in the census of governments data
cog  = pd.read_excel(os.path.join(config['raw_data'],'Census of Governments','census_of_gov_22.xlsx'))

#Filter out counties
cog = cog[cog['UNIT_TYPE'] != '1 - COUNTY']

# Get the filenames
filepath = os.path.join(config['raw_data'], 'Zoning Map Data', 'with_county_state')
filenames = get_filenames(filepath)

#Match the files
results = process_filenames(cog, filenames,os.path.join(config['raw_data'],'Zoning Map Data','manual_review.xlsx'))

#Save to excel
results.to_excel(os.path.join(config['raw_data'],'Zoning Map Data','cog_bridge.xlsx'),index = False)

#Make dictionary that maps from 'match_result' to filename
match_dict = dict(zip(results['match_result'],results['filename']))

#Now bring in sample data
sample = pd.read_excel(os.path.join(config['raw_data'],'Sample Data.xlsx'))

#Find rows in sample with CENSUS_ID_PID6 in results match_result column
sample_maps = sample[sample['CENSUS_ID_PID6'].isin(results['match_result'])]

#Loop over rows
for index, row in sample_maps.iterrows():
    #Get the filename
    filename = match_dict[row['CENSUS_ID_PID6']]

    #Get the muni dictionary
    muni = get_muni_dict(row['CENSUS_ID_PID6'], sample)

    #Process the zoning map

    process_zoning_map(muni, filename)
    print(f"Sucessfully processed zoning map for {filename}. Moving to next file.")


