#Connecticut county bridge (Some external datasets use counties and others use regional bodies in place of counties for Connecticut. We will have both ready.)
#Source: https://www1.ctdol.state.ct.us/lmi/misc/counties.asp
#Source2: https://unicede.air-worldwide.com/unicede/unicede_connecticut_fips_2.html
ct_map = {
    'norwich': {'County Name': 'New London County', 'County FIPS': 11},
    'seymour': {'County Name': 'New Haven County', 'County FIPS': 9},
    'madison': {'County Name': 'New Haven County', 'County FIPS': 9},
    'shelton': {'County Name': 'Fairfield County', 'County FIPS': 1},
    'bloomfield': {'County Name': 'Hartford County', 'County FIPS': 3},
    'enfield': {'County Name': 'Hartford County', 'County FIPS': 3},
    'vernon': {'County Name': 'Tolland County', 'County FIPS': 13},
    'southbury': {'County Name': 'New Haven County', 'County FIPS': 9},
    'avon': {'County Name': 'Hartford County', 'County FIPS': 3},
    'new haven': {'County Name': 'New Haven County', 'County FIPS': 9},
    'naugatuck': {'County Name': 'New Haven County', 'County FIPS': 9},
    'new britain': {'County Name': 'Hartford County', 'County FIPS': 3},
    'hartford': {'County Name': 'Hartford County', 'County FIPS': 3}
}

# Define a function to update the FIPS_COUNTY_ADJ based on the ct_map dictionary
def update_fips_adj(row):
    muni = row['Muni'].lower()  # Ensure case-insensitive matching
    if muni in ct_map:
        row['FIPS_COUNTY_ADJ'] = ct_map[muni]['County FIPS']
    return row