import os
import requests
from zipfile import ZipFile
from bs4 import BeautifulSoup
import yaml

#Load config file
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
shape_dir = config['shape_files']

#Make sub-folders if do not exist yet
# List of folder names to check or create
folders_to_check = ['Counties', 'County Subdivisions', 'Places', 'Urban', 'Water']

# Check each folder and create it if it doesn't exist
for folder in folders_to_check:
    folder_path = os.path.join(shape_dir, folder)
    
    # Create directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder {folder_path} already exists.")
        
        
        
states_info = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", 
    "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13", 
    "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", 
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", 
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", 
    "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", 
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", 
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", 
    "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", 
    "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"
}

#%%Get census places

# Base URL and directory
base_url = "https://www2.census.gov/geo/tiger/TIGER2022/PLACE/tl_2022_{fips}_place.zip"
directory = shape_dir+"Places"

# Iterate over each state and its FIPS code
for state, fips in states_info.items():
    # Create the full download URL
    url = base_url.format(fips=fips)

    # Fetch the zip file
    response = requests.get(url)
    zip_filename = os.path.join(directory, f"tl_2022_{fips}_place.zip")

    # Write the response content to the file
    with open(zip_filename, 'wb') as file:
        file.write(response.content)

    # Extract the zip file
    folder_name = f"tl_2022_{fips}_place"
    folder_path = os.path.join(directory, folder_name)
    os.makedirs(folder_path, exist_ok=True)  # Create directory if it doesn't exist

    with ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(folder_path)

    print(f"Downloaded and extracted data for {state} to {folder_path}")

#%% Get county subdivisions

# Base URL and directory
base_url = "https://www2.census.gov/geo/tiger/TIGER2022/COUSUB/tl_2022_{fips}_cousub.zip"

directory = shape_dir+"County Subdivisions"

# Iterate over each state and its FIPS code
for state, fips in states_info.items():
    # Create the full download URL
    url = base_url.format(fips=fips)

    # Fetch the zip file
    response = requests.get(url)
    zip_filename = os.path.join(directory, f"tl_2022_{fips}_cousub.zip")

    # Write the response content to the file
    with open(zip_filename, 'wb') as file:
        file.write(response.content)

    # Extract the zip file
    folder_name = f"tl_2022_{fips}_cousub"
    folder_path = os.path.join(directory, folder_name)
    os.makedirs(folder_path, exist_ok=True)  # Create directory if it doesn't exist

    with ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(folder_path)

    print(f"Downloaded and extracted data for {state} to {folder_path}")
    
    
    
#%%Now we want to download all county level water area maps

# Constants
BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2022/AREAWATER/"
DIRECTORY = shape_dir+"Water"

# Fetch the HTML content
response = requests.get(BASE_URL)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract all zip file links
zip_links = [link['href'] for link in soup.find_all('a') if 'href' in link.attrs and link['href'].endswith('_areawater.zip')]
        
        
for zip_link in zip_links:
    state_fips = zip_link.split('_')[2][:2]
    county_fips = zip_link.split('_')[2][2:5]

    state_folder = os.path.join(DIRECTORY, state_fips)
    county_folder = os.path.join(state_folder, county_fips)
    
    # Make directories if they don't exist
    os.makedirs(county_folder, exist_ok=True)

    # Fetch the zip file
    zip_url = BASE_URL + zip_link
    response = requests.get(zip_url)

    # Save zip file
    zip_path = os.path.join(county_folder, zip_link)
    with open(zip_path, 'wb') as file:
        file.write(response.content)

    # Extract the zip file
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(county_folder)

    print(f"Downloaded and extracted {zip_link} to {county_folder}")



#%%Download Urban and Counties shape files

# Function to download, save, and extract zip files
def download_shape_files(url, target_folder):
    # Create directory if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Fetch the zip file
    response = requests.get(url)
    
    # Extract the filename from the URL
    zip_filename = os.path.basename(url)
    
    # Save zip file
    zip_path = os.path.join(target_folder, zip_filename)
    with open(zip_path, 'wb') as file:
        file.write(response.content)
    
    # Extract the zip file
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)

    print(f"Downloaded and extracted {zip_filename} to {target_folder}")
    

# URLs and target folders
download_links = [
    ("https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip", shape_dir + 'Counties'),
    ("https://www2.census.gov/geo/tiger/TIGER2020/UAC/tl_2020_us_uac20.zip", shape_dir + 'Urban')
]

# Download each shape file to the corresponding folder
for url, target_folder in download_links:
    download_shape_files(url, target_folder)


