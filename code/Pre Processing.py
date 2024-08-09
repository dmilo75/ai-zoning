#%%Data Prep Code


import os

# Define the directory where the Python files are located
code_directory = "./Data Processing Code"

# Loop through each file in the directory
for filename in ['Download Shape Files.py','Make % Urban.py','Merge Raw Data.py']:
    file_path = os.path.join(code_directory, filename)
    
    with open(file_path, 'r') as file:
        exec(file.read())
    
    print(f"Executed {filename}")





