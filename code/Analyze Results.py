'''
Runs all code 
'''

import os

#%%Post Processing
    
#Run clean_llama_output.py
exec(open('./Main Model Code/clean_llama_output.py').read())


#%%Tables and Figures

# Define the directory where the Python files are located
code_directory = "./Table and Figures Code"

# Loop through each file in the directory
for filename in os.listdir(code_directory):
    if filename.endswith(".py"):
        file_path = os.path.join(code_directory, filename)
        
        with open(file_path, 'r') as file:
            exec(file.read())
        
        print(f"Executed {filename}")