from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import pandas as pd
import os
import yaml

# Load filepaths
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

text_path = config['muni_text']
embed_path = config['embeddings']

# Load the model
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# Setup text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2100, #Each chunk of 2100 characters of text
    chunk_overlap=200, #Consecutive chunks of text overlap with 200 characters of text
    length_function=len,
    is_separator_regex=False,
)

# Fetch all text files and their respective sub-folder
file_subfolder_mapping = {}
for root, _, files in os.walk(text_path):
    for file in files:
        if file.endswith('.txt'):
            relative_subfolder = os.path.relpath(root, text_path)
            file_subfolder_mapping[os.path.join(root, file)] = relative_subfolder

all_files = list(file_subfolder_mapping.keys())

# Parallelization setup
task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0)) #Id for first job array id
task_max = int(os.environ.get('SLURM_ARRAY_TASK_MAX', 0)) #Id for last job array id
num_nodes = task_max - task_min + 1 #Total number of job array ids
task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
num_files = len(all_files)
files_per_task = num_files // num_nodes
extra_files = num_files % num_nodes

if task_id < extra_files:
    start_idx = task_id * (files_per_task + 1)
    end_idx = start_idx + files_per_task + 1
else:
    start_idx = task_id * files_per_task + extra_files
    end_idx = start_idx + files_per_task

files_for_task = all_files[start_idx:end_idx]

# If task_id is 0, then embed questions
if task_id == 0:
    questions = pd.read_excel(os.path.join(config['raw_data'],'Questions.xlsx'))
    embeddings = model.encode(questions['ID'].to_list())
    with open(embed_path+"/Questions.pkl", "wb") as fOut:
        pickle.dump({'Questions': questions, 'embeddings': embeddings}, fOut)

# Processing the selected files for each node
for file_path in files_for_task:
    
    subfolder = file_subfolder_mapping[file_path]
    muni_name = os.path.basename(file_path).replace('.txt', '')
    embeddings_file_path = embed_path+f"//{subfolder}/{muni_name}.pkl"
    
    # Check if the pickle file already exists
    if os.path.exists(embeddings_file_path):
        continue  # Skip this loop iteration

    with open(file_path, 'r') as file:
        content = file.read()

    chunks = text_splitter.split_text(content)
    embeddings = model.encode(chunks)

    os.makedirs(embed_path+"//"+subfolder, exist_ok=True)

    with open(embeddings_file_path, "wb") as fOut:
        pickle.dump({'chunks': chunks, 'embeddings': embeddings}, fOut)


 