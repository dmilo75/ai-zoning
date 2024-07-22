import pickle
import random
random.seed(2)
import json
import os

from scipy.spatial.distance import cosine
import gpt_functions as cg
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# %% Pull in data on question embeddings

with open(os.path.join(config['embeddings'], "Questions.pkl"), "rb") as fIn:
    da = pickle.load(fIn)

# Extract the embeddings array and the 'Questions' DataFrame
embeddings = da['embeddings']
questions_df = da['Questions']

# Create a map from question text ('Question Detail' column) to its embedding
question_embeddings = {questions_df.iloc[i]['Question Detail']: embeddings[i] for i in range(len(questions_df))}

#Max tokens
max_tokens = 4000

completion_tokens = 0
prompt_tokens = 0


