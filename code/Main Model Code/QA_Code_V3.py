#%%Imports
import os
import random
random.seed(42)
import sys
import pandas as pd
import pickle
import yaml
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)


#%%Parameters

#System inputs

'''
First argument is model version, second argument is sample size if running in serial and just testing
'''

model_version = sys.argv[1]
try:
    sample_size = sys.argv[2]
except:
    sample_size = "all"

use_chatgpt = model_version in ['3.5','4']

model_dic = {'3.5':'gpt-3.5-turbo',
             '4':'gpt-4',
             '13b':'llama13b',
             '70b':'llama70b'}

os.environ['USE_CHATGPT'] = str(use_chatgpt)
os.environ['MODEL_TYPE'] = model_dic[model_version]

import helper_functionsV2 as hf

#Whether the model is running in paralell on multiple nodes
paralell = 'SLURM_ARRAY_TASK_ID' in os.environ

if use_chatgpt == 'True':
    
    gpt_model = model_dic[model_version]
    if paralell:
        raise ValueError("You are using a  job array and using Chat GPT")


#%%Get full list of munis

sample_data = pd.read_excel(config['raw_data']+'/Sample Data.xlsx')


# Create list of dictionaries
muni_list = []
for index, row in sample_data.iterrows():
    entry = {'State': row['State'],
             'Muni': f"{row['Muni']}#{row['FIPS_PLACE']}#{row['State']}"}
    muni_list.append(entry)

#Get questions
questions = hf.questions

#%%Paralellization parameters


#If running on multiple nodes
if paralell:
    #The number of nodes we are parallelizing off of
    task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0)) #Id for first job array id
    task_max = int(os.environ.get('SLURM_ARRAY_TASK_MAX', 0)) #Id for last job array id
    num_nodes = task_max - task_min + 1 #Total number of job array ids
    
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    num_munis = len(muni_list)
    
    # Calculate municipalities per node
    munis_per_task = num_munis // num_nodes
    extra_munis = num_munis % num_nodes
    
    # If task_id is less than extra_munis, it takes one more municipality
    if task_id < extra_munis:
        start_idx = task_id * (munis_per_task + 1)
        end_idx = start_idx + munis_per_task + 1
    else:
        start_idx = task_id * munis_per_task + extra_munis
        end_idx = start_idx + munis_per_task
    print("done")
    munis_for_task = muni_list[start_idx:end_idx]
    
    
#Otherwise just use the training sample  
else:
    print("muni len", len(muni_list))
    #Load in munis in training dataset
    training_file = config['raw_data']+"/training.pickle"
    with open(training_file,"rb") as fIn:
        training = pickle.load(fIn)
    training = training.to_list()
    print("Pre filter")
    print(len(training))
    training = [x+"#ma" for x in training] #Add #ma because we added this to the file format recently
    total_training = [entry for entry in muni_list if entry['Muni'] in training]
    print("Length of training")
    print(len(total_training))
    if sample_size == 'all':
        munis_for_task = total_training
    else:
        #Take a smaller random sample for testing purposes
        munis_for_task = random.sample(total_training,int(sample_size))
    
    if sample_size != 'all':
        questions = questions[questions['ID'] == 27]

#%%Tweaking Parameters

def build_model(muni):
    #%%   
    
    #Bring in muni embeddings
    embeddings_file_path = config['embeddings']+f"/{muni['State']}/{muni['Muni']}.pkl"

    with open(embeddings_file_path, "rb") as fIn:
        data = pickle.load(fIn)
    
    contexts = data['chunks']
    context_embeddings = data['embeddings']
    
    #Set up answers list
    answers = []
    
    #The default system prompting
      
    for i, question in questions.iterrows():
        #%% Main loop code
        print()
        print('***')
        print()
        print('Question: ',question['Question Detail'])
           
        #Get the context for the question
        context = hf.create_context(question,contexts,context_embeddings)

        #Determine whether to use Llama or Chat GPT
        if use_chatgpt:
            response = hf.chat_gpt(context,question,muni['Muni'],gpt_model)
           
        #Run Llama and get the response
        else:
           response = hf.llama_response(context,question,muni['Muni'])

        #%%  
        #Append the answer
        answers.append(response)
       
    return answers


#%%Main code here

#Only run the below code if executing from this file, not if being called from another file
if __name__ == "__main__":
    
    # Determine folder path
    folder_path = os.path.join(config['processed_data'], model_dic[model_version].replace('.', ''))
    
    # Check if the folder exists; if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Determine the full path to the file
    if paralell:
        filename = os.path.join(folder_path, f"llama{task_id}.pickle")
    else:
        filename = os.path.join(folder_path, "output.pickle")
    
    #Initialize answers list
    all_answers = []
    
    #Loop over all munis
    for muni in munis_for_task:
        
        #Process each muni
        print(muni['Muni'])
        all_answers.extend( build_model(muni))

        #Export data after each muni runs succesfully 
        with open(filename,'wb') as f:
            pickle.dump(all_answers,f)
  

