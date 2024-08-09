#%%Imports
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import pickle
import yaml
import helper_functionsV5 as hf
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#%%Parameters
#Set the model type being used
os.environ['MODEL_TYPE'] = 'gpt-4-0125-preview'#'gpt-4-turbo-2024-04-09'

export_folder_name = 'Replicability Test'

#Set the sample size (if running on subset of munis)
os.environ['SAMPLE_SIZE'] = 'all'

#%%Get full list of munis and questions

#The list of munis to process
munis_for_task = hf.get_munis('all')

#Just take the first 30 munis
munis_for_task = munis_for_task[:30]

#The list of questions to process
questions = hf.questions

'''
question = questions.iloc[2]

muni_name = 'hanson'

#munis_for_task is a list of dictionaries with a paramer 'Muni' which is a string
#We want to find the muni with 'Muni' where muni_name is in 'Muni'.
muni = [muni for muni in munis_for_task if muni_name in muni['Muni']][0]

'''

#Drop row indices 11, 12 and 15 from questions
questions = questions.drop([11,12,15])

def build_model(muni):
    #%%
    
    #Bring in muni embeddings
    embeddings_file_path = os.path.join(config['embeddings'],muni['Muni']+'.pkl')

    #Check if the file exists, otherwise return an empty list
    try:
        with open(embeddings_file_path, "rb") as fIn:
            tree = pickle.load(fIn)
    except:
        return []


    #Set up answers list
    answers = []

    #The default system prompting
    for i, question in questions.iterrows():
        #%%
        
        #Main loop code
        print()
        print('***')
        print()
        print('Question: ',question['Question Detail'])

        #Run Chat GPT
        response = hf.chat_gpt(tree,question,muni['Muni'],os.environ['MODEL_TYPE'])

        #print(response['Dont_Know'])
        #print(response['Answer'])
        #print(response['Explanation'])

        #Append the answer
        answers.append(response)
        
        print()
        
    #%%
       
    return answers


#%%Main code here
# Only run the below code if executing from this file, not if being called from another file
if __name__ != "__main__":

    # Determine folder path
    folder_path = os.path.join(config['processed_data'], export_folder_name)

    # Check if the folder exists; if not, create it
    try:
        os.makedirs(folder_path)
    except:
        pass

    # Determine the full path to the file
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        filename = os.path.join(folder_path, f"output_{int(os.environ['SLURM_ARRAY_TASK_ID'])}.pkl")
    else:
        filename = os.path.join(folder_path, "output.pkl")

    # Initialize or load answers list
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            all_answers = pickle.load(f)
    else:
        all_answers = []

    # Loop over all munis
    for muni in munis_for_task:
        # Check if muni is already in all_answers
        if not any(answer['Muni'] == muni['Muni'] for answer in all_answers):
            print(muni['Muni'])

            all_answers.extend(build_model(muni))

        # Export data after each muni runs successfully
        with open(filename, 'wb') as f:
            pickle.dump(all_answers, f)



