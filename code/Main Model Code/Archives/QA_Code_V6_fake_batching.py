###Main Code
#%%Imports
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import pickle
import yaml
import sys
import helper_functionsV6 as hf
import gpt_functions as gpt
from question_muni_pair import QuestionMuniPair
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#%%Parameters
#Set the model type being used
os.environ['MODEL_TYPE'] = 'gpt-4-turbo-2024-04-09'

#If windows
if os.name == 'nt':
    export_folder_name = 'fake_batch_30_first'
    task_id = 1
else:
    # Arguments processing
    if len(sys.argv) < 2:
        print("Error: No export folder name provided.")
        sys.exit(1)
    export_folder_name = sys.argv[1]

    # Optional: Handle SLURM task ID if your script needs it
    if len(sys.argv) > 2:
        task_id = sys.argv[2]
    else:
        task_id = 1  # Default value if not running under SLURM or no task ID provided

print("Export Folder Name:", export_folder_name)
print("Task ID:", task_id)

#Set the sample size (if running on subset of munis)
os.environ['SAMPLE_SIZE'] = 'all'

#%%Get full list of munis and questions

#The list of munis to process
munis_for_task = hf.get_munis('all')

'''
muni_name = 'west vincent#202485'

#munis_for_task is a list of dictionaries with a paramer 'Muni' which is a string
#We want to find the muni with 'Muni' where muni_name is in 'Muni'.
muni = [muni for muni in munis_for_task if muni_name in muni['Muni']][0]
'''
munis_for_task = munis_for_task[:30]

#The list of questions to process
questions = hf.questions

'''
question = questions.iloc[8]

muni_name = 'arlington'

#munis_for_task is a list of dictionaries with a paramer 'Muni' which is a string
#We want to find the muni with 'Muni' where muni_name is in 'Muni'.
muni = [muni for muni in munis_for_task if muni_name in muni['Muni']][1]
'''

#Drop row indices 11, 12 and 15 from questions
questions = questions.drop([11,12,15])

questions = questions.loc[[13]]

#questions = questions.iloc[[0]]

#Establish results folder if it doesn't exist
folder_path = os.path.join(config['processed_data'], export_folder_name)
os.makedirs(folder_path, exist_ok=True)

from openai import OpenAI

# Openai
client = OpenAI(api_key=config['openai_key'])

def generate_real_response(batch):
    responses = []

    for request in batch:
        # Extract the necessary parameters from the request
        model = request["body"]["model"]
        messages = request["body"]["messages"]
        tools = request["body"].get("tools")

        # Make the API call to create a chat completion
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=request['body']['temperature'],
            seed=request['body']['seed'],
            n=request['body'].get('n', 1),
        )

        # Prepare a list to store all choices from the completion
        all_choices = []
        for choice in completion.choices:
            choice_obj = {
                "index": choice.index,
                "message": choice.message,
                "logprobs": choice.logprobs,
                "finish_reason": choice.finish_reason
            }
            all_choices.append(choice_obj)

        # Create the response object including all choices
        response_obj = {
            "id": "batch_req_" + completion.id,
            "custom_id": request["custom_id"],
            "error": None,
            "status_code": 200,
            "response": {
                "body": {
                    "id": completion.id,
                    "object": "chat.completion",
                    "created": completion.created,
                    "model": model,
                    "system_fingerprint": completion.system_fingerprint,
                    "choices": all_choices,  # Include all choices here
                    "usage": {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens
                    }
                }
            }
        }
        responses.append(response_obj)

    return responses

def load_pair_queue():
    pair_queue_file = os.path.join(folder_path, f'pair_queue{slurm_name()}.pkl')
    if os.path.exists(pair_queue_file):
        with open(pair_queue_file, 'rb') as file:
            return pickle.load(file)
    else:
        return [QuestionMuniPair(row, muni, os.environ['MODEL_TYPE'])
                      for muni in munis_for_task
                      for index, row in questions.iterrows()]

def load_results():
    results_file = os.path.join(folder_path, f"output{slurm_name()}.pkl")

    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            return pickle.load(f)
    else:
        return []


def save_results(results):

    # Save the updated results after each batch
    results_file = os.path.join(folder_path, f"output{slurm_name()}.pkl")

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)




def save_pair_queue(pair_queue):

    status_check_file = os.path.join(folder_path, f'status_check_needed{slurm_name()}.txt')
    pair_queue_file = os.path.join(folder_path, f'pair_queue{slurm_name()}.pkl')

    if len(pair_queue) > 0:
        print("Still remaining pairs")
        with open(pair_queue_file, 'wb') as file:
            pickle.dump(pair_queue, file)
        with open(status_check_file, 'w') as file:
            file.write("True")
    else:
        print("No remaining pairs")
        # If the pair_queue is empty, delete the pair_queue pickle file
        if os.path.exists(pair_queue_file):
            os.remove(pair_queue_file)
        with open(status_check_file, 'w') as file:
            file.write("False")

def flatten_batches(batch):
    #Flatten the list of lists of batches needed
    return [item for sublist in batch for item in sublist]

def  build_model(pair_queue, results):

    batch = []
    batch_tokens = 0

    for pair in reversed(pair_queue): #Iterate in reverse to not corrupt indices when removing
        print("Before")
        print(pair.state)
        print(pair.action)
        pair.process()
        print("After")
        print(pair.state)
        print(pair.action)
        print()


        if pair.state == 'completed':
                results.append(pair.get_final_response())
                pair_queue.remove(pair)
        else:
            if batch_tokens + hf.get_token_count(pair.openai_object) <= hf.MAX_BATCH_TOKENS:
                batch.append(pair.openai_object)
                batch_tokens += hf.get_token_count(pair.openai_object)
            else:
                print('Hit Token Limit')
                break

    if len(batch) > 0:

        flat_batches = flatten_batches(batch)

        if True:#os.name == 'nt':
            batch_results = generate_real_response(flat_batches)

            #Save batch results to file
            batch_results_file = os.path.join(folder_path, f"batch_results{slurm_name()}.pkl")
            with open(batch_results_file, 'wb') as f:
                pickle.dump(batch_results, f)

            batch_id = 1

        else:
            batch_data = {
                "input_file_id": gpt.upload_jsonl(flat_batches),
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h"
            }
            batch_id = gpt.create_batch(batch_data)

    else:
        batch_id = None

    return batch_id, results, pair_queue

def update_pairs(pair_queue, batch_results):
    if len(batch_results) == 0:
        return pair_queue

    # Update the pair_queue with the results from the batch
    for pair in pair_queue:
        for openai_obj in pair.openai_object:  # Now iterating over a list of openai_objects
            result = next((r for r in batch_results if r['custom_id'] == openai_obj['custom_id']), None)
            if result:
                pair.response.append(result)

    return pair_queue

def slurm_name():
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        return f"_{int(os.environ['SLURM_ARRAY_TASK_ID'])}"
    else:
        return "_1"

def load_batch_id():

    batch_id_file = os.path.join(folder_path, f"batch_id{slurm_name()}.txt")

    if os.path.exists(batch_id_file):
        with open(batch_id_file, 'r') as file:
            return file.read().strip()
    else:
        return None

def save_batch_id(batch_id):

    batch_id_file = os.path.join(folder_path, f"batch_id{slurm_name()}.txt")

    with open(batch_id_file, 'w') as file:
        file.write(str(batch_id))

#%%Main code here
# Only run the below code if executing from this file, not if being called from another file
if __name__ == "__main__":

    #Get the batch_id
    batch_id = load_batch_id()

    # Get batch results
    if True:#os.name == 'nt':
        batch_results_file = os.path.join(folder_path, f"batch_results{slurm_name()}.pkl")
        if os.path.exists(batch_results_file):
            with open(batch_results_file, 'rb') as f:
                batch_results = pickle.load(f)
        else:
            batch_results = []
    else:
        batch_results = gpt.retrieve_batch_results(batch_id)

    #Proceed if its the first run (no batch id yet) or batch has returned results
    if batch_id is None or len(batch_results) != 0:

        print("Running batch")

        #Get pair queue
        pair_queue = load_pair_queue()

        #Update the pair_queue with the batch results
        pair_queue = update_pairs(pair_queue,batch_results)

        # Load in the results
        results = load_results()

        #Run the model
        batch_id, new_results, new_pair_queue = build_model(pair_queue, results)

        # Save results
        save_results(new_results)

        # Save pair queue
        save_pair_queue(new_pair_queue)

        #Save the batch id
        if batch_id is None:
            #Clean up unneeded files
            print("Finished Program")
        else:
            print("Running another batch")
            save_batch_id(batch_id)

    else:
        print("Batch still running")
        print()








