#Imports
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import yaml
import sys
import helper_functions as hf
from dotenv import load_dotenv
load_dotenv()
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Whether to use synchronous (fake batching) or asynchronous (real batching) processing
fake_batch = config['fake_batch'] == 'True'

#If uses a slurm job array
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    # Arguments processing
    if len(sys.argv) < 2:
        print("Error: No export folder name provided.")
        sys.exit(1)

    export_folder_name = sys.argv[1]

    # Optional: Handle SLURM task ID if provided
    task_id = None
    if len(sys.argv) > 2:
        task_id = sys.argv[2]

    print("Export Folder Name:", export_folder_name)
    if task_id:
        print("Task ID:", task_id)

    if task_id:
        os.environ['TASK_ID'] = str(task_id)

#Otherwise
else:
    os.environ['TASK_ID'] = '1'
    export_folder_name = config['export_folder_name']

#Establish results folder if it doesn't exist
folder_path = os.path.join(config['processed_data'], 'Model Output',export_folder_name)
os.makedirs(folder_path, exist_ok=True)

#If check status is false then we've already finished processing this node/folder
if hf.check_status(folder_path):

    #Get the batch_id
    batch_id = hf.load_batch_id(folder_path)

    # Get batch results
    batch_results = hf.retrieve_batch(batch_id, folder_path, fake_batch = fake_batch)

    #Proceed if its the first run (no batch id yet) or batch has returned results
    if batch_id is None or len(batch_results) != 0:

        print("Running batch")

        #Get pair queue
        pair_queue = hf.load_pair_queue(folder_path)

        #Update the pair_queue with the batch results
        pair_queue = hf.update_pairs(pair_queue,batch_results)

        # Load in the results
        results = hf.load_results(folder_path)

        #Load in the errors
        errors = hf.load_errors(folder_path)

        #Run the model
        batch_id, new_results, new_errors, new_pair_queue = hf.build_model(pair_queue, results,errors, folder_path, fake_batch=fake_batch)

        # Save results
        hf.save_results(new_results,folder_path)

        # Save errors
        hf.save_errors(new_errors,folder_path)

        # Save pair queue
        hf.save_pair_queue(new_pair_queue,folder_path)

        #Save the batch id
        if batch_id is None:
            #Clean up unneeded files
            print("Finished Program")

        else:
            print("Running another batch")
            hf.save_batch_id(batch_id,folder_path)

    else:
        print("Batch still running")
        print()
else:
    print("Already finished processing this specification. Create a new export_folder_name in config.yaml or delete the existing folder to run a new specification.")

