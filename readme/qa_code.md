# README for QA_Code.py

## Overview
This code serves as the entry point to run the broader model. It imports model configurations, which are setup in config.yaml, and iterates the list of question-municipality pairs through the broader code. The code is meant to be run several seperate times with the same export_folder_name allowing for time to wait in between for the API responses to generate. 


## Configuration Variables
These variables are defined in config.yaml and are used to determine the specification 
- **`fake_batch`**: Controls whether to use the OpenAI API that is synchronous (True) or asynchronous (False). Asynchronous batching costs half the amount of synchronous batching and allows OpenAI to paralellize the API responses rather than the user. However, this comes at the cost of waiting up to 24 hours for API responses and the requirement to keep track of all state variables between runs. For more information on the asynchronous API, see the OpenAI API documentation [here](https://platform.openai.com/docs/guides/batch).
- **`sample_size`**: If set to 'all', it processes all municipalities. If set to an integer, it processes a random sample of that size.
- **`export_folder_name`**: This is the name of the model specification being run which is also used as the name of the folder where results and intermediate files are saved. The export folder is created under processed data -> Model Output. Note that if running the code with a slurm job array then this variable is defined in command line.

## State Variables
The following are intermediate variables used to track all state variables of the program. These variables are saved in the folder specified by `export_folder_name`. If running with a slurm job array, files are suffixed by the node's ID.
- **Results**: A list of dictionaries with the final results of each question-municipality pair.
- **Errors**: A list of dictionaries with the state variables of question-municipality and the error message when that specific pair hit a fatal error.
- **Pair Queue**: The list of question-municipality pairs that have not finished processing yet.
- **Batch ID**: The key used to lookup the batch results from the OpenAI API.

## Main Code Logic

1. **Check Node/Folder Status**:  
   The script begins by checking the status of the folder for the current node/task using `hf.check_status(folder_path)`. If this function returns `True`, it means that the processing for this node/folder is not yet complete, and the script will proceed to retrieve or start processing batches.

2. **Load Batch ID**:  
   The batch ID is loaded using `hf.load_batch_id(folder_path)`. If a batch ID exists, it indicates that a previous batch has been processed. Otherwise, it returns `None`, meaning this is the first run for this specification.

3. **Retrieve Batch Results**:  
   The script then attempts to retrieve batch results using `hf.retrieve_batch(batch_id, folder_path, fake_batch=fake_batch)`. If a batch ID exists, it retrieves any available results from previous API responses or the local file system, depending on whether synchronous (fake) or asynchronous (real) batching is being used.

4. **Run New Batch (if no batch ID or results found)**:  
   If there is no batch ID (indicating this is the first run) or if batch results have been returned (i.e., batch processing is complete), the script proceeds to:
   - **Load Pair Queue**: The pair queue, which contains the list of question-municipality pairs to be processed, is initialized if this is the first time running the specification. This is done using `hf.load_pair_queue(folder_path, questions, munis_for_task)`. If the pair queue already exists, it is simply loaded from the folder.
   - **Update Pairs**: The pair queue is then updated with any new batch results using `hf.update_pairs(pair_queue, batch_results)`.
   - **Load Results and Errors**: Any existing results and errors from prior processing are loaded using `hf.load_results(folder_path)` and `hf.load_errors(folder_path)`.

5. **Build Model and Process Batch**:  
   The script runs the model using `hf.build_model(pair_queue, results, errors, folder_path, fake_batch=fake_batch)`. This function processes the current pairs of questions and municipalities, and the output includes:
   - **New Batch ID**: A key used for tracking the current batch being processed.
   - **New Results**: The results of the newly processed pairs.
   - **New Errors**: Any errors encountered during processing.
   - **New Pair Queue**: The updated queue of pairs still awaiting processing in future batches.

6. **Save Results, Errors, and Queue**:  
   After processing, the script saves the newly generated results, errors, and the updated pair queue using `hf.save_results(new_results, folder_path)`, `hf.save_errors(new_errors, folder_path)`, and `hf.save_pair_queue(new_pair_queue, folder_path)`.

7. **Batch Continuation**:  
   If there is a batch ID (indicating that further batches are required), the script saves the current batch ID using `hf.save_batch_id(batch_id, folder_path)` and prints a message indicating that another batch will run. If no further batches are needed, it prints "Finished Program," signaling that all pairs have been processed.

8. **Batch Still Running**:  
   If batch results have not yet been returned from the API, the script will print "Batch still running," indicating that processing is awaiting results from a previous API request.


