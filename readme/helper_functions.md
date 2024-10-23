# Helper Function README

This module, `Helper Function v7`, assists with loading, saving, and processing tasks related to municipal (Muni) zoning questions paired with a specific municipality. It is used as part of a broader project, with the main logic residing in the `QA Code` and `QuestionMuniPair` classes. The primary responsibilities of this module include managing intermediate states, batching requests, and handling SLURM integration for distributed processing.

## Overview

The module provides utility functions to handle various aspects of asynchronous API processing, including:
- Managing `QuestionMuniPair` objects.
- Saving intermediate states between batch requests.
- Tracking errors during processing.
- SLURM job scheduling for parallelization.

This code integrates seamlessly with both SLURM environments (for distributed processing) and non-SLURM environments (for single-node execution).

## Key Functions

### 1. `load_pair_queue()`
- **Purpose**: Loads a queue of `QuestionMuniPair` objects for processing. If no queue is found, it initializes the pairs by creating a combination of questions and municipalities.
- **Inputs**: 
  - `folder_path`: Directory where the pair queue is saved.
  - `questions`: DataFrame containing questions.
  - `munis_for_task`: List of municipalities.
  
### 2. `build_model()`
- **Purpose**: This function processes the `QuestionMuniPair` objects and batches them for asynchronous API calls. It ensures that token and request limits are not exceeded.
- **Key Features**:
  - Processes each pair through the state machine and tracks their progression.
  - Batches requests to OpenAI’s API, adhering to limits on tokens and requests.
  - Handles and logs errors to ensure no data is lost during processing.

### 3. `save_results()` & `save_errors()`
- **Purpose**: Save intermediate results and errors to pickle files. These functions are called after each batch to store the results and any encountered errors.

### 4. `update_pairs()`
- **Purpose**: Updates the `QuestionMuniPair` queue with the results from the latest API batch. Matches results back to the corresponding pair using their `custom_id`.

### 5. `slurm_name()`
- **Purpose**: Generates a task-specific suffix for filenames based on the SLURM array task ID, ensuring that output files from different nodes do not conflict. Defaults to `_1` if not running in a SLURM environment.

### 6. `get_munis()`
- **Purpose**: Loads a list of municipalities from an Excel file and prepares it for processing, optionally filtering the list based on the sample size set in the configuration.

### 7. `split_munis_across_nodes()`
- **Purpose**: Splits the list of municipalities across multiple nodes for parallel processing when using SLURM.

## SLURM Integration

This module automatically handles SLURM integration. If the SLURM environment variables are present, the code splits the task across multiple nodes. If not, it runs on a single node by default.

Key SLURM-related functions include:
- **`get_num_nodes()`**: Detects how many nodes are available for parallelization.
- **`slurm_name()`**: Handles task-specific filenames for nodes running in a SLURM environment.
- **`split_munis_across_nodes()`**: Splits the list of municipalities between nodes when parallelized via SLURM.

## Error Handling

Errors encountered during processing (e.g., issues with municipal embeddings or API errors) are logged into the `errors.pkl` file. Each error is stored along with the state of the `QuestionMuniPair` object and the full error message. This allows for easy tracking and reprocessing of failed requests.

## Token and Request Limits

The OpenAI API imposes constraints on the maximum number of tokens and requests that can be processed in one batch. This code respects those limits by tracking:
- **MAX_BATCH_TOKENS**: The maximum number of tokens allowed in a single batch. Defined in the `config.yaml` file.
- **MAX_REQUESTS**: The maximum number of requests allowed in a single batch. Also defined in the `config.yaml` file.

Both limits are configurable based on the user's API quota and can be updated in the `config.yaml` file.

## Configurable Parameters

The module pulls several configuration parameters from the `config.yaml` file:
- **max_batch_tokens**: Maximum number of tokens allowed per batch.
- **max_requests**: Maximum number of API requests allowed per batch.
- **model_type**: The OpenAI model to be used (e.g., `gpt-4`, `gpt-3.5`).
- **sample_size**: The size of the municipal sample to be processed, either `'all'` or a subset for testing.

These settings allow users to adjust the processing to suit their specific API quota and testing needs.

## File Management

The following files are generated during processing:
- **`pair_queue.pkl`**: Tracks the current state of `QuestionMuniPair` objects awaiting processing.
- **`output.pkl`**: Stores successfully processed results.
- **`errors.pkl`**: Logs errors encountered during processing.
- **`batch_results.pkl`**: Contains the results of completed batches.
- **`batch_id.txt`**: Tracks the ID of the current batch being processed.

These files are saved in the specified `folder_path` for easy access and reprocessing if needed.

## Data Loading

The input data, such as questions, subtasks, and municipalities, are loaded from Excel files specified in the `config.yaml` file. Details on the structure and content of these files will be explained in a separate README covering the data preparation process.

