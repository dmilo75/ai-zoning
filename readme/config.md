# Configuration Guide for `config.yaml`

This configuration file controls various aspects of the processing pipeline for context building and embedding generation. Below is an explanation of each key section.

## Filepaths
These are directories where different types of data and outputs will be stored. Ensure all directories are correctly set before running the pipeline.

- **muni_text**: Directory for the raw text data of all municipalities.
- **embeddings**: Directory for storing embeddings after processing.
- **pre_embed**: Directory for storing pre-processed text before embedding.
- **processed_data**: Directory for storing cleaned data.
- **raw_data**: Directory for raw data.
- **shape_files**: Directory for shapefiles used in mapping.
- **figures_path**: Directory for storing generated figures.
- **tables_path**: Directory for storing generated tables.
- **batch_folder**: Directory for storing batch processing results.

## Config Parameters
These parameters control how the pipeline interacts with external APIs and how data is processed.

- **max_tokens**: Controls the maximum number of tokens (in thousands) used for context building. For example, `max_tokens: 4` means the maximum context is 4,000 tokens.
- **max_batch_tokens**: Sets the total token limit in the queue for OpenAI API calls. This depends on your OpenAI usage tier, which can be found [here](https://platform.openai.com/settings/organization/limits). Currently set to the tier 5 limit.
- **max_batch_requests**: Limits the number of OpenAI requests in the queue. This also depends on the usage tier and is set to the tier 5 limit.
- **sample_size**: Determines how many municipalities to process. Set to `"all"` for full sample processing, or set to an integer (e.g., `30`) to select a random subset of municipalities for testing purposes.
- **model_type**: Specifies which OpenAI model to use (e.g., `"gpt-4-turbo-2024-04-09"`). You can change this to any OpenAI model, but if the model isn't already in the cost dictionary (found in `gptfunctions.py`), you'll need to manually add its cost features there.
- **fake_batch**: When set to `True`, this will simulate API batch processing synchronously, useful for testing on small samples. For general runs, leave this set to `False`.
- **export_folder_name**: Specifies the name of the folder where results will be stored.
- **testing_mode**: Set to `True` to only run on one municipality and one question to just test whether the code runs. 

