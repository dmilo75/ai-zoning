
---

# Embedding Preparation Module (embedding_v4.py)

## Description
This module prepares and embeds scraped municipal text as well as zoning questions for context building. It first prepares the text by combining small sections and splitting larger sections, ensuring that each section is appropriately sized for embedding. It then embeds the text into semantic vectors using OpenAI’s `text-embedding-3-large` model. Both the processed and embedded data are stored for later use in answering zoning-related questions.

## Key Functions

### 1. **Text Processing**
The text processing step prepares municipal text for embedding. This includes:
- **Combining small sections**: Adjacent tiny sections are merged.
- **Splitting large sections**: Sections exceeding the token threshold (1,000 tokens) are split into smaller chunks.
- **Threshold and Overlap**: The `threshold` is set to 1,000 tokens, with an overlap set at 200 tokens.

### 2. **Embedding Text**
The text is embedded into semantic vectors using the OpenAI `text-embedding-3-large` model. The `prepare_text` function appends each section with its path and calculates the number of tokens to ensure the section is ready for embedding.

- **Embedding Customization**: While OpenAI embeddings are the default, the code is flexible and can be adjusted for other embedding providers if needed.

### 3. **Parallelization Using SLURM**
This module is optimized to run in parallel on multiple nodes using SLURM job arrays. It automatically detects SLURM and splits the workload across nodes. If SLURM is not detected, the code runs in serial.

- **Other ways to paralellize**: The code is designed for parallelization on SLURM, though users may want to consider asynchronous batching.

### 4. **Question Embedding**
In addition to municipal text, this module also embeds questions from the `Questions.xlsx` file. The question embeddings are processed alongside the municipal text embeddings and stored in a pickle file for later use.

### 5. **Flags**
- **`skip_embed`**: Allows users to skip re-embedding files that have already been processed. This is useful for incremental additions of new municipal text without reprocessing existing files.
- **`skip_text_process`**: Skips the text processing step. This flag is used when more intensive preprocessing is not required but users still want to rerun the embedding process.

## File Processing and Storage
The `get_files` function retrieves the ordinance text as scraped. The ordinacne text is expected to be in a heirarchical dictionary structure with keys representing titles from levels in the table of contents and leaf nodes containing text. The get files function uses `Sample Data.xlsx` to find the list of all municipalities and then selects which municipalities will be run on the specific node if using a job array. For each selected municipality the scraped heirarchical dictionaries are imported.

The code expects scraped text to be stored in a certain format. Under the muni_text storage path defined in config.yaml the user should create a subfolder for each source of text (i.e. Municode, American Legal Publishing, and Ordinance.com) and then within each source folder a subfolder for each state. Files are are also expected to be name by the local government name (as defined in the Census of Governments variable UNIT_NAME), the CENSUS_ID_PID6 unique identifier for the municipality (again from the Census of Governments) and the state that the municipality is located in. The final file should be stored as a pickle file (.pkl) with the name muni_name + '#' + census_id_pid6+'#'+state.

- **Storage Path**: Users can define the storage location for embeddings and preprocessed files in `config.yaml` using the following parameters:
  ```yaml
  muni_text: "/path/to/municipal/text"      # Directory where raw municipal text is stored
  embeddings: "/path/to/embeddings"         # Directory where embeddings are stored
  pre_embed: "/path/to/pre_embed"           # Directory for preprocessed text before embedding
  raw_data: "/path/to/raw/data"             # Directory for raw data such as questions
  openai_key: "your-openai-api-key"         # OpenAI API key for embedding
  ```

## SLURM and Parallelization
The code is set up to distribute embedding tasks across multiple nodes in a supercomputer environment using SLURM job arrays. Each node processes a subset of the files, ensuring efficient parallel processing of large datasets.

- **Task ID Handling**: The SLURM task ID is used to determine which files a node will process, while the first node is assigned to embed the questions.
