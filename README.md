# NYU AI-Zoning Project

The Advent of Large Language Models (LLMs) has transformed many facets of society, enabling groundbreaking applications across diverse fields. This project aims to leverage LLMs to analyze and study the zoning landscape in the United States. This current repository offers a demo model that is tested with zoning ordinances extracted from Wheaton, Illinois. 

For the latest paper please see [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4627587). 

For the latest data please see [here](https://www.dropbox.com/scl/fi/x6zr6ejh63frx65qneq92/Public-Dataset-v1.xlsx?rlkey=zxjjj2j7tt54imatif2gjddql&st=u6h0kmhk&dl=0). 

All are included in the repository besides the .env file, which you will need to create in the main AI-Zoning directory (same directory as config.yaml). A guide on what to include in you .env file, as well as documentation for all the key components of the repository, can be found in the readme folder of this repository.  

The code in this repository was containerized using Docker. To demo the code, please refer to the following guide on running containerized code: https://drive.google.com/file/d/1BiEs74T4dKHhyQI2Je3EUJxNfzEcvsD0/view?usp=sharing

# AI and Zoning: Using Large Language Models for Regulatory Analysis

This repository is dedicated to the use of Large Language Models (LLMs) for parsing zoning documents. We introduce a generative regulatory measurement approach to decode and interpret statutes and administrative documents. This project leverages LLMs to construct a detailed assessment of U.S. zoning regulations and examines the correlation between these regulations, housing costs, and construction. 

The work demonstrates the reliability of LLMs in analyzing complex regulatory datasets.

For the latest paper, please see [here](https://static1.squarespace.com/static/56086d00e4b0fb7874bc2d42/t/653b143abbdc5f5bfacf947a/1698370623319/AI_Zoning.pdf).

## Table of Contents
1. [Setup](#setup)
2. [Overview](#overview)
3. [Code Structure](#code-structure)
4. [Data Structure](#data-structure)
5. [Results](#results)
6. [Contact and License](#contact-and-license)

## Setup

### Main Dependencies
To install the Python packages required for this project, please run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Overview

The project is divided into several key components:

1. **Embeddings Setup:** This process creates embeddings from raw text and stores them in the defined path.
2. **LLM Inference:** We use OpenAI models (e.g., GPT-3.5, GPT-4) to process zoning questions and generate structured answers.
3. **Data Preprocessing:** This includes downloading shape files, merging raw housing/demographic data, and preparing municipality identifier datasets.
4. **Main Model Code:** The core processing logic handles question-municipality pairs, organizes the data flow, and runs the model using parallel processing.

### For a visual overview of the embedding and inference processes, see the diagrams linked below:
- Embeddings Workflow
- Inference Workflow

## Code Structure

The codebase is organized into several key folders and files:

1. **Raw Data**  
   - Contains the raw zoning and housing-related data necessary for analysis.
   
2. **Processed Data**  
   - Stores the output from embedding processes, LLM inferences, and additional processed datasets.
   
3. **Code**  
   - Split into the following sections:
     - **Pre-Processing Code:** Prepares raw datasets for analysis, including downloading shape files and merging data.
     - **Main Model Code:** Runs the core model logic, including LLM inference and embeddings.
     - **Tables and Figures Code:** Generates the tables and figures used for analysis and reporting.

### Key Files:

- [Configuration Setup](readme/config.md): Defines paths and settings required for the embedding and LLM processes, including API keys and paths to data directories.
- [Context Building Code](readme/context_building.md): Builds the context needed to process and answer zoning-related questions.
- [Embedding Code](readme/embedding.md): Manages embedding processes, ensuring raw text is split into manageable sections.
- [GPT Functions](readme/gpt_functions.md): Helper functions for interacting with OpenAI's API, including token counting and batching logic.
- [Helper Functions](readme/helper_functions.md): Utility functions for data management and error handling.
- [Question-Answer Code](readme/qa_code.md): Core logic for processing question-municipality pairs and managing LLM inferences.
- [Question-Municipality Pairing](readme/question_muni_pair.md): Manages the lifecycle of question-municipality pairs, including initialization and embedding.
- [Model Batch Process](readme/model_batch.md): Batch processing logic for SLURM job arrays, allowing for distributed computing across nodes.

## Data Structure

### Raw Data

The `raw_data` folder contains several essential datasets:

- **Sample Data.xlsx:** List of municipalities and their zoning ordinances.
- **American Community Survey Data:** Demographic and housing data for municipalities.
- **Building Permits Survey Data:** 2022 building permits data, detailing construction permits issued by local authorities.
- **Questions.xlsx:** List of questions used in the analysis, including binary and numerical categories.
- **Kor Training Data:** Manually parsed llama output used to train Kor, which maps unstructured LLM responses to structured answers.

### Processed Data

Processed data includes:

- **Embeddings:** Contains text embeddings created from raw zoning data, stored in the directory defined in `config.yaml`.
- **Model Output:** Inference results from LLMs, stored in separate folders for each model (GPT-3.5, GPT-4).
- **Enriched Sample Data:** A merged dataset of municipality characteristics, zoning regulations, and additional variables used for analysis.

## Results

The `results` folder contains the outputs of model runs and visualizations:

- **Tables Folder:** Contains Excel files with table outputs from the analysis.
- **Figures Folder:** Contains images of charts and maps generated from the zoning analysis.

Tables and figures can be recreated by running the appropriate scripts in the `Table and Figures Code` section.

## Contact and License

**Contact Information**  
For any inquiries, please contact [dm4766@stern.nyu.edu](mailto:dm4766@stern.nyu.edu).

**License**  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

