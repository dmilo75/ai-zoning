# AI and Zoning

This is the Github repository for a project using Large Language Models (LLMs) to parse zoning documents. We introduce a new approach to decode and interpret statutes and administrative documents employing LLMs for data collection and analysis that we call generative regulatory measurement. We use this tool to construct a detailed assessment of U.S. zoning regulations. We estimate the correlation of these housing regulations with housing costs and construction. Our work highlights the efficacy and reliability of LLMs in measuring and interpreting complex regulatory datasets.

For the latest paper please see [here](https://static1.squarespace.com/static/56086d00e4b0fb7874bc2d42/t/653b143abbdc5f5bfacf947a/1698370623319/AI_Zoning.pdf).

Please note that all LLMs used in this project have a degree of randomness and thus cannot be exactly replicated. We limit the degree of randomness by, for example, setting the temperature of each model to very low levels. Replication results should still be very similar. 

# Setup

**Main dependencies:** Please see `requirements.txt` for a full list of the Python packages that need to be installed to replicate the project. Run the below command in the terminal to install dependencies.

`pip install -r requirements.txt`

### Overview

#### Embeddings setup
![llm drawio (3)](https://github.com/dmilo75/ai-zoning/assets/64778259/cd894ecb-9c9a-4359-af70-7c06f66fae3c)

1) You can clone the repository using:
   
   `git clone https://github.com/dmilo75/ai-zoning.git`
3) To create embeddings, we have to configure config.yaml for
   - muni_text: path where the raw text of all municipalities is stored.
   - embeddings: path where you want to store the embeddings.
   - raw_data: path where the questions are present
4) After setting the paths we can run embedding.py
   
   `python3 "path-to-repository/code/Main Model Code/embeddings.py"`

This process creates embeddings from raw text and stores them in the embeddings path provided in config.yaml file.

#### Llama2 or Chat GPT inference
![llm1 drawio (1)](https://github.com/dmilo75/ai-zoning/assets/64778259/da656361-171c-45e7-ad02-3d4d3eeffc03)

Steps 1 and 2 should be done if planning to use the llama2 model. Else, it is not required.

1) For downloading llama2 13B/70B GPTQ, you can run the Python script (ai-zoning/code/download.py) that can download the model from huggingface. Make sure to change the download directory while using the code.
2) Clone the exllama repository in a separate directory using:
   
   `git clone https://github.com/turboderp/exllama.git`.
4) To perform inference, we have to configure config.yaml for
   - exllama_path: the path of the cloned exllama repository if using llama2.
   - llama13b_path/llama70b_path: the path where the llama2 model is downloaded.
   - openai_key: if you are using GPT, OpenAI API key should be provided here.
   - processed_data: path where you want to store the inference results.
   - num_neighbors: optional. Determines the number of chunks of text to include in the context.
5) After setting the paths we can run QA_Code_V3.py
   
   `python3 "path-to-repository/code/Main Model Code/QA_Code_V3.py"`

This process gathers inference results on the LLM and stores them inside the processed_data path provided in config.yaml file.

More detailed information has been provided further.

### Hardware and Runtime

To run the code on the Llama2 13B model, it takes 27GB of GPU memory per node.

For running the Llama2 13B model on the whole of the national sample, after data parallelization, it took 7 hours for each node with a total of 50 NVIDIA Quadro RTX 8000 nodes.

### Creating and Organizing Municipal Ordinance Text Files

The 'Sample_Data' Excel in the `raw data` contains the list of all municipalities in our sample along with their source and unique identifiers. The file was created by merging the 2022 Census of Governments [dataset](https://www.census.gov/programs-surveys/cog.html) with the metadata (name, state, zip code, website, address, etc.) for all scraped municipalities. As a result, please reference the Census of Governments data dictionary [here](https://www.census.gov/data/datasets/2022/econ/gus/public-use-files.html) for variable definitions. We will reference this file to scrape text and format the filename of ordinance text files. All scraped text is stored in a directory defined by the user as 'muni_text' in the `config.yaml` file.

**Scraping Ordinance Text**

The text for each ordinance must be scraped from one of the three following sources:

- [Municode](https://library.municode.com/)
- [American Legal Publishing](https://codelibrary.amlegal.com/)
- [Ordinance.com](https://ordinance.com/)

The 'Source' column in the `Sample_Data` Excel file indicates the source used for each municipality. Use this column to identify where to scrape the text for each ordinance file.

Take caution when scraping tables. While American Legal Publishing and Municode store tables in html format (which can easily be scraped), Ordinance.com stores tables in image format. Since LLMs available at the time of the study only handle text, we use [Amazon Textract](https://aws.amazon.com/textract/) to extract text from images of tables from Ordinance.com. 

**Naming Convention and Source Reference**

Use the following convention for naming text files. Note that the variable names like 'UNIT_NAME' correspond to columns in the 'Sample Data' excel. 

- **UNIT_NAME**: The name of the municipal unit (e.g., "alabaster").
- **FIPS_PLACE**: Use the municipalities FIPS place code (e.g., "820"). Please note that for townships this is the county subdivision code. 
- **State**: The state's two-letter abbreviation (e.g., "al").

The naming format for the text files should be: `UNIT_NAME#FIPS_PLACE#State.txt`.

For example, see the below:

```plaintext
/muni_text
  /al
    alabaster#820#al.txt
    albertville#988#al.txt
  /ak
    anchorage#3000#ak.txt
    angoon#3440#ak.txt
```

### Setting up LLMs

**Llama-2 Based Models**

To use llama-2 based models you must first register with Meta [here](https://ai.meta.com/llama/). Once, registered please download the specific models used in the analysis from Hugging Face which houses quantized models compatible with exllama. We use the following models:
1.[ Llama-2-13B-GPTQ ](https://huggingface.co/TheBloke/Llama-2-13B-GPTQ) under the 'main' branch. 
2. [Llama-2-70B-GPTQ](https://huggingface.co/TheBloke/Llama-2-70B-GPTQ) under the 'main' branch.

We also use [Exllama](https://github.com/turboderp/exllama) to speed up processing time. Please download this as well and provide the appropriate path.

**Chat GPT Based Models**
For Chat GPT based models please register for an account with OpenAI [here](https://platform.openai.com/signup). Please place your API key in `config.yaml`. We use the model 'gpt-4' for Chat GPT 4 and 'gpt-3.5-turbo' for Chat GPT 3.5; see [here]([url](https://platform.openai.com/docs/models)) for a list of all available Chat GPT based models. 

# code

The code is split into three parts/folders: `Pre Processing Code`, `Main Model Code`, and `Table and Figures Code`. 

### Config setup

The user must provide a set of paths in `config.yaml`.

1. Model input data
- `muni_text`: This is the path of the directory that contains all municipal text files.
- `embeddings`: This is the path of the directory with all embedding files. `embedding.py` produces the files for this directory.
2. LLM paths
- `exllama_path`: Path where exllama is stored
- `llama13b_path`: Path where the model for 11ama-2 13b is stored.
3. Main folder paths
- `processed_data`: Path to `processed data` directory
- `raw_data`: Path to `raw data` directory
- `shape_files`: Path to shape files directory
- `figures_path`: Path to figures folder in `results`
- `tables_path`: Path to tables folder in `results`

The user also can input their OpenAI (Chat GPT) API Key and the number of neighbors used to construct context. Be careful when adjusting the number of neighbors used to construct context. Most models used in our study (all except for Chat GPT 4) use 4k tokens of context. If you increase the number of neighbors (or number of relevant chunks of text) then you must decrease the size of each text chunk in `embedding.py`. 

### Pre-Processing Code
Pre-processing consists of three scripts
1. `Download Shape Files.py`: Given a directory to store shape files under `shape_files` in `config.yaml`, this script will download all relevant US Census shape files for counties, county subdivisions, places, urban areas, and water areas. We use the 2022 TIGER/Line shape files downloaded from [here](https://www2.census.gov/geo/tiger/TIGER2022/), but you may find the web interface more helpful [here](https://www.census.gov/cgi-bin/geo/shapefiles/index.php) if manually downloading shape files.
2. `Make % Urban.py`: This script calculates the percent overlap between the 2022 shape files for municipalities and the 2020 shape file for urban areas. It produces the Excel file `urban_raw.xlsx` in the `raw data` directory.
3. `Merge Raw Data.py`: This script merges all raw housing/demographic data (American Community Survey data, Building Permits data, Urban Area, and MSA classifications) with our municipality identifier dataset `Sample Data.xlsx`. The resultant excel file is `Sample_Enriched.xlsx` located in `processed data`.

### Main Model Code

The embeddings and Q&A Code (which runs Llama-2) require powerful graphics cards (we use the RTX8000). Each of these codes requires SLURM job arrays, which we use to parallelize the code across several graphics cards/processors. Note that these scripts can be adapted to use other forms of parallelization or to run in serial. 

#### Embeddings Code (`embeddings.py`)

Below is an example SLURM job array request for the embeddings file which parallelizes the code across two nodes. 

```bash
sbatch --array=0-1 embed.sbatch
```

Your corresponding `embed.sbatch` batch file should be similar to the following, but please consult the staff at your High Performing Computer center for more advice:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=embed
module purge
singularity exec --nv \
  --bind /path/to/cache:$HOME/.cache \
  --overlay /path/to/pytorch-ext:/path/to/pytorch-ext:ro \
  /path/to/singularity-image.sif\
  /bin/bash -c "source /path/to/env.sh; python3 embedding.py"
```

Note: The above uses Python with singularity. 

#### Q&A Code (`QA_Code_V3.py`)

The QA Code is the question-and-answer code or the main hub for running the model. For the `QA_Code_V3.py` script, you must specify 1-2 system arguments:

1. The first argument is for choosing the model. Valid options are:
   - '3.5' for Chat GPT 3.5
   - '4' for Chat GPT 4
   - '13b' for llama-2 13b
   - '70b' for llama-2 70b
Note that when using a Chat GPT based model you do not need any graphics card since the code just calls the Chat GPT API where the heavy lifting happens in the cloud. You will still benefit from a powerful graphics card for the embedding script (though it can still be run without one with significantly reduced speed). 

2. The second argument is only relevant when running the code on one node (i.e. not using a SLURM job array). This is an integer representing the sample size (number of munis) to use from the training sample (107 randomly selected munis from the Pioneer Institute [study](http://www.masshousingregulations.com/Qforcat.asp?nval=3&step=2&id=3)). Use this parameter for either testing the code (by running on small samples) or for comparing model performance to the Pioneer Institute. 

Example of a job array parallelized across 50 nodes, run with a single argument '13b' for Llama-2 13B::

```bash
sbatch --array=0-49 qacode.sbatch 13b
```
Example job run with one argument '13b' run on one node (thus only run on the training sample):

```bash
sbatch qacode.sbatch 13b
```
Example job run with two arguments '13b' and '5' (thus run on a random sample of 5 municipalities from the training sample):

```bash
sbatch qacode.sbatch 13b 5
```
Your corresponding `qacode.sbatch` batch file should be similar to the following, but please consult the staff at your High Performing Computer center for more advice:

```bash
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=qacode
module purge
singularity exec --nv \
  --bind /path/to/cache:$HOME/.cache \
  --overlay /path/to/pytorch-ext:/path/to/pytorch-ext:ro \
  /path/to/singularity-image.sif\
  /bin/bash -c "source /path/to/env.sh; python3 QA_Code_V3.py $1 $2"
```

Note: The above uses Python with singularity. 

The QA Code also calls on three other scripts:

1. **Helper Functions (`helper_functionsV2.py`):** This script contains various functions used by the QA code like finding relevant text and querying the LLM. 
2. **Chat GPT Functions (`gpt_functions.py`):** To elicit structured responses (for example, binary responses) from Chat GPT we use function calling. Please see [here](https://platform.openai.com/docs/guides/gpt/function-calling) for further details.
3. **Kor Helper (`korhelper.py`):** [Kor](https://github.com/eyurtsev/kor) allows us to map open-ended unstructured Llama-2 responses to parseable answers like "Yes", "No", or "I don't know". We provide training data for Kor in the `raw data` directory under the Excel file `Kor Training.xlsx`. 

#### Cleaning Llama output (`clean_llama_output.py`)

This script finishes any remaining answer parsing from Kor and calculates the minimum minimum lot size and the mean minimum lot size for each municipality. 

### Table and Figures Code
There are three scripts to make tables and figures
1. `map_maker.py`: This script loops over a few select cities and questions creating maps of the regulatory environment in that area. Please ensure that you have populated a shape file folder and provided its path in `config.yaml` before running this script.
2. `Table 1.py`: This script creates tables 1 from the paper about our dataset coverage.
3. `Tables 2 and A2 and Figures Histogram and Cleveland Dot.py`: As the title suggests, this script creates charts for tables 2, A2, and for the histogram and Cleveland dot plot figures. 

# processed data
Processed data consists of model output and a merged dataset with municipality characteristics. A folder for each model (llama-13b, llama-70b, gpt-35-turbo, and gpt-4) will contain the model output in pickle format. The model output for llama-2 based models is also available in Excel format as the output from `clean_llama_output.py`. Finally, various municipality characteristics used to make figures and tables are stored in `Sample_Enriched.xlsx`. 

# raw data

Information on `Sample Data.xlsx` is explained previously. 

- `Kor Training.xlsx` is a collection of manually parsed raw llama-2 output that serves as training data for Kor to parse further responses. To edit/add more training data simply take open-ended responses from llama-2 and manually parse the answer. 
- `msa_crosswalk_mar_2020.xlsx` maps counties to their respective MSA. We use this to understand our MSA population coverage. 
- `Questions.xlsx` is the list of questions used for the analysis along with an ID for each question and a categorization of the type of question (i.e. 'Binary' or 'Numerical'). You may add additional questions here if needed. 
- `bps_raw.xlsx` refers to the building permits survey data. We use the 2022 Building Permits Survey data. For a data dictionary see [here](https://www2.census.gov/econ/bps/Documentation/placeasc.pdf
- `training.pickle` is a list of the random sample of 107 of the municipalities used in the Pioneer Institute study. We use this cut of the dataset to check model performance as we iterate. We do not test performance on the remaining municipalities to have data that we can fine-tune on and test after improvements have been made.
- `ACS Data` is a folder with all American Community Survey variables relevant to the analysis.
- `2022 Population Data` contains data on MSA and state level population data in 2022. 

# results
This directory holds the results in the form of Excel file tables in the `tables` folder and in the form of images in the `figures` folder. All tables and figures in this directory are generated from code found in `code`->`Table and Figures Code`

# Extra

**Contact Info**
Please contact dm4766@stern.nyu.edu with any inquiries. 

**License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

