# NYU AI-Zoning Project

The Advent of Large Language Models (LLMs) has transformed many facets of society, enabling groundbreaking applications across diverse fields. This project aims to leverage LLMs to analyze and study the zoning landscape in the United States. This current repository offers a demo model that is tested with zoning ordinances extracted from Wheaton, Illinois. 

For the latest paper please see [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4627587). 

For the latest data please see [here](https://www.dropbox.com/scl/fi/x6zr6ejh63frx65qneq92/Public-Dataset-v1.xlsx?rlkey=zxjjj2j7tt54imatif2gjddql&st=u6h0kmhk&dl=0).

The successful running of the demo model depends on the following code files:  
config.yaml  
context_building.py  
embedding.py  
gpt_functions.py  
helper_functions.py  
model_batch.sbatch  
QA_Code.py  
question_muni_pair.py  
.env  

All are included in the repository besides the .env file, which you will need to create in the main AI-Zoning directory (same directory as config.yaml). A guide on what to include in you .env file, as well as documentation for all the key components of the repository, can be found in the readme folder of this repository.  

The code in this repository was containerized using Docker. To demo the code, please refer to the following guide on running containerized code: https://drive.google.com/file/d/1BiEs74T4dKHhyQI2Je3EUJxNfzEcvsD0/view?usp=sharing
