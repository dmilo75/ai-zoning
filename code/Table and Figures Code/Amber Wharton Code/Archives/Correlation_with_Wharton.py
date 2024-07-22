total_municipalities = 19429
##checked it from google
# from google.colab import files
#This is the code to upload the file to google colab. If use other platform, plz delete it.
# uploaded = files.upload()
import pandas as pd
import yaml
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load filepaths
data_path = config['processed_data']
raw_path = config['raw_data']
figures_path = config['figures_path']

#%%First, let's draw in all of the data

#Questions
questions = pd.read_excel(os.path.join(raw_path,"Questions.xlsx"))

#Cleaned llama output
fullrun = pd.read_excel(os.path.join(data_path,"Full Run.xlsx"),index_col = 0)
questions = list(set(fullrun["Question"]))
print(len(questions))

 # to check the question list
types = list(set(fullrun["Type"]))
types = types[0:1]+types[2:]
# types#to saperate 27 into two question in the columns.
#adjust the format of our data
for i in questions:
  if i == questions[0]:
    edited_fullrun = fullrun[fullrun["Question"]==i]
    order = ["Unnamed: 0","Muni","State","geoid","Answer","Question"]
    edited_fullrun = edited_fullrun[order]
    edited_fullrun = edited_fullrun.rename({"Answer":"A2", "Question":"Q2"}, axis='columns')
    print(len(edited_fullrun))
  else:
    if i == 27:
      for j in types:
        question_filtered = fullrun[(fullrun["Question"]==i) & (fullrun["Type"]==j)]
        question_filtered = question_filtered[["Answer","Question","geoid"]]
        question_filtered = question_filtered.rename({"Answer":"A"+str(i)+j, "Question":"Q"+str(i)+j}, axis='columns')
        edited_fullrun = pd.merge(edited_fullrun, question_filtered, on ="geoid")
    else:
      question_filtered = fullrun[fullrun["Question"]==i]
      question_filtered = question_filtered[["Answer","Question","geoid"]]
      question_filtered = question_filtered.rename({"Answer":"A"+str(i), "Question":"Q"+str(i)}, axis='columns')
    ##For reading, I change Answer into A, and Question in to Q
      edited_fullrun = pd.merge(edited_fullrun, question_filtered, on ="geoid")
    print(len(question_filtered))
edited_fullrun.to_excel("edited_fullrun.xlsx")

wharton = pd.read_stata(os.path.join(raw_path,"WRLURI_01_15_2020.dta"))
wharton = wharton.rename({"cencus_id_pid6":"geoid"}, axis='columns')
print(wharton.head())



