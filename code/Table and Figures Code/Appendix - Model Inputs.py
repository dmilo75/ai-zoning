import pandas as pd
import os
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def read_question_background(config, question_id):
    background_file = os.path.join(config['raw_data'], 'question_backgrounds', f"{question_id}.txt")
    try:
        with open(background_file, 'r', encoding='utf-8') as file:
            background = file.read().strip()
    except FileNotFoundError:
        background = ""
    return background

def read_question_keywords(config, question_id):
    keywords_df = pd.read_excel(os.path.join(config['raw_data'], 'Keywords.xlsx'), dtype=str)
    if int(question_id) in keywords_df.columns:
        keywords = keywords_df[int(question_id)].dropna().tolist()
        return keywords
    return []

def read_question_subtasks(config, question_id):
    subtasks_df = pd.read_excel(os.path.join(config['raw_data'], 'Subtasks.xlsx'))
    subtasks_df['Subtask Questions'] = subtasks_df['Subtask Questions'].astype(str)
    subtask = subtasks_df[subtasks_df['Subtask Questions'] == question_id].to_dict('records')
    return subtask[0] if subtask else None

def format_question_latex(question, config):
    latex_code = f"\\subsection*{{Question {question['ID']}}}\n"

    if question['ID'] in [27,28]:
        latex_code += f"\\noindent\\textbf{{How We Phrased the Question:}} {question['Pioneer Question']}\n\n"
    else:
        latex_code += f"\\noindent\\textbf{{Question Phrased by Pioneer:}} {question['Pioneer Question']}\n\n"

    latex_code += f"\\noindent\\textbf{{Question Text That We Embed:}} {question['Question Detail']}\n\n"
    if question['Background']:
        latex_code += f"\\noindent\\textbf{{Question Background and Assumptions:}} {question['Background']}\n\n"
    latex_code += f"\\noindent\\textbf{{Question Type:}} {question['Question Type']}\n\n"
    latex_code += f"\\noindent\\textbf{{Rephrased Question the LLM Sees:}} {question['Question Rephrase']}\n\n"

    if pd.notna(question['If The Answer Is Not This Value Then We Double Check']):
        latex_code += f"\\noindent\\textbf{{If The Answer Is Not This Value Then We Double Check:}} {question['If The Answer Is Not This Value Then We Double Check']}\n\n"
        latex_code += f"\\noindent\\textbf{{Rephrased Question the LLM Sees When Double Checking:}} {question['Double Check Question']}\n\n"
        keywords = read_question_keywords(config, question['ID'])
        if keywords:
            latex_code += "\\noindent\\textbf{Keywords We Use to Build Context When Double Checking in Order of Importance:}\n"
            keywords = ["'"+keyword+"'" for keyword in keywords]
            if len(keywords) > 1:
                keywords[-1] = f"and {keywords[-1]}"
                keywords = ', '.join(keywords)
                latex_code += f"\\noindent {keywords}\n"
            else:
                latex_code += f"\\noindent {keywords[0]}\n"

    subtask = read_question_subtasks(config, question['ID'])
    if subtask:
        latex_code += "\\noindent\\textbf{Subtask:}\n"
        latex_code += "\\begin{itemize}\n"
        latex_code += f"\\item Subtask Question That Gets Embedded: {subtask['Question Detail']}\n"
        latex_code += f"\\item Rephrased Subtask Question the LLM Sees: {subtask['Question Rephrase']}\n"
        latex_code += f"\\item Additional Subtask Instructions: {subtask['Subtask Instructions']}\n"
        latex_code += f"\\item How The Subtask Results Are Described to the LLM Afterwards: {subtask['Subtask Results Description']}\n"
        latex_code += "\\end{itemize}\n"
        latex_code += "\n"

    latex_code += "\\vspace{1cm}\n"
    return latex_code

def generate_latex_appendix(config):
    latex_code = "\\section*{Appendix: Question Details}\n"
    latex_code += "This appendix provides detailed information about each question used in the study. Each question is presented with its original phrasing by the Pioneer Institute, the text that we embed for the question, background information and assumptions, question type, and the rephrased question that the language model sees. For some questions, we also include a value that triggers double-checking if the model's answer does not match it, along with the rephrased question used for double-checking and the keywords used to build context during the double-checking process. Additionally, certain questions involve subtasks, which are described in detail.\n\n"

    questions_df = pd.read_excel(os.path.join(config['raw_data'], 'Questions.xlsx'))

    for _, row in questions_df.iterrows():
        question_id = str(row['ID'])
        background = read_question_background(config, question_id)

        question = {
            'ID': question_id,
            'Pioneer Question': row['Pioneer Question'],
            'Question Detail': row['Question Detail'],
            'Question Rephrase': row['Question Rephrase'],
            'Question Type': row['Question Type'],
            'Background': background,
            'If The Answer Is Not This Value Then We Double Check': row['Prior'],
            'Double Check Question': row['Double Check Question'] if pd.notna(row['Double Check Question']) else row['Question Rephrase'],
        }

        latex_code += format_question_latex(question, config)

    return latex_code

latex_appendix = generate_latex_appendix(config)

#Export the latex
with open(os.path.join(config['tables_path'],'latex', 'appendix.tex'), 'w', encoding='utf-8') as file:
    file.write(latex_appendix)