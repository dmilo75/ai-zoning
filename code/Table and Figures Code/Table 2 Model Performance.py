import pandas as pd
import pickle
import os
import yaml
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model = 'Testing Sample'

# Draw in pioneer pickle file from raw_data
with open(os.path.join(config['raw_data'], 'pioneer.pkl'), 'rb') as f:
    pioneer = pickle.load(f)

questions = pioneer.keys()

def adjust_index(index):

    try:
        index = int(index)
        return list(questions)[index - 1]
    except:
        if index == '27Mean':
            return 'Mean of Min Lot Sizes (Square Feet)'
        elif index == '27Min':
            return 'Minimum of Min Lot Sizes (Square Feet)'
        elif index == '28Mean':
            return 'Mean of Residential Min Lot Sizes (Square Feet)'
        elif index == '28Min':
            return 'Minimum of Residential Min Lot Sizes (Square Feet)'

        return index

def df_to_latex_panel(df, format_type='Default'):
    # Adjust index
    adjusted_index = [adjust_index(idx) for idx in df.index]
    df.index = adjusted_index
    df.index.name = 'Question'

    # Determine the number of decimal places for each column based on the format_type
    if format_type == 'Binary':
        decimal_places = [2, 0]
    else:
        decimal_places = [2, 2]

    # Round the values in each column based on the specified decimal places
    for col, dec in zip(df.columns, decimal_places):
        if dec == 0:
            #Make int
            df[col] = df[col].apply(lambda x: f"{int(x):d}" + "\\%")
        else:
            df[col] = df[col].apply(lambda x: round(x, dec))

    # Convert the dataframe to a LaTeX formatted string
    latex_str = df.to_latex(index=True, escape=False, header=True)

    # Cut out formatting stuff
    latex_str = latex_str.split('midrule')[1].split('bottomrule')[0][:-1].strip()

    # Add midrule for cumulative average
    latex_str = latex_str.split('Cumulative Average')[0] + "\\midrule\nCumulative Average" + \
                latex_str.split('Cumulative Average')[1]

    return latex_str.strip() + '\n\\bottomrule'


def make_latex_tables(perf_metrics, export = False):

    '''
    Table 2: Model performance (chat gpt4-t)
	• Column 1/2: includes ambiguous (RSE, corr)
	• Column 3/4: excludes ambiguous (RSE, corr)
	• Will need panel A and B for binary and cts
	• Wait for Scott
    '''

    # Establish dictionary with continuous and binary categories
    results = {'Continuous': pd.DataFrame(), 'Binary': pd.DataFrame()}
    for key in results:

        #Set first column to be 'With Ambiguous' and 'Main Performance Metric'
        #results[key]['RSE Amb'] = perf_metrics['With Ambiguous'][key]['Main Performance Metric']

        #Second column is 'With Ambiguous' and 'Alternative Performance Metric'
        #results[key]['Alt Amb'] = perf_metrics['With Ambiguous'][key]['Alternative Performance Metric']

        #Third column is 'Without Ambiguous' and 'Main Performance Metric'
        results[key]['RSE'] = perf_metrics['Without Ambiguous'][key]['Main Performance Metric']

        #Fourth column is 'Without Ambiguous' and 'Alternative Performance Metric'
        results[key]['Alt'] = perf_metrics['Without Ambiguous'][key]['Alternative Performance Metric']

        #Set index to be the question
        results[key].index = perf_metrics['Without Ambiguous'][key]['Question']

        #Now adjust the index
        results[key].index = [adjust_index(idx) for idx in results[key].index]

    # Set up the display options
    pd.set_option('display.max_colwidth', None)

    # Make cumulative means and format
    for cat, df in results.items():

        # Make cumulative means and medians
        df.loc['Cumulative Average'] = df.mean(axis=0)
        df.loc['Cumulative Median'] = df.median(axis=0)

        # Save the results
        results[cat] = df

    # Make latex
    latex = {}
    for cat, df in results.items():
        latex[cat] = df_to_latex_panel( df, format_type = cat)

    # Export to latex
    if export:
        with open(os.path.join(config['tables_path'], 'latex', 'Table 2 - Continuous Perf.tex'), 'w') as file:
            file.write(latex['Continuous'])

        with open(os.path.join(config['tables_path'], 'latex', 'Table 2 - Binary Perf.tex'), 'w') as file:
            file.write(latex['Binary'])

    return latex

def idk_latex(perf_metrics, export = False):

    #Establish dictionary with continuous and binary categories
    cats = ['Continuous','Binary']
    dic = {}
    for cat in cats:
        dic[cat] = pd.DataFrame()
        for dataset in perf_metrics:
            dic[cat][dataset] = perf_metrics[dataset][cat]['Don\'t Know']
        dic[cat].index = perf_metrics[dataset][cat]['Question']

    #Make cumulative means and format
    for cat, df in dic.items():
        df.loc['Cumulative Average'] = df.mean(axis=0)

        #Formatting the columns to have no decimals
        df = df.applymap(lambda x: f"{int(x):d}" + "\\%")
        dic[cat] = df

    #Make latex
    latex = {}
    for cat, df in dic.items():
        latex[cat] = df_to_latex_panel(df)

    if export:
        with open(os.path.join(config['tables_path'], 'latex', 'Table 2 - Continuous idk.tex'), 'w') as file:
            file.write(latex['Continuous'])

        with open(os.path.join(config['tables_path'], 'latex', 'Table 2 - Binary idk.tex'), 'w') as file:
            file.write(latex['Binary'])

    return latex

# Load in performance metrics
with open(os.path.join(config['processed_data'],'Model Output',model,'Performance Metrics.pkl'), 'rb') as f:
    perf_metrics = pickle.load(f)

# Make latex tables
latex = make_latex_tables(perf_metrics, export = 'Figure 2')


