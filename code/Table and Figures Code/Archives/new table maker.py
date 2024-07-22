import pickle
import pandas as pd
import numpy as np
from scipy.stats import mstats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

# Load config file
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load filepaths
data_path = config['processed_data']
raw_path = config['raw_data']
figures_path = config['figures_path']

# %%First, let's draw in all of the data

# Questions
questions = pd.read_excel(os.path.join(raw_path, "Questions.xlsx"))
sample_data = pd.read_excel(os.path.join(raw_path, "Sample Data.xlsx"))

sample_data = sample_data[['CENSUS_ID_PID6','FIPS_PLACE']]
# Cleaned llama output
df = pd.read_excel(os.path.join(data_path, "Full Run.xlsx"), index_col=0)
df.rename(columns={'fips': 'CENSUS_ID_PID6'}, inplace=True)
df = pd.merge(df,sample_data,on='CENSUS_ID_PID6',how='inner')

# Get sample data with ACS data
sample = pd.read_excel(os.path.join(data_path, "Sample_Enriched.xlsx"), index_col=0)

# Take the relevant variables and their identifiersr
rel = sample[['POPULATION', 'Single Unit Permits', 'Multi-Unit Permits', 'All Unit Permits', '% Urban',
              'Median Household Income_2021', 'Median Gross Rent_Percent_Change',
              'Median Home Value (Owner Occupied)_Percent_Change', 'Median Gross Rent_2021',
              'Median Home Value (Owner Occupied)_2021', 'FIPS_PLACE', 'STATE']]

df = pd.merge(df, rel, left_on=['FIPS_PLACE', 'State'], right_on=['FIPS_PLACE', 'STATE'])

# Merge in question info too
df = pd.merge(df, questions, left_on='Question', right_on='ID')

# %%Get means

sums = df[df['Question Type'].isin(['Numerical', 'Binary'])]

# Drop rows with null 'Answer'
sums = sums.dropna(subset=['Answer'])

sums['Answer'] = sums['Answer'].astype(int)

# Compute Mean
sums_mean = sums.groupby('Question')['Answer'].mean()

# Compute Population Weighted Mean
population_weighted_mean = (sums['Answer'] * sums['POPULATION']).groupby(sums['Question']).sum() / \
                           sums.groupby('Question')['POPULATION'].sum()

sums = pd.DataFrame({
    'Mean': sums_mean,
    'Population Weighted Mean': population_weighted_mean
})

# Get sample size
filtered_df = df[(df['Question Type'].isin(['Numerical', 'Binary'])) & df['Answer'].notnull()]
sample_size = filtered_df['Question'].value_counts()
sums['Sample Size'] = sums.index.to_series().apply(
    lambda x: sample_size[int(x) if isinstance(x, str) and x.isnumeric() else x])

# %%Income terciles code

# Total number of unique 'Muni'
total_munis = df['Muni'].nunique()

# Number of unique 'Muni' where 'Median Household Income_2021' is not NaN
valid_munis = df.dropna(subset=['Median Household Income_2021'])['Muni'].nunique()

# Calculate the percentage of valid 'Muni'
percentage_valid = (valid_munis / total_munis) * 100

# Calculate the number of 'Muni' without income data
munis_without_income = total_munis - valid_munis

print(f"Percentage of munis with valid 'Median Household Income_2021': {percentage_valid:.2f}%")
print(f"Number of munis without 'Median Household Income_2021' data: {munis_without_income}")

# Calculate terciles for 'Median Household Income_2021'
income_terciles = df['Median Household Income_2021'].dropna().quantile([1 / 3, 2 / 3])
low_threshold, middle_threshold = income_terciles.values


# Assigning income category based on the thresholds
def assign_income_category(value):
    if value <= low_threshold:
        return 'Low Income'
    elif value <= middle_threshold:
        return 'Middle Income'
    else:
        return 'High Income'


df['Income Category'] = df['Median Household Income_2021'].apply(assign_income_category)


# Assigning categories based on specific thresholds for % Urban
def assign_urban_category(value):
    if value == 0:
        return 'Rural'
    elif 0 < value < 100:
        return 'Suburban'
    elif value == 100:
        return 'Urban'


df['Urban Category'] = df['% Urban'].apply(assign_urban_category)

# Combining the calculations for both Income and Urban categories
categories = ['Low Income', 'Middle Income', 'High Income', 'Rural', 'Suburban', 'Urban']

for category in categories:
    category_col = 'Income Category' if 'Income' in category else 'Urban Category'

    df_filtered = df[(df['Question Type'].isin(['Numerical', 'Binary'])) & (df[category_col] == category)]

    # Compute Mean for each category
    sums_mean_category = df_filtered.groupby('Question')['Answer'].mean()

    # Add to main sums dataframe
    sums[f'Mean ({category})'] = sums_mean_category

# %%Get means for min lot size questions

# Extract the '27mean', '27min' and 'POPULATION' values based on updated criteria
df_27mean = df[(df['Question'] == 27) & (df['Type'] == 'Mean')][
    ['Answer', 'POPULATION', 'Income Category', 'Urban Category']]
df_27min = df[(df['Question'] == 27) & (df['Type'] == 'Min')][
    ['Answer', 'POPULATION', 'Income Category', 'Urban Category']]

print(df_27min['Answer'].value_counts())
# Compute the overall mean for '27mean' and '27min'
overall_mean_of_means = df_27mean['Answer'].mean()
overall_mean_of_mins = df_27min['Answer'].mean()

# Compute the population weighted mean for '27mean' and '27min'
pop_weighted_mean_of_means = (df_27mean['Answer'] * df_27mean['POPULATION']).sum() / df_27mean['POPULATION'].sum()
pop_weighted_mean_of_mins = (df_27min['Answer'] * df_27min['POPULATION']).sum() / df_27min['POPULATION'].sum()

# Compute the sample sizes (number of non-null entries) for each metric
sample_size_mean = df_27mean['Answer'].dropna().shape[0]
sample_size_min = df_27min['Answer'].dropna().shape[0]

sums.loc['Mean of Mean Lot Sizes (Square Feet)', ['Mean', 'Population Weighted Mean', 'Sample Size']] = [
    overall_mean_of_means, pop_weighted_mean_of_means, sample_size_mean]
sums.loc['Mean of Min Lot Sizes (Square Feet)', ['Mean', 'Population Weighted Mean', 'Sample Size']] = [
    overall_mean_of_mins, pop_weighted_mean_of_mins, sample_size_min]

# Now, compute the means for each category (both income and urban) for '27mean' and '27min'
for category in categories:
    category_col = 'Income Category' if 'Income' in category else 'Urban Category'

    df_27mean_filtered = df_27mean[df_27mean[category_col] == category]
    df_27min_filtered = df_27min[df_27min[category_col] == category]

    # Compute Mean for '27mean' for each category
    mean_of_means_category = df_27mean_filtered['Answer'].mean()
    # Compute Mean for '27min' for each category
    mean_of_mins_category = df_27min_filtered['Answer'].mean()

    # Add these values to the sums dataframe in the appropriate columns
    sums.loc['Mean of Mean Lot Sizes (Square Feet)', f'Mean ({category})'] = mean_of_means_category
    sums.loc['Mean of Min Lot Sizes (Square Feet)', f'Mean ({category})'] = mean_of_mins_category


# %%Get the latex formatted table


def adjust_index(x):
    try:
        return questions.set_index('ID')['Question Detail'].to_dict()[int(x)]
    except:
        return x


sums.index = sums.index.to_series().apply(lambda x: adjust_index(x))

# Setting max_colwidth to avoid truncation
pd.set_option('display.max_colwidth', None)

yes_no_indices = questions[questions['Question Type'] == 'Binary']['Question Detail']

# Find the rows that correspond to questions in yes_no_qs
mask = sums.index.isin(yes_no_indices)

# Multiply all columns (excluding 'Sample Size') by 100 for those rows
sums.loc[mask, sums.columns.difference(['Sample Size'])] = sums.loc[
                                                               mask, sums.columns.difference(['Sample Size'])] * 100


# Adjusting custom_format function to handle new column
def custom_format(value, col_name, index):
    if col_name == "Sample Size":
        return str(int(value))
    return '{:.0f}%'.format(value) if index in yes_no_indices and isinstance(value, (int, float)) else str(
        int(round(value)))


# Re-apply custom formatting for each column
for col in sums.columns:
    sums[col] = sums.apply(lambda row: custom_format(row[col], col, row.name), axis=1)

sum_dic = {
    'Continuous': sums.drop(index=yes_no_indices),
    'Binary': sums.loc[yes_no_indices]
}

# Export to Excel
with pd.ExcelWriter(os.path.join(config['tables_path'], 'Table 2 - National Means.xlsx')) as writer:
    for name, data in sum_dic.items():
        data.to_excel(writer, sheet_name=name)

# Generate LaTeX and trim for both
sum_dic['Continuous_latex'] = sum_dic['Continuous'].to_latex().split('midrule')[1].split('\\bottomrule')[0]
sum_dic['Binary_latex'] = sum_dic['Binary'].to_latex().split('midrule')[1].split('\\bottomrule')[0]

# %%Make histograms of continuous random variables

fontsize = 20  # Set the fontsize for all text elements

mean_values = (df_27mean['Answer'] / 1000).dropna().to_list()
min_values = (df_27min['Answer'] / 1000).dropna().to_list()

df_q2 = df[df['Question'] == 2]['Answer'].dropna().astype(int)
df_q22 = df[df['Question'] == 22]['Answer'].dropna().astype(int)

q2_low, q2_high = df_q2.quantile([0.0, 0.99])
q22_low, q22_high = df_q22.quantile([0.0, 0.95])

q2_mean, q2_median = df_q2.mean(), df_q2.median()
q22_mean, q22_median = df_q22.mean(), df_q22.median()

mean_values_95th = np.percentile(mean_values, 95)
mean_of_mean_values = np.mean(mean_values)
median_of_mean_values = np.median(mean_values)

min_values_95th = np.percentile(min_values, 95)
mean_of_min_values = np.mean(min_values)
median_of_min_values = np.median(min_values)


# Function to create histogram
def create_histogram(ax, data, title, mean, median, bins, data_range):
    ax.hist(data, bins=bins, align='left', edgecolor='black', range=data_range)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.text(0.95, 0.95, f'Mean: {mean:.1f}\nMedian: {median:.1f}',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5), fontsize=fontsize)

    # Set tick label size
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(fontsize)


# Original combined figure
fig, ax = plt.subplots(2, 2, figsize=(15, 15), dpi=300)
create_histogram(ax[0][0], df_q2, 'Zoning Districts in Municipality', q2_mean, q2_median, int(q2_high), (0, q2_high))
create_histogram(ax[0][1], df_q22,
                 'What is the longest frontage requirement\nfor single family residential development in any district?',
                 q22_mean, q22_median, 50, (0, q22_high))
create_histogram(ax[1][0], mean_values, 'Mean Lot Sizes (Thousand Feet)', mean_of_mean_values, median_of_mean_values,
                 100, (0, mean_values_95th))
create_histogram(ax[1][1], min_values, 'Min Lot Sizes (Thousand Feet)', mean_of_min_values, median_of_min_values, 100,
                 (0, min_values_95th))

fig.tight_layout()
fig.savefig(os.path.join(figures_path, "combined_histograms.png"), dpi=300)
plt.show()

# Creating standalone figures
for i, (data, title, mean, median, bins, data_range, file_name) in enumerate([
    (
    df_q2, 'How many zoning districts,\nincluding overlays, are in the municipality?', q2_mean, q2_median, int(q2_high),
    (0, q2_high), "q2_histogram.png"),
    (df_q22, 'What is the longest frontage requirement\nfor single family residential development in any district?',
     q22_mean, q22_median, 50, (0, q22_high), "q22_histogram.png"),
    (mean_values, 'Mean Lot Sizes (Thousand Feet)', mean_of_mean_values, median_of_mean_values, 100,
     (0, mean_values_95th), "mean_lot_sizes_histogram.png"),
    (min_values, 'Min Lot Sizes (Thousand Feet)', mean_of_min_values, median_of_min_values, 100, (0, min_values_95th),
     "min_lot_sizes_histogram.png")]):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    create_histogram(ax, data, title, mean, median, bins, data_range)
    fig.savefig(os.path.join(figures_path, file_name), dpi=300)
    plt.show()
    plt.close(fig)

# %%Print match stats for correlation data

# Total number of unique 'Muni'
total_munis = df['Muni'].nunique()

for var in ['Median Home Value (Owner Occupied)_2021', 'Median Gross Rent_2021',
            'Median Home Value (Owner Occupied)_Percent_Change', 'Median Gross Rent_Percent_Change',
            'Single Unit Permits', 'Multi-Unit Permits', 'All Unit Permits']:
    # Number of unique 'Muni' where var is not NaN
    valid_munis = df.dropna(subset=[var])['Muni'].nunique()

    # Calculate the percentage of valid 'Muni'
    percentage_valid = (valid_munis / total_munis) * 100

    # Calculate the number of 'Muni' without income data
    munis_missing = total_munis - valid_munis

    print(f"Percentage of munis with valid {var}: {percentage_valid:.2f}%")
    print(f"Number of munis without {var} data: {munis_missing}")


# %%Now we make cleveland dot plots and appendix correlation table

def question_name(q):
    try:
        return questions.set_index('ID')['Question Detail'].to_dict()[int(q)]
    except:
        if q == '27mean':
            return 'Mean Minimum Lot Size'
        else:
            return 'Minimum Minimum Lot Size'


def calculate_correlation(df, graph=False):
    # Convert 'Answer' to numeric
    df['Answer'] = pd.to_numeric(df['Answer'], errors='coerce')

    # Create per capita variables
    for col in ['Single Unit Permits', 'Multi-Unit Permits', 'All Unit Permits']:
        df[f'{col} per capita'] = df[col] / df['POPULATION']

    # Split Question 27 into '27min' and '27mean'
    mask_min = (df['Question'] == 27) & (df['Type'] == 'Min')
    mask_mean = (df['Question'] == 27) & (df['Type'] == 'Mean')
    df.loc[mask_min, 'Question'] = '27min'
    df.loc[mask_mean, 'Question'] = '27mean'

    # Winsorize for questions in 'numerical' and the new '27min' and '27mean' questions
    for num in questions[questions['Question Type'] == 'Numerical']['ID'].tolist() + ['27min', '27mean']:
        mask = (df['Question'] == num) & (df['Answer'].notna())
        non_null_answers = df.loc[mask, 'Answer']
        df.loc[mask, 'Answer'] = mstats.winsorize(non_null_answers, limits=[0.05, 0.05])
        print(len(non_null_answers))
    # List of columns to calculate correlation for
    cols_to_corr = [
        'Median Gross Rent_2021',
        'Median Gross Rent_Percent_Change',
        'Median Home Value (Owner Occupied)_2021',
        'Median Home Value (Owner Occupied)_Percent_Change',
        'Single Unit Permits per capita',
        'Multi-Unit Permits per capita',
        'All Unit Permits per capita'
    ]

    correlations = {}

    # Calculate correlations and save scatter plot data for each column in nums list
    for col in cols_to_corr:
        correlations[col] = {}
        for num in questions[questions['Question Type'].isin(['Binary', 'Numerical'])]['ID'].tolist() + ['27min',
                                                                                                         '27mean']:
            relevant_data = df[df['Question'] == num].dropna(subset=['Answer', col])
            correlations[col][num] = relevant_data['Answer'].corr(relevant_data[col])

    return pd.DataFrame(correlations)


# Calculate correlations for the sample data
correlations = calculate_correlation(df.copy())

# Update index
correlations.index = correlations.index.to_series().apply(lambda x: question_name(x))

# Set float format
pd.options.display.float_format = '{:,.2f}'.format

# Convert to latext and split questions


corr_dic = {
    'Continuous': correlations.drop(index=yes_no_indices),
    'Binary': correlations.loc[yes_no_indices]
}

# Export to Excel
with pd.ExcelWriter(os.path.join(config['tables_path'], 'Table A2 - National Correlations.xlsx')) as writer:
    for name, data in corr_dic.items():
        data.to_excel(writer, sheet_name=name)

# Generate LaTeX and trim for both
corr_dic['Continuous_latex'] = corr_dic['Continuous'].to_latex().split('midrule')[1].split('\\bottomrule')[0]
corr_dic['Binary_latex'] = corr_dic['Binary'].to_latex().split('midrule')[1].split('\\bottomrule')[0]

# %%

rename_map = {
    'Median Gross Rent_2021': 'Median Gross Rent\n(2021)',
    'Median Gross Rent_Percent_Change': 'Median Gross Rent\n% Change, 2021-2010',
    'Median Home Value (Owner Occupied)_2021': 'Median Home Value\n2021',
    'Median Home Value (Owner Occupied)_Percent_Change': 'Median Home Value\n% Change, 2021-2010',
    'Single Unit Permits per capita': 'Building Permits\nSingle Unit',
    'Multi-Unit Permits per capita': 'Building Permits\nMulti-Unit',
    'All Unit Permits per capita': 'Building Permits\nAll Units'
}


def plot_correlations(correlations, sort_by_column=None, save_filename=None):
    correlations = correlations.rename(columns=rename_map)

    if sort_by_column:
        sort_by_column = rename_map[sort_by_column]
        # Reorder columns to put sort_by_column first
        columns_order = [sort_by_column] + [col for col in correlations.columns if col != sort_by_column]
        correlations = correlations[columns_order]

    fontsize = 18

    # Melt the dataframe to long format for easier plotting
    df_melted = correlations.reset_index().melt(id_vars='index', value_vars=correlations.columns)
    df_melted.columns = ['Question', 'Metric', 'Correlation']

    if sort_by_column:
        # Filter rows that match the sorting column and sort them
        sort_data = df_melted[df_melted['Metric'] == sort_by_column].sort_values('Correlation', ascending=False)
        sort_order = sort_data['Question'].tolist()
    else:
        # If not sorting by a column, sort by the index
        sort_order = df_melted['Question'].drop_duplicates().tolist()

    # Define a color palette
    palette = sns.color_palette(["#303F9F", "#D2691E", "#BDC3C7", "#A2D149"])

    # Create the Cleveland dot plot with increased height for better visualization
    plt.figure(figsize=(20, len(sort_order) * 1.2), dpi=100)  # Adjust height based on number of items
    ax = sns.stripplot(data=df_melted, x='Correlation', y='Question', hue='Metric', jitter=False, size=20, alpha=0.5,
                       order=sort_order, palette=palette)

    plt.axvline(0, color='grey', linestyle='--')  # Adds a reference line for zero correlation

    # Wrap y-axis labels and adjust font size
    ax.set_yticklabels(ax.get_yticklabels(), wrap=True, fontsize=fontsize)  # Adjust fontsize as needed

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)

    ax.set_xlabel('Correlation', fontsize=fontsize)

    ax.set_ylabel('', fontsize=fontsize)

    plt.legend(loc='lower right', ncol=1, fontsize=fontsize, edgecolor='none')

    # Adjust layout to make room for the legend below and increase space for y-labels
    plt.tight_layout(rect=[0.35, 0.2, 1, 1])  # Adjust the left margin

    # Save the plot to file if save_filename is provided
    if save_filename:
        save_path = os.path.join(figures_path, save_filename)
        plt.savefig(save_path, dpi=300)

    plt.show()


plot_correlations(correlations.copy()[['Median Gross Rent_2021', 'Median Home Value (Owner Occupied)_2021',
                                       'All Unit Permits per capita']],
                  sort_by_column='Median Home Value (Owner Occupied)_2021',
                  save_filename='Cleveland Dot1.png')

plot_correlations(correlations.copy().drop(
    columns=['Median Gross Rent_2021', 'Median Home Value (Owner Occupied)_2021', 'All Unit Permits per capita']),
                  sort_by_column='Median Home Value (Owner Occupied)_Percent_Change',
                  save_filename='Cleveland Dot2.png')


