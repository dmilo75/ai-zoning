import pandas as pd
import numpy as np
import os
import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('../../')
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

os.environ['MPLCONFIGDIR'] = config['raw_data']
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt

model = 'latest_combined'

#Draw in the index too
index = pd.read_excel(os.path.join(config['processed_data'], 'Model Output', model, 'Overall_Index.xlsx'))

# Draw in excel file of model output
df = pd.read_excel(os.path.join(config['processed_data'], 'Model Output', model, 'Light Data.xlsx'))

# %%Make histograms of continuous random variables

fontsize = 20  # Set the fontsize for all text elements

mean_values = df[df['Question'] == '28Mean']['Answer'].dropna().astype(int)/1000
min_values = df[df['Question'] == '28Min']['Answer'].dropna().astype(int)/1000

df_q2 = df[df['Question'] == '2']['Answer'].dropna().astype(int)
df_q22 = df[df['Question'] == '22']['Answer'].dropna().astype(int)

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
def create_histogram(ax, data, title, bins, data_range):

    #Calculate mean and median
    mean = np.mean(data)
    median = np.median(data)

    #Check if range is None, None
    if data_range == (None,None):
        ax.hist(data, bins=bins, align='left', edgecolor='black')
    else:
        ax.hist(data, bins=bins, align='left', edgecolor='black', range=data_range)

    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.text(0.95, 0.95, f'Mean: {mean:.1f}\nMedian: {median:.1f}',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5), fontsize=fontsize)

    # Set tick label size
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(fontsize)

#Get min and 99th percentile for second principal component
second_pc_low, second_pc_high = index['Second_Principal_Component'].quantile([0.0, 0.99])

# Original combined figure
fig, ax = plt.subplots(3, 2, figsize=(15, 15), dpi=300)

create_histogram(ax[0][0], index['First_Principal_Component'], 'Zoning Index (First Principal Component)',
                 5, (None, None))

create_histogram(ax[0][1], index['Second_Principal_Component'], 'Zoning Index (Second Principal Component)',
                 5, (second_pc_low, second_pc_high))

create_histogram(ax[1][0], df_q2, 'The Number of Zoning Districts in Each Municipality',
                 int(q2_high), (0, q2_high))
create_histogram(ax[1][1], df_q22,
                 'What is the longest frontage requirement\nfor single family residential development\nin any district? (Thousand Feet)',
                 50, (0, q22_high))
create_histogram(ax[2][0], mean_values, 'Mean Min Lot Size (Thousand Feet)',
                 100, (0, mean_values_95th))
create_histogram(ax[2][1], min_values, 'Minimum Min Lot Size (Thousand Feet)',
                 100, (0, min_values_95th))

fig.tight_layout()
fig.savefig(os.path.join(config['figures_path'], "Figure 6 - Histograms.png"), dpi=300)
plt.show()