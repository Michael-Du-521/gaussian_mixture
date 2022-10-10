import statistics
import numpy as np
import pandas as pd
from math import sqrt, pi, exp
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns


class Gaussian:
    "Model Univariate Gaussian"

    def __init__(self, mu, sigma):
        # "Mean and standard deviation"
        self.mu = mu
        self.sigma = sigma

    def pdf(self, datum):
        "Probability of a data point given the current parameters"
        u = (datum - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y

    def __repr__(self):
        return f'Gaussian(mu={self.mu}, sigma={self.sigma})'

# Read csv file, and convert it to DataFrame
df = pd.read_csv("C:\\Users\\13104\\Desktop\\D_W_X_Chopin_op10_no3_p01-p22 - 40 beats per column - 0922.csv",
                 header=None)
# Standardize the axis name as well as the rows and colum number of the DataFrame
df = df.set_axis([i for i in range(1, 41)], axis='index')
df = df.set_axis([i for i in range(1, 23)], axis='columns')
df = df.rename_axis('Beat Index', axis='index')
df = df.rename_axis('Pianist', axis='columns')

# create empty list for storing each audio data
df_1_beat_interval_revise_smoothing_zscore_list = []
for i in range(1, 23):
    # use for loop to go through all the column in my audio dataset
    df_1 = df.loc[:, i]
    df_1_beat_interval = df_1.diff()
    # since after differencing, each column only had 39 remained rows
    df_1_beat_interval_revise = df_1_beat_interval.iloc[1:40]
    df_1_beat_interval_revise = df_1_beat_interval_revise.set_axis([i for i in range(1, 40)], axis='index')

    # the smoothing window size is 3
    window_size = 3
    df_1_beat_interval_revise_smoothing = df_1_beat_interval_revise.rolling(window_size).mean()
    # the first smoothing value is calculated by equaling to corresponding the first beat value of original frame
    df_1_beat_interval_revise_smoothing[1] = df_1_beat_interval_revise[1]
    # rhe second smoothing value is calculated by averaging the first two corresponding beats value of original frame
    df_1_beat_interval_revise_smoothing[2] = statistics.mean(df_1_beat_interval_revise[0:2])
    # tempo standardization
    # 1 z-score regulation for origal tempo after smoothing
    df_1_beat_interval_revise_smoothing_zscore = (df_1_beat_interval_revise_smoothing - df_1_beat_interval_revise_smoothing.mean()) / ( df_1_beat_interval_revise_smoothing.std())
    # add each zscore result of each audios into the list for storing
    df_1_beat_interval_revise_smoothing_zscore_list.append(df_1_beat_interval_revise_smoothing_zscore)


#draw all the best-fit guassian for each histogram
# rows, cols = 5, 5
# fig = plt.figure(figsize=(10, 10))
# fig.suptitle('Best fit gaussian for each histogram')
# for i in range(1, 23):
#     data = pd.DataFrame(df_1_beat_interval_revise_smoothing_zscore_list[i - 1])
#     best_single = Gaussian(np.mean(data), np.std(data))
#     # fit a single gaussian curve to the data
#     x = np.linspace(data.min() - 1, data.max() + 1, 100)
#     g_single = stats.norm(best_single.mu, best_single.sigma).pdf(x)
#     ax = fig.add_subplot(rows, cols, i)
#     ax.set_title(i)
#     ax = sns.distplot(data, bins=20, kde=False, norm_hist=True)
#     ax = plt.plot(x, g_single, label='single_gaussian')
#     plt.xlabel('zscore of beat interval')
# plt.tight_layout()
# plt.show()
