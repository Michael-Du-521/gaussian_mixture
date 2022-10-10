import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import gaussian
import random
from math import log, sqrt

class GaussianMixture:
    "Model mixes two uni-variate Gaussian Model"
    def __init__(self, data, sigma_min=1, sigma_max=1, mix=0.5):
        self.data = data
        self.mu_min = data.min()
        self.mu_max = data.max()
        # initiate with multiple gaussians
        self.one = gaussian.Gaussian(mu=random.uniform(self.mu_min, self.mu_max), sigma=random.uniform(sigma_min, sigma_max))
        self.two = gaussian.Gaussian(mu=random.uniform(self.mu_min, self.mu_max), sigma=random.uniform(sigma_min, sigma_max))
        self.mix = mix
        self.loglike = 0.

    def Estep(self):
        for datum in self.data:
            wp1 = self.one.pdf(datum) * self.mix
            wp2 = self.two.pdf(datum) * (1 - self.mix)
            den = wp1 + wp2
            wp1 /= den
            wp2 /= den
            self.loglike += log(wp1 + wp2)
            # yield weight tuple
            yield (wp1, wp2)


    def Mstep(self, weights):
        # unzip weights by using preceding*
        (left, right) = zip(*weights)
        one_den = sum(left)
        two_den = sum(right)
        self.one.mu = sum(w * d / one_den for (w, d) in zip(left, self.data))
        self.two.mu = sum(w * d / two_den for (w, d) in zip(right, self.data))
        self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2) for (w, d) in zip(left, self.data)) / one_den)
        self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2) for (w, d) in zip(right, self.data)) / two_den)
        self.mix = one_den / len(self.data)


    def iterate(self, N=1, verbose=False):
        "Perform N iterations, then compute log-likelihood"
        self.Mstep(self.Estep())
        print(self.__repr__())

    def pdf(self, x):
            return (self.mix) * self.one.pdf(x) + (1 - self.mix) * self.two.pdf(x)

    def __repr__(self):
            return 'GaussianMixture:{0},{1},mix={2:.03}'.format(self.one, self.two, self.mix)

# Read csv file, and convert it to DataFrame
df = pd.read_csv("C:\\Users\\13104\\Desktop\\D_W_X_Chopin_op10_no3_p01-p22 - 40 beats per column - 0922.csv",header=None)

# Standardize the axis name as well as the rows and colum number of the DataFrame
df = df.set_axis([i for i in range(1, 41)], axis='index')
df = df.set_axis([i for i in range(1, 23)], axis='columns')
df = df.rename_axis('Beat Index', axis='index')
df = df.rename_axis('Pianist', axis='columns')

# the list to store gaussian component
gaussian_components={'gaussian_component1':[],'gaussian_component2':[]}
gaussian_mixture_both=[]

# create empty list for storing each audio data
df_1_beat_interval_revise_smoothing_zscore_list = []
for i in range(1, 23):
    print('The ' +str(i)+' '+'column '+'of the list: \n')
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

    #run GMM
    data = df_1_beat_interval_revise_smoothing_zscore_list[i-1]

    #GMM fitting process
    n_iteration = 100
    best_mix = None
    best_loglike = float('-inf') # -INFINITY
    mix = GaussianMixture(data=data)
    for _ in range(n_iteration):
        print('------')
        print(str(_+1)+'th iteration: ')
        try:
            mix.iterate()
            if mix.loglike > best_loglike:
                best_loglike = mix.loglike
                best_mix = mix
            if _==(n_iteration-1):
                gaussian_components['gaussian_component1'].append(best_mix.one)
                gaussian_components['gaussian_component2'].append(best_mix.two)
                gaussian_mixture_both.append(best_mix)
        except(ZeroDivisionError, ValueError, RuntimeError):
            pass
    print('------')
    print('\n')

#draw all the guassian mixture for each histogram
rows, cols = 5, 5
fig = plt.figure(figsize=(10, 10))
fig.suptitle('Histograms')
for i in range(1, 23):
    data =df_1_beat_interval_revise_smoothing_zscore_list[i - 1]
    # fit a single gaussian curve to the data
    x = np.linspace(data.min()-5, data.max()+5, 100)
    ax = fig.add_subplot(rows, cols, i)
    ax.set_title(i)
    ax = sns.histplot(data, bins=20, kde=False, stat="count", linewidth=0)
    plt.yticks([i for i in range(0,7)])
    plt.xticks([i for i in range(-3,4,1)])
    # ax = sns.histplot(data, bins=20, kde=False, stat="density", linewidth=0)
    g_1=[gaussian_components['gaussian_component1'][i-1].pdf(e) for e in x]
    g_2 = [gaussian_components['gaussian_component2'][i - 1].pdf(e) for e in x]
    g_both = [gaussian_mixture_both[i - 1].pdf(e) for e in x]
    # plt.plot(x, g_1, label='gaussian_component1',color='red')
    # plt.plot(x, g_2, label='gaussian_component2',color='blue')
    # plt.plot(x, g_both, label='gaussian mixture',color='black')
    plt.xlabel('zscore of beat interval')
    # plt.legend(fontsize=4)
plt.tight_layout()
plt.show()






