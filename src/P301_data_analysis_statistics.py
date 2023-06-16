import pandas as pd
import numpy as np

import scipy
import seaborn as sns
import matplotlib.pyplot as plt

class Statistics:
    def __init__(self, data_path):
        df = pd.read_csv(data_path + '\\6_feature_dataset\\2023-06-15_eye_features.csv')

        df = df.drop(columns=['condition', 'stimulus', 'Response'])
        df_n = pd.pivot_table(df.iloc[:,:-3], index = ['ID','dimension'], aggfunc = 'mean')
        df_n['index'] = df_n.index

        df_n  = df_n['index'].str.split(pat='/')
        dfw = pd.pivot(df_n, index='ID', columns='dimension', values=['RT', 'Correct'])

        dfw.columns = ['2DRT', '3DRT','2DCorrect', '3DCorrect' ]

        dfa = pd.pivot(df_n, index='ID', columns='dimension')

        print(scipy.stats.ttest_rel(dfw['2DRT'].values, dfw['3DRT'].values))
        print(scipy.stats.wilcoxon(dfw['2DRT'].values, dfw['3DRT'].values))

        Diff = dfw['2DRT'] - dfw['3DRT']

        mean1 = np.mean(dfw['2DRT'])
        mean2 = np.mean(dfw['3DRT'])
        d = (mean1 - mean2) / np.std(Diff)

        from scipy.stats import sem
        print(sem(Diff))
        print(np.std(Diff))
        print('Cohens d ', d)

        print('2D')
        print(dfw['2DCorrect'].describe())
        print('3D')
        print(dfw['3DCorrect'].describe())
        print(scipy.stats.ttest_rel(dfw['2DCorrect'].values, dfw['3DCorrect'].values))
        print(scipy.stats.wilcoxon(dfw['2DCorrect'].values, dfw['3DCorrect'].values))

        df_RT = dfw[['2DRT', '3DRT']]
        df_RT = df_RT.rename(columns={'2DRT': '2D', '3DRT': '3D'})
        ax = sns.barplot(data=df_RT)
        plt.title('Average reaction time (RT) for both conditions')
        ax.set(xlabel='Condition', ylabel='RT')

        for i in ax.containers:
            ax.bar_label(i, fmt='%.3f')

        plt.show()

        df_RT = dfw[['2DCorrect', '3DCorrect']]
        df_RT = df_RT.rename(columns={'2DCorrect': '2D', '3DCorrect': '3D'})
        ax = sns.barplot(data=df_RT)
        plt.title('Average Correct Answer (CA) for both conditions')
        ax.set(xlabel='Condition', ylabel='CA')

        for i in ax.containers:
            ax.bar_label(i, fmt='%.3f')

        plt.show()