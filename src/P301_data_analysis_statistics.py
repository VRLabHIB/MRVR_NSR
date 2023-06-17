import pandas as pd
import numpy as np

import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats import multitest


class Statistics:
    def __init__(self, data_path):
        df = pd.read_csv(data_path + '\\6_feature_dataset\\2023-06-17_eye_features.csv')

        df = df[~df['stimulus'].isin([21,24])]
        df = df.drop(columns=['condition', 'stimulus', 'Response'])
        df_n = pd.pivot_table(df.iloc[:,:-3], index = ['ID','dimension'], aggfunc = 'mean')
        df_n = df_n.reset_index()

        df2  = df_n[df_n['dimension'] == 2]
        df3 = df_n[df_n['dimension'] == 3]

        dfw = df2.merge(df3, how = 'inner', on='ID')
        dfw =dfw.drop(columns=['dimension_x', 'dimension_y'])


        df_stats = pd.DataFrame({'feature':[], 'M 2D':[], 'SD 2D':[], 'M 3D':[], 'SD 3D':[], 't':[], 'p-value':[]})
        for col in df.iloc[:,2:-3].columns:
            m_2d = np.round(np.mean(dfw['{}_x'.format(col)].values), 3)
            m_3d = np.round(np.mean(dfw['{}_y'.format(col)].values), 3)
            s_2d = np.round(np.std(dfw['{}_x'.format(col)].values), 3)
            s_3d = np.round(np.std(dfw['{}_y'.format(col)].values), 3)

            ttest = scipy.stats.ttest_rel(dfw['{}_x'.format(col)].values, dfw['{}_y'.format(col)].values)
            t  = np.round(ttest[0], 3)
            p = ttest[1]

            df_stats.loc[len(df_stats)] = [col, m_2d, s_2d, m_3d, s_3d, t, p]

        p_values = df_stats['p-value'].iloc[2:]

        corr_p = multitest.multipletests(p_values, alpha=0.05, method='hs', maxiter=1, is_sorted=False,
                                                  returnsorted=False)

        x = corr_p[1]
        df_stats['p-value'].iloc[2:] = corr_p[1]

        table = df_stats.to_latex(index=False,

              formatters={"name": str.upper},

              float_format="{:.3f}".format)

        table =table.replace('0.000', '<0.001')

        old = ['Correct', 'RT', 'Mean fixation duration', 'Mean regressive fixation duration']
        new = ['Correct (ratio)', 'Reaction time (sec)', 'Mean fixation duration (ms)', ]
        print(table)

        print(scipy.stats.ttest_rel(dfw['RT_x'].values, dfw['3DRT'].values))
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