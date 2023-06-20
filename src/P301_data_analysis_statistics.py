import pandas as pd
import numpy as np

import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats import multitest

from scipy.stats import shapiro, normaltest
from scipy.stats import wilcoxon

class Statistics:
    def __init__(self, data_path):
        df = pd.read_csv(data_path + '\\6_feature_dataset\\2023-06-17_eye_features.csv')

        df_cond = df[np.logical_and(df['dimension']==2, df['stimulus']==15)]
        print('2D condition first')
        print(df_cond['condition'].value_counts())

        df_stim = df[~df['stimulus'].isin([21, 24])]
        df_stim = df_stim[np.logical_and(df_stim['dimension'] == 2, df_stim['ID'] == 100)]
        df_stim = df_stim.iloc[:,3:]

        df_stim = pd.pivot_table(df_stim, index=[ 'stimulus'],aggfunc='first')
        print(df_stim['AngularDisp'].value_counts())
        print(df_stim['DiffType'].value_counts())



        df = df[~df['stimulus'].isin([21,24])]
        df = df.drop(columns=['condition', 'stimulus', 'Response'])
        df_n = pd.pivot_table(df.iloc[:,:-3], index = ['ID','dimension'], aggfunc = 'mean')
        df_n = df_n.reset_index()

        df_p = pd.read_csv(data_path + '\\1_questionnaire_preprocessed\\2023-06-12_preprocessed_questionnaire.csv')
        df_p = df_p[['ID', 'gender']]

        df_n = df_n.merge(df_p, how='left', on='ID')

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

            print(col)
            Diff = dfw['{}_x'.format(col)].values - dfw['{}_y'.format(col)].values
            stat, p = normaltest(Diff)
            if p>0.05:
                print('Gaussian')
            if p<=0.05:
                print('Not Gaussian')
                print(p)

            stat, p = wilcoxon(dfw['{}_x'.format(col)].values, dfw['{}_y'.format(col)].values)
            print('Statistics=%.3f, p=%.3f' % (stat, p))

            ttest = scipy.stats.ttest_rel(dfw['{}_x'.format(col)].values, dfw['{}_y'.format(col)].values)
            #print(ttest)
            #t  = np.round(ttest[0], 3)
            #p = ttest[1]
            t = stat

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

        dfx = df.drop(columns=['dimension'])
        df_a = pd.pivot_table(dfx.iloc[:, :-3], index=['ID'], aggfunc='mean')
        df_a = df_a.merge(df_p, how='left', on='ID')

        df_male = df_a[df_a['gender']=='male']
        df_female = df_a[df_a['gender']=='female']

        print('RT total gender')
        print(df_a.groupby('gender')['RT'].describe())
        print(scipy.stats.ttest_ind(df_male['RT'].values, df_female['RT'].values))

        print('Correct total gender')
        print(df_a.groupby('gender')['Correct'].describe())
        print(scipy.stats.ttest_ind(df_male['Correct'].values, df_female['Correct'].values))


        print('RT 2D')
        print(df2.groupby('gender')['RT'].describe())
        df_male2 = df2[df2['gender'] == 'male']
        df_female2 = df2[df2['gender'] == 'female']
        print('Correct 2D')
        print(df2.groupby('gender')['Correct'].describe())
        print('ttest RT 2D')
        print(scipy.stats.ttest_ind(df_male2['RT'].values, df_female2['RT'].values))
        print('ttest Correct 2D')
        print(scipy.stats.ttest_ind(df_male2['Correct'].values, df_female2['Correct'].values))

        print('RT 3D')
        print(df3.groupby('gender')['RT'].describe())
        df_male3 = df3[df3['gender'] == 'male']
        df_female3 = df3[df3['gender'] == 'female']
        print('Correct 3D')
        print(df3.groupby('gender')['Correct'].describe())
        print('ttest RT 3D')
        print(scipy.stats.ttest_ind(df_male3['RT'].values, df_female3['RT'].values))
        print('ttest Correct 3D')
        print(scipy.stats.ttest_ind(df_male3['Correct'].values, df_female3['Correct'].values))




        #######
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