import os

import numpy as np
import pandas as pd

from datetime import date

import src.P101_experiment_preprocessing_data as P001
import src.P201_questionnaire_preprocessing_data as P002

import src.P301_eye_feature_calculation as P301

import scipy
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # set project and data path
    project_path = os.path.abspath(os.getcwd())
    data_path = project_path + "\\data\\"
    os.chdir(data_path)

    save_path = project_path + "\\results\\"

    #####################################
    #### Uncomment to run the pipeline!!!
    #####################################

    ### Preprocess questionnaire ###
    #prep_quest = P002.Preprocessing(data_path)
    #prep_quest.refactor_and_select_columns()
    #prep_quest.save_dataframe()

    ### Preprocess experiment data ###
    ####################################################################################################################
    ## First Cleaning

    #prep_exp = P001.Preprocessing(data_path, location = '\\0_experiment_raw\\')
    #prep_exp.data_cleaning()
    #prep_exp.add_experiment_condition()

    #save_path_step1 = data_path + '\\1_experiment_first_cleaning\\'
    #prep_exp.save_dataframes(save_path_step0)

    ####################################################################################################################
    ## Process pupil and select valid cases

    #pupil_exp = P001.Preprocessing(data_path, location='\\1_experiment_first_cleaning\\')
    #pupil_exp.process_pupil()

    #df_tr = pupil_exp.get_tr_data()
    #today = date.today()
    #df_tr.to_csv(data_path + '\\meta\\{}_tracking_ratio.csv'.format(today), index=False)

    ## Remove invalid cases
    #valid_path = data_path + '\\2_experiment_valid_cases\\'
    #invalid_path = data_path + '\\2_experiment_removed_cases\\'
    #pupil_exp.select_cases_on_tracking_ratio(valid_path, invalid_path)

    ####################################################################################################################
    ## Add necessary variables
    #add_var = P001.Preprocessing(data_path, location='\\2_experiment_valid_cases\\')
    #add_var.calculate_and_process_variables()

    #save_path_step3 = data_path + '\\3_experiment_preprocessed\\'
    #add_var.save_dataframes(save_path_step3)



    #######
    # Test
    '''
    features = P301.Features(data_path, location='\\3_experiment_preprocessed\\')
    features.create_dataframe()
    df_f = features.get_feature_dataset()

    df_a = pd.DataFrame({'ID':[], '2DRT':[],'3DRT':[], '2DCorrect':[], '3DCorrect':[], 'first':[]})

    for id in df_f['ID'].unique():
        df_sub = df_f[df_f['ID']==id]
        RT2D = np.mean(df_sub['2DRT'])
        RT3D = np.mean(df_sub['3DRT'])
        Correct2D = np.mean(df_sub['2DCorrect'])
        Correct3D = np.mean(df_sub['3DCorrect'])
        first = df_sub['first'].iloc[0]

        df_a.loc[len(df_a.index)] = [id, RT2D, RT3D, Correct2D, Correct3D, first]

    df_a.to_csv(data_path + '\\5_feature_dataset\\features.csv',index = False)
    '''

    df_a = pd.read_csv(data_path + '\\5_feature_dataset\\features.csv')

    df1 = df_a[['ID', '2DRT', 'first']].copy()
    df1['dimension'] = ['2D']*len(df1)
    #df1['3D'] = np.zeros(len(df1))
    df1['condition'] = np.where(df1['first']=='2D','first', 'second')
    df1 = df1.rename(columns={'2DRT':'RT'})
    print(df1['condition'].value_counts())

    df2 = df_a[['ID', '3DRT', 'first']].copy()
    df2['dimension'] = ['3D']*len(df2)
    df2['condition'] = np.where(df2['first'] == '3D', 'first', 'second')
    df2 = df2.rename(columns={'3DRT':'RT'})

    df_b = pd.concat([df1,df2])
    df_b['ID'] = df_b['ID'].astype(str)
    #df_b.drop(columns=['first'])

    print('2D RT')
    print('2D first')
    print(df_a[df_a['first'] == '2D']['2DRT'].describe())
    print('2D second')
    print(df_a[df_a['first'] == '3D']['2DRT'].describe())

    print('3D RT')
    print('3D first')
    print(df_a[df_a['first'] == '3D']['3DRT'].describe())
    print('3D second')
    print(df_a[df_a['first'] == '2D']['3DRT'].describe())

    df_b.to_csv(data_path + '\\5_feature_dataset\\ANOVA.csv')

    from datamatrix import io
    from statsmodels.stats.anova import AnovaRM

    dm = io.readtxt(data_path + '\\5_feature_dataset\\ANOVA.csv')
    aov = AnovaRM(
        dm,
        depvar='RT',
        subject='ID',
        within=['dimension','condition'],
        aggregate_func = 'mean'
    ).fit()
    print(aov)


    print('2D')
    print(df_a['2DRT'].describe())
    print('2D Median ', np.median(df_a['2DRT']))
    print('3D')
    print('3D Median ', np.median(df_a['3DRT']))
    print(df_a['3DRT'].describe())
    print(scipy.stats.ttest_ind(df_a['2DRT'].values, df_a['3DRT'].values))
    print(scipy.stats.wilcoxon(df_a['2DRT'].values,df_a['3DRT'].values))

    Diff = df_a['2DRT']-df_a['3DRT']
    from scipy.stats import sem

    mean1 = np.mean(df_a['2DRT'])
    mean2 = np.mean(df_a['3DRT'])
    d = (mean1-mean2)/np.std(Diff)

    print(sem(Diff))
    print(np.std(Diff))
    print('Cohens d ',d)

    print('2D')
    print(df_a['2DCorrect'].describe())
    print('3D')
    print(df_a['3DCorrect'].describe())
    print(scipy.stats.ttest_rel(df_a['2DCorrect'].values,df_a['3DCorrect'].values ))
    print(scipy.stats.wilcoxon(df_a['2DCorrect'].values,df_a['3DCorrect'].values))

    df_RT = df_a[['2DRT', '3DRT']]
    df_RT = df_RT.rename(columns={'2DRT':'2D', '3DRT':'3D'})
    ax = sns.barplot(data=df_RT)
    plt.title('Average reaction time (RT) for both conditions')
    ax.set(xlabel='Condition', ylabel='RT')

    for i in ax.containers:
        ax.bar_label(i,fmt='%.3f' )

    plt.show()

    df_RT = df_a[['2DCorrect', '3DCorrect']]
    df_RT = df_RT.rename(columns={'2DCorrect':'2D', '3DCorrect':'3D'})
    ax = sns.barplot(data=df_RT)
    plt.title('Average Correct Answer (CA) for both conditions')
    ax.set(xlabel='Condition', ylabel='CA')

    for i in ax.containers:
        ax.bar_label(i,fmt='%.3f' )

    plt.show()


    print('Hello')


