import os

import numpy as np
import pandas as pd

import src.P001_preprocessing_experiment_data as P001
import src.P002_preprocessing_questionnaire_data as P002

if __name__ == '__main__':
    # set project and data path
    project_path = os.path.abspath(os.getcwd())
    data_path = project_path + "\\data\\"
    os.chdir(data_path)

    save_path = project_path + "\\results\\"

    ### Preprocess questionnaire ###
    #prep_quest = P002.Preprocessing(data_path)
    #prep_quest.refactor_and_select_columns()
    #prep_quest.save_dataframe()

    ### Preprocess experiment data ###
    prep_exp = P001.Preprocessing(data_path, location = '\\0_raw_experiment\\')
    #prep_exp.data_cleaning()
    #dfs = prep_exp.get_dataframes()

    # First Cleaning
    save_path_step0 = data_path + '\\1_first_cleaning_experiment\\'
    #prep_exp.save_dataframes(save_path_step0)
    print('Hello')


    prep_exp = P001.Preprocessing(data_path, location='\\1_first_cleaning_experiment\\')
    dfs = prep_exp.get_dataframes()
