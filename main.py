import os

import numpy as np
import pandas as pd

from datetime import date

import src.P101_experiment_preprocessing_data as P001
import src.P201_questionnaire_preprocessing_data as P002

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
    ####################################################################################################################
    ## First Cleaning

    #prep_exp = P001.Preprocessing(data_path, location = '\\0_experiment_raw\\')
    #prep_exp.data_cleaning()
    #prep_exp.add_experiment_condition()

    #save_path_step0 = data_path + '\\1_experiment_first_cleaning\\'
    #prep_exp.save_dataframes(save_path_step0)

    ####################################################################################################################
    ## Process pupil and select valid cases

    pupil_exp = P001.Preprocessing(data_path, location='\\1_experiment_first_cleaning\\')
    pupil_exp.process_pupil()

    df_tr = pupil_exp.get_tr_data()
    today = date.today()
    df_tr.to_csv(data_path + '\\meta\\{}_tracking_ratio.csv'.format(today), index=False)

    # Remove invalid cases
    valid_path = data_path + '\\2_experiment_valid_cases\\'
    invalid_path = data_path + '\\2_experiment_removed_cases\\'
    pupil_exp.select_cases_on_tracking_ratio(valid_path, invalid_path)

    ####################################################################################################################
    ## Add necessary variables
    add_var = P001.Preprocessing(data_path, location='\\2_experiment_valid_cases\\')



    print('Hello')


