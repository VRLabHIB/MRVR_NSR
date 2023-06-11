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

    prep_quest = P002.Preprocessing(data_path)
    df_q = prep_quest.get_dataframe()

    prep_exp = P001.Preprocessing(data_path)
    #prep_exp.add_experiment_condition()
    #prep_exp.set_stimulus_and_timing()
    #prep_exp.drop_unused_variables()

    #data_lst = prep_exp.get_data_lst()
    #dataframes = prep_exp.get_dataframes()

    #for i in range(len(data_lst)):
    #    name = data_lst[i]
    #    df = dataframes[i]
    #    df.to_csv(name, index=False)