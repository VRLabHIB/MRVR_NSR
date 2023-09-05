import pandas as pd
import numpy as np
from pathlib import Path
import os

if __name__ == '__main__':
    val_path = os.path.abspath(os.getcwd())
    project_path = os.path.abspath(Path(val_path).parent)
    data_path = project_path + '\\data\\1_questionnaire_preprocessed\\'

    feature_path = project_path + '\\data\\6_feature_dataset\\'

    df_f = pd.read_csv(feature_path + '2023-06-17_eye_features.csv')

    IDs = df_f['ID'].unique()



    df = pd.read_csv(data_path + '2023-06-12_preprocessed_questionnaire.csv')

    df = df[df['ID'].isin(IDs)]

    print(df['gender'].value_counts())
    print(df['visual_aid'].value_counts())
    print(df['age'].describe())
    print('')