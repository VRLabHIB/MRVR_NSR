import os
from datetime import datetime
import pandas as pd

# import warnings
# warnings.filterwarnings('ignore')

import numpy as np
import glob

class FixationDetection:
    def __init__(self, data_path, location):
        os.chdir(data_path + location)
        self.data_lst = glob.glob("ID*.csv")
        print("Number of files: ", len(self.data_lst))
        self.dataframes = [pd.read_csv(f, sep=',', header=0, index_col=False, low_memory=False) for f in self.data_lst]

    def get_dataframes(self):
        return self.dataframes

    def get_data_lst(self):
        return self.data_lst

    def save_dataframes(self, save_path):
        for i in range(0, len(self.dataframes)):
            name = self.data_lst[i]
            df = self.dataframes[i]
            df.to_csv(save_path + name, index=False)
        print('Files saved to: ', save_path)


    def run_IVT(self):
        for i in range(0, len(self.dataframes)):
            name = self.data_lst[i]
            df = self.dataframes[i]

            df = run_IVT(df)


def run_IVT(df, velo_threshold):
    df = df.copy()

    df['time_diff'].iloc[0] = np.mean(df['time_diff'].iloc[1:].values)
    ps0 = df[['2d_x', '2d_y']].iloc[0:-1].to_numpy()
    ps1 = df[['2d_x', '2d_y']].iloc[1:].to_numpy()

    # velo = np.divide(np.linalg.norm(ps1-ps0,axis=1),df['time_diff'].iloc[1:])
    velo = np.linalg.norm(ps1 - ps0, axis=1)
    velo = np.concatenate(([0], velo))
    df['velo'] = velo / df['time_diff']
    # TODO head velocity?
    df['gaze_label'] = np.where(df['velo'] <= velo_threshold, 'fixation', 'saccade')


    # drop single fixation intervals TODO
    df['fix_num'] = np.where(df['gaze_label'] == 'fixation', 1, 0)

    s0 = df['fix_num'].iloc[0:-1].to_numpy()
    s1 = df['fix_num'].iloc[1:].to_numpy()

    v = np.concatenate(([0], s1 - s0))
    df['fix_change'] = v

    for i in range(len(df) - 1):
        if df['fix_change'].iloc[i] == 1:
            if df['fix_change'].iloc[i + 1] == -1:
                df['gaze_label'].iloc[i] = 'saccade'

    df['gaze_label'] = np.where(df['pupilleft_r'].isna(), 'blink', df['gaze_label'])

    # label with numbers
    fix_lst = list()
    fix_index = 1
    sacc_index = 1
    blink_index = 1

    label = df['gaze_label'].iloc[0]

    if label == 'fixation':
        fix_lst.append(label + '_' + str(fix_index))
    if label == 'saccade':
        fix_lst.append(label + '_' + str(sacc_index))
    if label == 'blink':
        fix_lst.append(label + '_' + str(blink_index))

    for i in range(1, len(df)):
        if label == df['gaze_label'].iloc[i]:
            if label == 'fixation':
                fix_lst.append(label + '_' + str(fix_index))
            if label == 'saccade':
                fix_lst.append(label + '_' + str(sacc_index))
            if label == 'blink':
                fix_lst.append(label + '_' + str(blink_index))

        if label != df['gaze_label'].iloc[i]:
            if label == 'fixation':
                fix_index += 1
            if label == 'saccade':
                sacc_index += 1
            if label == 'blink':
                blink_index += 1

            label = df['gaze_label'].iloc[i]

            if label == 'fixation':
                fix_lst.append(label + '_' + str(fix_index))
            if label == 'saccade':
                fix_lst.append(label + '_' + str(sacc_index))
            if label == 'blink':
                fix_lst.append(label + '_' + str(blink_index))

    df['gaze_label_number'] = fix_lst
    return df