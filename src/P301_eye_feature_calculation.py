import os
from datetime import datetime
import pandas as pd

# import warnings
# warnings.filterwarnings('ignore')

import numpy as np
import glob

class Features:
    def __init__(self, data_path, location):
        self.df_f = None
        os.chdir(data_path + location)
        self.data_path = data_path
        self.data_lst = glob.glob("ID*.csv")
        print("Number of files: ", len(self.data_lst))
        self.dataframes = [pd.read_csv(f, sep=',', header=0, index_col=False, low_memory=False) for f in self.data_lst]

    def create_dataframe(self):

        stimuli = np.arange(1, 31)


        # Load pairs with the same ID
        self.df_f = pd.DataFrame({'ID':[], 'stimulus':[], '2DResponse':[], '3DResponse':[], '2DRT':[], '3DRT':[], 'first':[]})
        for i in range(0, len(self.dataframes), 2):
            df2 = self.dataframes[i]
            print(self.data_lst[i])
            df3 = self.dataframes[i + 1]
            print(self.data_lst[i + 1])
            if df2['ID'].iloc[0] == df3['ID'].iloc[0]:
                print('same id')
            else:
                print('not same id')
            print(' ')
            df2_cond = df2['cond'].iloc[0]
            df3_cond = df3['cond'].iloc[0]

            if df2_cond == 'first':
                first = '2D'
                pass
            if df3_cond == 'first':
                first = '3D'
                pass

            ID = df2['ID'].iloc[0]

            dim2 = df2['dimension'].iloc[0]
            print('Dim2 ',dim2)

            dim3 = df3['dimension'].iloc[0]
            print('Dim3 ',dim3)

            for stim in stimuli:
                # Add experiment information

                # Select stimulus interval
                df_sub2 = df2[df2['stimulus_ue'] == str(stim)]
                df_sub3 = df3[df3['stimulus_ue'] == str(stim)]

                # Get controller response (equal or unequal)
                click_idx2 = df_sub2.index[-1]
                response2D = click_response(df2, click_idx2)

                click_idx3 = df_sub3.index[-1]
                response3D = click_response(df3, click_idx3)

                # Reaction time is the last row of the relative time variable
                RT2 = df_sub2['rel_time'].iloc[-1]
                RT3 = df_sub3['rel_time'].iloc[-1]

                self.df_f.loc[len(self.df_f.index)] = [ID, stim, response2D, response3D, RT2, RT3, first]

        os.chdir(self.data_path + '\\meta\\')
        df_info = pd.read_csv('All_stimulus_information.csv')
        df_info_s = df_info[['Equal?', 'stimulus', 'AngularDisp', 'DiffType', 'x_rot_r', 'y_rot_r']]

        self.df_f = self.df_f.merge(df_info_s,on='stimulus')
        self.df_f['2DCorrect'] = (((self.df_f['2DResponse'] + self.df_f['Equal?']) % 2) + 1) % 2
        self.df_f['3DCorrect'] = (((self.df_f['3DResponse'] + self.df_f['Equal?']) % 2) + 1) % 2
        print('Done')

    def get_feature_dataset(self):
        return self.df_f


def click_response(df, idx):
    # left click == unequal, right click == equal
    left_click_t1 = df['controllerleftclick'].iloc[idx + 1]
    left_click_t0 = df['controllerleftclick'].iloc[idx]

    right_click_t1 = df['controllerrightclick'].iloc[idx + 1]
    right_click_t0 = df['controllerrightclick'].iloc[idx]

    if np.logical_and(left_click_t1 > left_click_t0, right_click_t1 == 0):
        response = 0

    if np.logical_and(right_click_t1 > right_click_t0, left_click_t1 == 0):
        response = 1

    return response