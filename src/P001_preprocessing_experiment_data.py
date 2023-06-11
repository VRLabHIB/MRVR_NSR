import os
from datetime import datetime
import pandas as pd
import numpy as np
import glob


class Preprocessing:
    def __init__(self, data_path):
        os.chdir(data_path + '/0_raw_experiment/')
        self.data_lst = glob.glob("ID*.csv")
        print("Number of files: ", len(self.data_lst))
        self.dataframes = [pd.read_csv(f, sep=',', header=0, index_col=False, low_memory=False) for f in self.data_lst]

    def get_dataframes(self):
        return self.dataframes

    def get_data_lst(self):
        return self.data_lst

    def add_experiment_condition(self):
        # Load pairs with the same ID
        for i in range(0, len(self.dataframes), 2):
            df1 = self.dataframes[i]
            print(self.data_lst[i])
            df2 = self.dataframes[i + 1]
            print(self.data_lst[i + 1])
            print(' ')

            t1 = datetime.strptime(df1['date'].iloc[0] + '.' + df1['time'].iloc[0], '%Y.%m.%d.%H.%M.%S.%f')
            t2 = datetime.strptime(df2['date'].iloc[0] + '.' + df2['time'].iloc[0], '%Y.%m.%d.%H.%M.%S.%f')

            print(t1)
            print(t2)
            result = t1 < t2
            print(result, ' \n')

            if result:
                df1.insert(1, 'cond', ['first'] * len(df1))
                df2.insert(1, 'cond', ['second'] * len(df2))
            if not result:
                df1.insert(1, 'cond', ['second'] * len(df1))
                df2.insert(1, 'cond', ['first'] * len(df2))

            self.dataframes[i] = df1
            self.dataframes[i + 1] = df2

            def set_stimulus_and_timing(self):
                for i in range(0, len(self.dataframes)):
                    df = self.dataframes[i]
                    df = clean_gaze_location_name(df, 'state')
                    df = rename_stimuli_to_align_2D_and_3D(df)
                    df = create_onset_offset(df, False)
                    self.dataframes[i] = df

            def drop_unused_variables(self):
                for i in range(0, len(self.dataframes)):
                    df = self.dataframes[i]
                    df.columns = df.columns.str.replace(' ', '')
                    df = df.drop(
                        ['state', 'substate', 'date', 'time', 'raylocationX', 'rayscreenlocationX',
                         'rayscreenlocationX.1',
                         'raylocationY',
                         'raylocationZ', 'combined.convergence_distance_mm', 'combined.convergence_distance_validity',
                         'combined.eye_data.eye_openness', 'combined.pupil_diameter_mm', 'coloredcube',
                         'combined.pupil_position_in_sensor_area.X', 'combined.pupil_position_in_sensor_area.Y'],
                        axis=1)

                    self.dataframes[i] = df

        ##### Helper functions for cleaning variables #####
        def clean_gaze_location_name(df, name):
            state = [i for i in df[name].unique() if i.startswith('BP_Gameplay')]
            df = df[df[name] == state[0]].copy()
            idx = np.where(df['stimulus'] == 'disabled')[0]
            df = df.iloc[:idx[0] + 1]
            return df

        def rename_stimuli_to_align_2D_and_3D(df):
            df['stimulus'] = df['stimulus'].str.replace(' ', '')
            df['stimulus'] = df['stimulus'].replace(
                {'20': '1', '9': '2', '27': '3', '3': '4', '4': '5', '16': '6', '17': '7', '2': '8', '21': '9',
                 '13': '10',
                 '7': '11', '25': '12', '12': '13', '28': '14', '1': '15',
                 '10': '16', '18': '17', '22': '18', '6': '19', '30': '20', '11': '21', '24': '22', '5': '23',
                 '29': '24',
                 '15': '25', '8': '26', '26': '27', '14': '28',
                 '23': '29', '19': '30'})
            return df

        def create_onset_offset(df, info):
            df.columns = df.columns.str.replace('stimulus', 'stimulus_ue')
            df = df.rename(columns={'rel_time': 'time'})

            if info:
                print(
                    'Create variables: rel_time (relative time until the appearance of the stimulus [stimulus appearance==0])'
                    ' stimulus (now includes the pause before the appearance), '
                    'onset (0==no stimulus shown, 1==stimulus shown)\n')

            df.insert(7, 'onset', np.zeros(len(df)))
            df.insert(7, 'stimulus', np.zeros(len(df)))
            # df.insert(3, 'rel_time', np.zeros(len(df)))

            # set up the stimulus distinction
            start_idx = 0
            stim_lst = list()  # check if all 30 stimuli are correctly detected
            i = 0
            while i < len(df):

                if np.logical_and(df['stimulus_ue'].iloc[i] != 'Pause', df['stimulus_ue'].iloc[i] != 'disabled'):

                    zero_time = df['time'].iloc[i]

                    while np.logical_and(df['stimulus_ue'].iloc[i] != 'Pause', df['stimulus_ue'].iloc[i] != 'disabled'):
                        i += 1
                        # stop at the end of the dataframe
                        if i == len(df):
                            break

                    end_idx = i

                    df_sub = df.iloc[start_idx:end_idx].copy()
                    rel_time = df_sub['time'] - zero_time
                    df_sub['onset'] = np.where(df_sub['stimulus_ue'] == 'Pause', 0, 1)

                    df["rel_time"].iloc[start_idx:end_idx] = rel_time
                    df['onset'].iloc[start_idx:end_idx] = df_sub['onset'].values
                    df['stimulus'].iloc[start_idx:end_idx] = [df_sub['stimulus_ue'].iloc[-1]] * len(df_sub)
                    stim_lst.append(df_sub['stimulus_ue'].iloc[-1])
                    start_idx = i
                i += 1
            print('Number of Stimuli processed: ', len(stim_lst))

            # Cut long pause before the first stimulus
            if info:
                print('\nCut long pause before the first stimulus up to max 4 sec before stimulus appearance \n')

            i = 0
            while df['rel_time'].iloc[i] < -4:
                i += 1
            df = df.iloc[i:]

            if info:
                print('Reset time variable after cut at beginning')

            df['time'] = df['time'] - df['time'].iloc[0]

            if info:
                print('Variable created: Time_Diff')

            df['time_diff'] = calculate_time_diff(df, 'time')

            return df.reset_index(drop=True)

        def calculate_time_diff(df, time):
            """
            Parameters
            ----------
            df : pandas dataframe
                raw, cleaned dataframe.
            time : string
                variable that stores the time changes of an experiment [sec].

            Returns
            -------
            time_diff : list
                conveys time difference between following steps of an experiment.

            """

            # create list with start time of the experiment
            time_diff = np.array([0])

            # create two vectors
            t0 = np.array(df[time][:-1].tolist())
            t1 = np.array(df[time][1:].tolist())

            # vectors subtraction
            diff = np.subtract(t1, t0)

            time_diff = np.append(time_diff, diff)

            return time_diff
