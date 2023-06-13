import os
from datetime import datetime
import pandas as pd

# import warnings
# warnings.filterwarnings('ignore')

import numpy as np
import glob

from scipy.spatial.transform import Rotation as R


class Preprocessing:
    def __init__(self, data_path, location):
        self.df_tr = None
        os.chdir(data_path + location)
        self.data_lst = glob.glob("ID*.csv")
        print("Number of files: ", len(self.data_lst))
        self.dataframes = [pd.read_csv(f, sep=',', header=0, index_col=False, low_memory=False) for f in self.data_lst]

    def get_dataframes(self):
        return self.dataframes

    def get_data_lst(self):
        return self.data_lst

    def get_tr_data(self):
        return self.df_tr

    def save_dataframes(self, save_path):
        for i in range(0, len(self.dataframes)):
            name = self.data_lst[i]
            df = self.dataframes[i]
            df.to_csv(save_path + name, index=False)
        print('Files saved to: ', save_path)

    def data_cleaning(self):
        for file in range(0, len(self.dataframes)):
            print(self.data_lst[file])
            df = self.dataframes[file]
            dim = df['dimension'].iloc[0]
            print(dim)

            state = [i for i in df['state'].unique() if i.startswith('BP_Gameplay')]
            df = df[df['state'] == state[0]].copy()

            stim = np.arange(1, 31)
            stim_lst = np.append(stim, "Pause")

            # Remove everything, which is not pause or a stimulus
            idx = np.where(df['stimulus'] == 'disabled')[0]
            df = df.iloc[:idx[0] + 1]

            # Drop columns and rename rel_time to time
            df.columns = df.columns.str.replace(' ', '')
            df = df.drop(
                ['state', 'substate', 'raylocationX', 'rayscreenlocationX', 'rayscreenlocationX.1',
                 'raylocationY',
                 'raylocationZ', 'combined.convergence_distance_mm', 'combined.convergence_distance_validity',
                 'combined.eye_data.eye_openness', 'combined.pupil_diameter_mm', 'coloredcube',
                 'combined.pupil_position_in_sensor_area.X', 'combined.pupil_position_in_sensor_area.Y'], axis=1)
            df = df.rename(columns={'time': 'systemtime'})
            df = df.rename(columns={'rel_time': 'time'})

            df['stimulus'] = df['stimulus'].str.replace(' ', '')

            # Change stimuli names for the 2D condition, such that they align with the 3D naming
            if df['dimension'].iloc[0] == 2:
                df['stimulus'] = df['stimulus'].replace(
                    {'20': '1', '9': '2', '27': '3', '3': '4', '4': '5', '16': '6', '17': '7', '2': '8', '21': '9',
                     '13': '10', '7': '11', '25': '12', '12': '13', '28': '14', '1': '15',
                     '10': '16', '18': '17', '22': '18', '6': '19', '30': '20', '11': '21', '24': '22', '5': '23',
                     '29': '24', '15': '25', '8': '26', '26': '27', '14': '28',
                     '23': '29', '19': '30'})

            df.rename(columns={'stimulus': 'stimulus_ue'}, inplace=True)

            # Create variables: rel_time (relative time until the appearance of the stimulus [stimulus appearance==0]),
            # stimulus (now includes the pause before the appearance), onset (0==no stimulus shown, 1==stimulus shown)
            df.insert(5, 'onset', np.zeros(len(df)))
            df.insert(5, 'stimulus', np.zeros(len(df)))
            df.insert(3, 'rel_time', np.zeros(len(df)))

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
                    rel_time = df_sub['time'].values - zero_time
                    df_sub['onset'] = np.where(df_sub['stimulus_ue'] == 'Pause', 0, 1)

                    df["rel_time"].iloc[start_idx:end_idx] = rel_time
                    df['onset'].iloc[start_idx:end_idx] = df_sub['onset'].values
                    df['stimulus'].iloc[start_idx:end_idx] = [df_sub['stimulus_ue'].iloc[-1]] * len(df_sub)
                    stim_lst.append(df_sub['stimulus_ue'].iloc[-1])
                    start_idx = i
                i += 1
            print('Number of Stimuli processed: ', len(stim_lst))

            # Cut long pause before the first stimulus up to max 4 sec before stimulus appearance
            i = 0
            while df['rel_time'].iloc[i] < -4:
                i += 1
            df = df.iloc[i:]

            df['time'] = df['time'] - df['time'].iloc[0]

            time = df['time'].values
            df = df.drop(['time'], axis=1)
            df.insert(2, 'time', time)

            self.dataframes[file] = df.reset_index(drop=True)

    def add_experiment_condition(self):
        # Load pairs with the same ID
        for i in range(0, len(self.dataframes), 2):
            df1 = self.dataframes[i]
            print(self.data_lst[i])
            df2 = self.dataframes[i + 1]
            print(self.data_lst[i + 1])
            if df1['ID'].iloc[0] == df2['ID'].iloc[0]:
                print('same id')
            else:
                print('not same id')
            print(' ')

            t1 = datetime.strptime(df1['date'].iloc[0] + '.' + df1['systemtime'].iloc[0], '%Y.%m.%d.%H.%M.%S.%f')
            t2 = datetime.strptime(df2['date'].iloc[0] + '.' + df2['systemtime'].iloc[0], '%Y.%m.%d.%H.%M.%S.%f')

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

            df1 = df1.drop(['date', 'systemtime'], axis=1)
            df2 = df2.drop(['date', 'systemtime'], axis=1)

            self.dataframes[i] = df1
            self.dataframes[i + 1] = df2

    def process_pupil(self):
        self.df_tr = pd.DataFrame({'name': [], 'left_tr': [], 'right_tr': []})

        for file in range(0, len(self.dataframes)):
            df = self.dataframes[file]
            name = self.data_lst[file]
            print(name)

            # Blink detection and cleaning
            df['left.pupil_diameter_mm'] = df['left.pupil_diameter_mm'].replace(-1, np.nan)
            df['right.pupil_diameter_mm'] = df['right.pupil_diameter_mm'].replace(-1, np.nan)

            df['left.pupil_diameter_clean'] = df['left.pupil_diameter_mm'].copy()
            df['right.pupil_diameter_clean'] = df['right.pupil_diameter_mm'].copy()

            i = 0
            while i < len(df):
                idx_s = 0
                idx_e = 0

                if np.isnan(df['left.pupil_diameter_clean'].iloc[i]):
                    idx_s = i

                    while np.logical_and(np.isnan(df['left.pupil_diameter_clean'].iloc[i]), i < len(df)-1):
                        i += 1

                    idx_e = i + 1
                    if idx_e < len(df):
                        dur = df['time'].iloc[idx_e] - df['time'].iloc[idx_s]
                        if dur < 0.5:
                            df['left.pupil_diameter_clean'].iloc[idx_s - 1:idx_e] = np.nan
                    if idx_e >= len(df):
                        dur = df['time'].iloc[idx_e-1] - df['time'].iloc[idx_s]
                        if dur < 0.5:
                            df['left.pupil_diameter_clean'].iloc[idx_s - 1:idx_e-1] = np.nan
                i += 1

            i = 0
            while i < len(df):
                idx_s = 0
                idx_e = 0

                if np.isnan(df['right.pupil_diameter_clean'].iloc[i]):
                    idx_s = i

                    while np.logical_and(np.isnan(df['right.pupil_diameter_clean'].iloc[i]), i < len(df)-1):
                        i += 1

                    idx_e = i + 1
                    if idx_e < len(df):
                        dur = df['time'].iloc[idx_e] - df['time'].iloc[idx_s]
                        if dur < 0.5:
                            df['right.pupil_diameter_clean'].iloc[idx_s - 1:idx_e] = np.nan
                    if idx_e >= len(df):
                        dur = df['time'].iloc[idx_e-1] - df['time'].iloc[idx_s]
                        if dur < 0.5:
                            df['right.pupil_diameter_clean'].iloc[idx_s - 1:idx_e-1] = np.nan

                i += 1

            # Combined pupil diameter
            combined = np.mean(df[['left.pupil_diameter_clean', 'right.pupil_diameter_clean']], axis=1)
            print('len of combined ', len(combined))
            print('len of data ', len(df))

            df['combined.pupil_diameter_clean'] = combined

            # Baseline correction
            df['combined.pupil_diameter_corr'] = np.nan
            df_sub = df[df['onset'] == 0]
            stim_lst = np.arange(1, 31)

            for i in stim_lst:
                df_off = df_sub[df_sub['stimulus'] == i]
                baseline = np.nanmedian(df_off['combined.pupil_diameter_clean'])

                df_c = df[df['stimulus'] == int(i)]
                idx_c = df_c.index

                df['combined.pupil_diameter_corr'].iloc[idx_c] = df['combined.pupil_diameter_clean'].iloc[
                                                                     idx_c].values - baseline

            # Tracking ratio
            left_tr = get_tracking_ratio(df, 'left.pupil_diameter_clean')
            right_tr = get_tracking_ratio(df, 'right.pupil_diameter_clean')

            self.df_tr.loc[len(self.df_tr.index)] = [name, left_tr, right_tr]

            self.dataframes[file] = df

    def select_cases_on_tracking_ratio(self, valid_path, invalid_path):
        for file in range(0, len(self.dataframes), 2):
            df1 = self.dataframes[file]
            name1 = self.data_lst[file]
            print(self.data_lst[file])
            df2 = self.dataframes[file+1]
            name2 = self.data_lst[file+1]
            print(self.data_lst[file + 1])
            if df1['ID'].iloc[0] == df2['ID'].iloc[0]:
                print('same id')
            else:
                print('not same id')
            print(' ')

            trs1 = self.df_tr[self.df_tr['name'] == name1]
            tr_avg1 = (trs1['left_tr'].values + trs1['right_tr'].values) / 2

            trs2 = self.df_tr[self.df_tr['name'] == name2]
            tr_avg2 = (trs2['left_tr'].values + trs2['right_tr'].values) / 2

            if np.logical_and(tr_avg1 >= 80, tr_avg2 >= 80):
                print('\nTracking ratio sufficient:')
                print('tr avg1: ', tr_avg1)
                print('tr avg2: ', tr_avg2)
                df1.to_csv(valid_path + '{}'.format(name1), index=False)
                df2.to_csv(valid_path + '{}'.format(name2), index=False)

            if np.logical_or(tr_avg1 < 80, tr_avg2 < 80):
                print('\nTracking ratio NOT sufficient:')
                print('tr avg1: ', tr_avg1)
                print('tr avg2: ', tr_avg2)
                df1.to_csv(invalid_path + '{}'.format(name1), index=False)
                df2.to_csv(invalid_path + '{}'.format(name2), index=False)


    def calculate_and_process_variables(self):
        for file in range(0, len(self.dataframes)):
            df = self.dataframes[file]
            name = self.data_lst[file]
            print(name)

            # Variable created: time_diff
            df['time_diff'] = calculate_time_diff(df, 'time')
            df['hitobject'] = object_validation(df, 'combined.pupil_diameter_corr', 'rayhitcomponent')

            df['head_x'], df['head_y'], df['head_z'] = get_head_direction(df,'playerrotationRoll','playerrotationPitch','playerrotationYaw')

            angledir = get_angle(df,'combined.pupil_diameter_corr', 'combined.gaze_direction_normalized.X',
                                    'combined.gaze_direction_normalized.Y',
                                    'combined.gaze_direction_normalized.Z',
                                    'head_x',
                                    'head_y',
                                    'head_z')

            df['head_angle'] = angledir[1]
            df['gaze_angle'] = angledir[0]

            self.dataframes[file] = df


def get_tracking_ratio(df, variable, thresholds=[1, 9], missings=[-1]):
    df_sub = df[[variable]]

    df_sub['ratio'] = [1] * len(df_sub)

    if thresholds is not None:
        df_sub.loc[np.logical_or(df_sub[variable] <= thresholds[0], df_sub[variable] >= thresholds[1]), 'ratio'] = 0

    # mark all np.nan values
    df_sub.loc[np.isnan(df_sub[variable]), 'ratio'] = 0

    if missings is not None:
        for mis in missings:
            df_sub.loc[df_sub[variable] == mis, 'ratio'] = 0

    ratio = df_sub['ratio'].sum() / len(df_sub) * 100

    return ratio


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

def object_validation(df, combined_pupil, hitcomponent):
    """

    Parameters
    ----------
    df : pandas dataframe
        raw, cleaned dataframe.
    left_pupil : string
        preprocessed variable with left pupil diameter information [mm].
    right_pupil : string
        preprocessed variable with right pupil diameter information [mm].
    hitcomponent : string
        column name with objects looked at.

    Returns
    -------
    hitobject : list
        preprocessed list of objects looked at.

    """

    hitobject = list()

    for i in range(len(df)):
        if np.isnan(df[combined_pupil].iloc[i]):
            hitobject.append('NaN')
        else:
            hitobject.append(df[hitcomponent].iloc[i].split(' ')[-1])
            if hitobject[i] == "partitionwall" or hitobject[i] == "Lockers_02_A_SM":
                hitobject[i] = "wall"
            else:
                hitobject[i] = hitobject[i].split('.')[-1]
    return hitobject

def get_head_direction(df, roll, pitch, yaw):
    """

    Parameters
    ----------
    df : pandas dataframe
        should contain variables with names roll, pitch, yaw.
    roll : string
        variable name of roll rotation variable [given in angle degree].
    pitch : string
        variable name of pitch rotation variable [given in angle degree].
    yaw : string
        variable name of yaw rotation variable [given in angle degree]..

    Returns
    -------
    headX : float
        X coordinate of the head direction.
    headY : float
        Y coordinate of the head direction.
    headZ : float
        Z coordinate of the head direction.

    """

    df_sub = df[[roll, pitch, yaw]]

    headX = list()
    headY = list()
    headZ = list()

    for i in range(len(df_sub)):
        # create vector of rotation angles
        rv = np.array(df_sub.iloc[i].values)

        # create rotation
        # variables are roll pitch yaw
        r = R.from_euler('xyz', rv, degrees=True)

        # rotate forward head direction
        x = np.array([1, 0, 0])
        v = np.matmul(r.as_matrix(), x)

        # store variables
        headX.append(v[0])
        headY.append(v[1])
        headZ.append(v[2])

    return headX, headY, headZ


def get_angle(df,combined_pupil, gazeX, gazeY, gazeZ, headX, headY, headZ):
    """

    Parameters
    ----------
    df : pandas dataframe
        raw, cleaned dataframe.
    gazeX : string
        variable corresponding to gaze X vector value [normalized].
    gazeY : string
        variable corresponding to gaze Y vector value [normalized].
    gazeZ : string
        variable corresponding to gaze Z vector value [normalized].
    headX : string
        variable corresponding to head X vector value [not normalized].
    headY : string
        variable corresponding to head X vector value [not normalized].
    headZ : string
        variable corresponding to head X vector value [not normalized].

    Returns
    -------
    gaze_angle : numpy array
        gaze angle between two following experimental steps [degree].
    head_angle : numpy array
        head angle between two following experimental steps [degree].

    """

    # for gaze angle
    #df[gazeX] = df[gazeX].replace(-1, np.nan)
    #df[gazeY] = df[gazeY].replace(-1, np.nan)
    #df[gazeZ] = df[gazeZ].replace(-1, np.nan)

    df[gazeX] = np.where(df[combined_pupil].isna(), np.nan, df[gazeX])
    df[gazeY] = np.where(df[combined_pupil].isna(), np.nan, df[gazeY])
    df[gazeZ] = np.where(df[combined_pupil].isna(), np.nan, df[gazeZ])

    m_t1 = np.array(df[[gazeX, gazeY, gazeZ]].iloc[1:])  # array starts at t1 until tn; converted into matrix
    m_t0 = np.array(df[[gazeX, gazeY, gazeZ]].iloc[:-1])  # array starts at t0 until tn-1; converted into matrix

    gaze_angle = np.degrees(np.arccos((m_t1 * m_t0).sum(-1)))
    gaze_angle = np.concatenate(([0], gaze_angle))  # add the zero at the start

    # for head angle
    m_t1 = np.array(df[[headX, headY, headZ]].iloc[1:])  # array starts at t1 until tn; converted into matrix
    m_t0 = np.array(df[[headX, headY, headZ]].iloc[:-1])  # array starts at t0 until tn-1; converted into matrix

    head_angle = np.degrees(np.arccos((m_t1 * m_t0).sum(-1)))
    head_angle = np.concatenate(([0], head_angle))  # add the zero at the start

    return gaze_angle, head_angle


def gaze_interpolate(df, gazeX, gazeY, gazeZ):
    """

    Parameters
    ----------
    df : pandas dataframe
        DATAFRAME WITH GAZE COLUMNS.
    gazeX : string
        NAME OF COLUMN WITH GAZE X VECTOR.
    gazeY : string
        NAME OF COLUMN WITH GAZE Y VECTOR.
    gazeZ : string
        NAME OF COLUMN WITH GAZE Z VECTOR.

    Returns
    -------
    pandas series
        GAZE X INTERPOLATED AND NORMALIZED.
    pandas series
        GAZE Y INTERPOLATED AND NORMALIZED.
    pandas series
        GAZE Z INTERPOLATED AND NORMALIZED.

    """

    df["gazeX"] = df[gazeX].copy()
    df["gazeY"] = df[gazeY].copy()
    df["gazeZ"] = df[gazeZ].copy()

    df = df[['gazeX', 'gazeY', 'gazeZ']]

    ## according to the documentation of eyetracker
    df[df["gazeX"] == -1.0] = np.nan
    df[df["gazeY"] == -1.0] = np.nan
    df[df["gazeZ"] == -1.0] = np.nan

    ## perform interpolation

    df['gazeX'] = df["gazeX"].interpolate(method='polynomial', order=1, limit_area='inside', limit=5)
    df['gazeY'] = df["gazeY"].interpolate(method='polynomial', order=1, limit_area='inside', limit=5)
    df['gazeZ'] = df["gazeZ"].interpolate(method='polynomial', order=1, limit_area='inside', limit=5)

    # TODO check here
    ## separate dataframe into no-nan and nan values
    # df_v = df[~df.isna().any(axis=1)]
    # df_m = df[df.isna().any(axis=1)]

    ## normalization and overwriting the column with normalized values
    # preprocessing.normalize(df_v[["gazeX", 'gazeY', 'gazeX']], norm='l2', copy=False)

    ##concat missings and non missings dataframes
    # df = pd.concat([df_v, df_m])
    # df = df.sort_index()

    return df['gazeX'].values, df['gazeY'].values, df['gazeZ'].values
