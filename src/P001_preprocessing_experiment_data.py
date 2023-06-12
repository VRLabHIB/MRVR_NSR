import os
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import glob

import src.pupil as pupil
from scipy.spatial.transform import Rotation as R
from sklearn import preprocessing


class Preprocessing:
    def __init__(self, data_path,location):
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
                ['state', 'substate', 'date', 'time', 'raylocationX', 'rayscreenlocationX', 'rayscreenlocationX.1',
                 'raylocationY',
                 'raylocationZ', 'combined.convergence_distance_mm', 'combined.convergence_distance_validity',
                 'combined.eye_data.eye_openness', 'combined.pupil_diameter_mm', 'coloredcube',
                 'combined.pupil_position_in_sensor_area.X', 'combined.pupil_position_in_sensor_area.Y'], axis=1)

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

            df.rename(columns={'stimulus':'stimulus_ue'},inplace=True)

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
            df = df.drop(['time'], axis = 1)
            df.insert(2, 'time', time)

            self.dataframes[file] = df.reset_index(drop=True)


    def process_and_create_variables(self):

        for i in range(0, len(self.dataframes)):
            df = self.dataframes[i]

            # Variable created: time_diff
            df['time_diff'] = calculate_time_diff(df, 'time')


    def add_experiment_condition(self):
        # Load pairs with the same ID
        for i in range(0, len(self.dataframes), 2):
            df1 = self.dataframes[i]
            print(self.data_lst[i])
            df2 = self.dataframes[i + 1]
            print(self.data_lst[i + 1])
            print(' ')
            if df1['ID'].iloc[0] == df2['ID'].iloc[0]:
                print('same id')
            else:
                print('not same id')

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


def get_angle(df, gazeX, gazeY, gazeZ, headX, headY, headZ):
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
    df[gazeX] = df[gazeX].replace(-1, np.nan)
    df[gazeY] = df[gazeY].replace(-1, np.nan)
    df[gazeZ] = df[gazeZ].replace(-1, np.nan)

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


def object_validation(df, left_pupil, right_pupil, hitcomponent):
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

    for i in df.index:
        if np.logical_or(df[right_pupil].isna()[i] == False, df[left_pupil].isna()[i] == False):
            hitobject.append(df[hitcomponent][i].split(' ')[-1])
            if hitobject[i] == "partitionwall" or hitobject[i] == "Lockers_02_A_SM":
                hitobject[i] = "wall"
            else:
                hitobject[i] = hitobject[i].split('.')[-1]
        else:
            hitobject.append('NaN')

    return hitobject

def pupil_preprocessing(df, left_pupil, right_pupil, save_path, vt_start=5, gap_margin=5, plot=False, save_plot=True):

    # TODO check parameters
    left_tr, df['ratio'] = pupil.tracking_ratio(df, left_pupil, thresholds=[0.1, 9], missings=[-1])
    df[left_pupil].loc[df['ratio'] == 0] = np.nan

    right_tr, df['ratio'] = pupil.tracking_ratio(df, right_pupil, thresholds=[0.1, 9], missings=[-1])
    df[right_pupil].loc[df['ratio'] == 0] = np.nan

    left, right = pupil.reconstruct_stream(df,save_path, variables=[left_pupil, right_pupil], vt_start=vt_start,
                                                gap_margin=gap_margin, plotting=plot, save_plot=save_plot)

    df_tr = pd.DataFrame({'left_tr': [left_tr], 'right_tr': [right_tr]})
    df_tr.index = [df['ID'].iloc[0]]
    return left, right, df_tr

# TODO gaze_interpolate necessary?

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

    ## separate dataframe into no-nan and nan values
    df_v = df[~df.isna().any(axis=1)]
    df_m = df[df.isna().any(axis=1)]

    ## normalization and overwriting the column with normalized values
    preprocessing.normalize(df_v[["gazeX", 'gazeY', 'gazeX']], norm='l2', copy=False)

    ##concat missings and non missings dataframes
    df = pd.concat([df_v, df_m])
    df = df.sort_index()

    return df
