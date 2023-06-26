import os
from datetime import date
import pandas as pd

import numpy as np
import glob

import scipy
import src.helper as h

class FixationDetection:
    def __init__(self, data_path, valid_path, location):
        os.chdir(data_path + location)
        self.data_lst = glob.glob("ID*.csv")
        self.data_path = data_path
        self.valid_path = valid_path
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

    def run_fixation_detection(self, method):
        id_lst = list()
        dim_lst = list()
        corr_lst = list()

        for file in range(0, len(self.dataframes)):
            name = self.data_lst[file]
            print(name)
            df = self.dataframes[file]
            df['ivt_gaze_labels'], df['ivt_gaze_label_number'], df['ivt_label_duration'] = run_IVT(df)

            disp_threshold = np.tan(2 * np.pi / 180)*(84-np.nanmean(df['playerlocationX']))
            df['idt_gaze_label'],df['idt_gaze_label_number'], df['idt_label_duration'] = run_IDT(df, disp_threshold)

            # calculate union of both algorithms
            df['gaze_label'] = df['ivt_gaze_labels'].copy()

            for fix in df[df['idt_gaze_label_number'].str.startswith('fix')]['idt_gaze_label_number'].unique():
                df_s = df[df['idt_gaze_label_number']==fix]
                index = df_s.index
                if all(c == 'None' for c in df_s['ivt_gaze_labels']):
                    df['gaze_label'].iloc[index] = 'fixation'

            df['gaze_label_number'] = calc_gaze_labels_with_numbers(df, 'gaze_label')
            df['label_duration'] = np.zeros(len(df))

            for label in df['gaze_label_number'].unique():
                df_f = df[df['gaze_label_number'] == label]
                idx = df_f.index
                dur = (df_f['time'].iloc[-1] - df_f['time'].iloc[0])
                df['label_duration'].loc[idx] = dur

            self.dataframes[file] = df

    def calculate_fixation_midpoint(self):
        for file in range(len(self.data_lst)):
            name = self.data_lst[file]
            print(name)
            df = self.dataframes[file]
            df['fixation_midpointX'] = np.nan
            df['fixation_midpointY'] = np.nan

            stim_lst = np.arange(1,31)

            for stim in stim_lst:
                df_s = df[np.logical_and(df['stimulus']==stim,df['onset']==1)]

                fix_lst = df_s[df_s['gaze_label_number'].str.startswith('fix')]['gaze_label_number'].unique()
                for fix in fix_lst:
                    df_f = df_s[df_s['gaze_label_number']==fix]
                    index = df_f.index

                    xm, ym, dist = h.calculate_centroid(df_f['2d_x'].values,df_f['2d_y'].values)

                    df['fixation_midpointX'].iloc[index] = xm
                    df['fixation_midpointY'].iloc[index] = ym


            self.dataframes[file] = df

    def compute_detection_correlation(self):
        today = date.today()
        id_lst = list()
        dim_lst = list()
        corr_lst = list()

        for file in range(0, len(self.dataframes)):
            name = self.data_lst[file]
            print(name)
            df = self.dataframes[file]
            ID = df['ID'].iloc[0]
            dim = df['dimension'].iloc[0]

            # Create testing dataframe
            stim_lst = np.arange(1, 31)
            avg_ivt_dur_lst = list()
            avg_idt_dur_lst = list()
            avg_dur_lst = list()
            avg_sacc_lst = list()

            for stim in stim_lst:
                df_s = df[np.logical_and(df['stimulus'] == stim, df['onset'] == 1)]
                ivt_fix_lst = df_s[df_s['ivt_gaze_label_number'].str.startswith('fix')][
                    'ivt_gaze_label_number'].unique()
                idt_fix_lst = df_s[df_s['idt_gaze_label_number'].str.startswith('fix')][
                    'idt_gaze_label_number'].unique()
                fix_lst = df_s[df_s['gaze_label_number'].str.startswith('fix')][
                    'gaze_label_number'].unique()

                ivt_dur_lst = list()
                for fix in ivt_fix_lst:
                    df_f = df_s[df_s['ivt_gaze_label_number'] == fix]
                    ivt_dur_lst.append(df_f['ivt_label_duration'].iloc[0])
                avg_ivt_dur_lst.append(np.mean(ivt_dur_lst))

                idt_dur_lst = list()
                for fix in idt_fix_lst:
                    df_f = df_s[df_s['idt_gaze_label_number'] == fix]
                    idt_dur_lst.append(df_f['idt_label_duration'].iloc[0])
                avg_idt_dur_lst.append(np.mean(idt_dur_lst))

                dur_lst = list()
                for fix in fix_lst:
                    df_f = df_s[df_s['gaze_label_number'] == fix]
                    dur_lst.append(df_f['label_duration'].iloc[0])
                avg_dur_lst.append(np.mean(dur_lst))

                sac_lst = df_s[df_s['ivt_gaze_label_number'].str.startswith('sac')]['ivt_gaze_label_number'].unique()
                dur_lst = list()
                for sac in sac_lst:
                    df_f = df_s[df_s['ivt_gaze_label_number'] == sac]
                    dur_lst.append(df_f['ivt_label_duration'].iloc[0])

                avg_sacc_lst.append(np.mean(dur_lst))

            s = pd.DataFrame({'stim': stim_lst, 'avg_ivt_fix_dur': avg_ivt_dur_lst, 'avg_idt_fix_dur': avg_idt_dur_lst,
                              'avg_dur_lst': avg_dur_lst, 'avg_sac_dur': avg_sacc_lst})

            # If NaN replace with mean
            s['avg_ivt_fix_dur'] = s['avg_ivt_fix_dur'].replace(np.nan, np.mean(s['avg_ivt_fix_dur']))
            s['avg_idt_fix_dur'] = s['avg_idt_fix_dur'].replace(np.nan, np.mean(s['avg_idt_fix_dur']))

            pearson = scipy.stats.pearsonr(s['avg_ivt_fix_dur'], s['avg_idt_fix_dur'])
            corr_lst.append(pearson[0])
            id_lst.append(ID)
            dim_lst.append(dim)

        df_cor = pd.DataFrame({'ID': id_lst, 'dimension': dim_lst, 'pearson_corr': corr_lst})
        df_cor.to_csv(self.valid_path + '{}_ivt_idt_correlations.csv'.format(today), index=False)


def run_IVT(df):
    df = df.copy()

    df['gaze_label'] = np.where(np.logical_and(df['head_angle_velo'] < 7, df['gaze_angle_velo'] < 30), 'fixation',
                                'None')
    df['gaze_label'] = np.where(df['gaze_angle_velo'] > 60, 'saccade',
                                df['gaze_label'])

    df['gaze_label'] = np.where(df['gaze_angle_velo'].isna(), 'blink', df['gaze_label'])

    # label with numbers
    df['gaze_label_number'] = calc_gaze_labels_with_numbers(df,'gaze_label')


    fix_lst = df[df['gaze_label_number'].str.startswith('fix')]['gaze_label_number'].unique()
    for fix in fix_lst:
        df_f = df[df['gaze_label_number']==fix].copy()
        idx = df_f.index
        dur = (df_f['time'].iloc[-1]-df_f['time'].iloc[0])
        if np.logical_or(dur < 0.1, dur > 0.7):
            df['gaze_label'].loc[idx] = 'None'

    sac_lst = df[df['gaze_label_number'].str.startswith('sac')]['gaze_label_number'].unique()
    for sac in sac_lst:
        df_f = df[df['gaze_label_number']==sac].copy()
        idx = df_f.index
        dur = (df_f['time'].iloc[-1]-df_f['time'].iloc[0])
        if np.logical_or(dur <=0, dur > 0.08):
            df['gaze_label'].loc[idx] = 'None'


    df['gaze_label_number'] = calc_gaze_labels_with_numbers(df,'gaze_label')
    df['label_duration'] = np.zeros(len(df))

    for label in df['gaze_label_number'].unique():
        df_f = df[df['gaze_label_number']==label]
        idx = df_f.index
        dur = (df_f['time'].iloc[-1] - df_f['time'].iloc[0])
        df['label_duration'].loc[idx] = dur

    return df['gaze_label'].values, df['gaze_label_number'].values, df['label_duration'].values


def run_IDT(df, disp_threshold, dur_threshold=0.1):
    dfs = df[['time', '2d_x', '2d_y','combined.pupil_diameter_clean']].copy()
    dfs['idt_gaze_label'] = 'None'

    start_idx = 0
    end_idx = 0

    while start_idx < len(dfs):
        #search for 100ms interval
        start_time = dfs['time'].loc[start_idx]
        end_idx = start_idx

        while dfs['time'].iloc[end_idx]-start_time < 0.1:
            diff = dfs['time'].iloc[end_idx]-start_time
            end_idx+=1
            if end_idx >= len(dfs):
                break

        init_index = np.arange(start_idx,end_idx)
        init_window = dfs.iloc[init_index]

        D = (np.max(init_window['2d_x'])-np.min(init_window['2d_x']))+(np.max(init_window['2d_y'])-np.min(init_window['2d_y']))

        add = 0
        if D <= disp_threshold:
            add += (end_idx-start_idx)
            while np.logical_and(D <= disp_threshold, end_idx+1 < len(df)):
                init_index = np.append(init_index,end_idx)
                init_window = dfs.loc[init_index]
                D = (np.max(init_window['2d_x']) - np.min(init_window['2d_x'])) + (
                            np.max(init_window['2d_y']) - np.min(init_window['2d_y']))
                end_idx += 1
                add+=1
            dfs['idt_gaze_label'].loc[init_index[:-2]] = 'fixation'

        start_idx += add
        start_idx += 1


    dfs['idt_gaze_label'] = np.where(dfs['combined.pupil_diameter_clean'].isna(), 'blink', dfs['idt_gaze_label'])

    dfs['idt_gaze_label_number'] = calc_gaze_labels_with_numbers(dfs,'idt_gaze_label')
    dfs['idt_label_duration'] = np.zeros(len(dfs))

    for label in dfs['idt_gaze_label_number'].unique():
        df_f = dfs[dfs['idt_gaze_label_number'] == label]
        idx = df_f.index
        dur = (df_f['time'].iloc[-1] - df_f['time'].iloc[0])
        dfs['idt_label_duration'].loc[idx] = dur


    dfs['idt_gaze_label'] = np.where(np.logical_and(dfs['idt_gaze_label'].str.startswith('fix'),np.logical_or(
                                                    dfs['idt_label_duration']<0.1,dfs['idt_label_duration']>0.8)),
                                                    'None', dfs['idt_gaze_label'])

    # repeat procedure for cleaned variables
    dfs['idt_gaze_label_number'] = calc_gaze_labels_with_numbers(dfs,'idt_gaze_label')
    dfs['idt_label_duration'] = np.zeros(len(dfs))

    for label in dfs['idt_gaze_label_number'].unique():
        df_f = dfs[dfs['idt_gaze_label_number'] == label]
        idx = df_f.index
        dur = (df_f['time'].iloc[-1] - df_f['time'].iloc[0])
        dfs['idt_label_duration'].loc[idx] = dur

    return dfs['idt_gaze_label'].values, dfs['idt_gaze_label_number'].values, dfs['idt_label_duration'].values

def calc_gaze_labels_with_numbers(df, label_var):
    fix_lst = list()
    fix_index = 1
    sacc_index = 1
    blink_index = 1
    none_index = 1

    label = df[label_var].iloc[0]

    if label == 'fixation':
        fix_lst.append(label + '_' + str(fix_index))
    if label == 'saccade':
        fix_lst.append(label + '_' + str(sacc_index))
    if label == 'blink':
        fix_lst.append(label + '_' + str(blink_index))
    if label == 'None':
        fix_lst.append(label + '_' + str(none_index))

    for i in range(1, len(df)):
        if label == df[label_var].iloc[i]:
            if label == 'fixation':
                fix_lst.append(label + '_' + str(fix_index))
            if label == 'saccade':
                fix_lst.append(label + '_' + str(sacc_index))
            if label == 'blink':
                fix_lst.append(label + '_' + str(blink_index))
            if label == 'None':
                fix_lst.append(label+ '_' + str(none_index))

        if label != df[label_var].iloc[i]:
            if label == 'fixation':
                fix_index += 1
            if label == 'saccade':
                sacc_index += 1
            if label == 'blink':
                blink_index += 1
            if label == 'None':
                none_index += 1

            label = df[label_var].iloc[i]

            if label == 'fixation':
                fix_lst.append(label + '_' + str(fix_index))
            if label == 'saccade':
                fix_lst.append(label + '_' + str(sacc_index))
            if label == 'blink':
                fix_lst.append(label + '_' + str(blink_index))
            if label == 'None':
                fix_lst.append(label+ '_' + str(none_index))

    return fix_lst