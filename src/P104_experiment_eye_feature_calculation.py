import os
from datetime import date
import pandas as pd

# import warnings
# warnings.filterwarnings('ignore')

import numpy as np
import glob

import src.helper as h

class Features:
    def __init__(self, data_path, project_path, location):
        self.df_feature = None
        self.df_f = None
        os.chdir(data_path + location)
        self.data_path = data_path
        self.project_path = project_path
        self.data_lst = glob.glob("ID*.csv")
        print("Number of files: ", len(self.data_lst))
        self.dataframes = [pd.read_csv(f, sep=',', header=0, index_col=False, low_memory=False) for f in self.data_lst]

    def create_feature_dataset_count_measures(self):
        ID_lst = list()
        dim_lst = list()
        cond_lst = list()
        stim_lst = list()

        n_fix_lst = list()
        n_fix_per_sec_lst = list()
        n_sacc_lst = list()
        n_sacc_per_sec_lst = list()
        ratio_mean_lst = list()
        mean_fix_dur_lst = list()
        mean_distance_to_figure_lst = list()
        mean_angle_around_figure_lst = list()

        figure_location = np.array([84, 5, 109])
        player_start_position = np.array([-21, 5])

        for file in range(len(self.data_lst)):
            stimuli = np.arange(1,31)
            name = self.data_lst[file]
            print(name)
            df = self.dataframes[file]

            ID = df['ID'].iloc[0]
            dim = df['dimension'].iloc[0]
            cond = df['cond'].iloc[0]

            for stim in stimuli:
                ratio_lst =list()
                fix_dur_lst = list()

                df_s = df[df['stimulus_ue']==str(stim)]
                df_s['segment_part'] = df_s['segment_part'].replace(np.nan, 'None')
                df_s['segment_arm'] = df_s['segment_arm'].replace(np.nan, 'None')

                # Controller response and reaction time
                RT = df_s['rel_time'].iloc[-1]

                ID_lst.append(ID)
                dim_lst.append(dim)
                cond_lst.append(cond)
                stim_lst.append(stim)

                # distance to object
                player_location = df_s[['playerlocationX', 'playerlocationY', 'playerlocationZ']].to_numpy()
                distance_vector = player_location - figure_location
                distance = np.linalg.norm(distance_vector, axis=1)
                mean_distance_to_figure_lst.append(np.nanmean(distance))

                # player circular rotation around the figure
                alpha = np.abs(2 * np.degrees(np.sin(0.5 * (np.linalg.norm(player_start_position-player_location[:,:2],axis=1))
                                            / np.linalg.norm(player_start_position-figure_location[:2]))))
                mean_angle_around_figure_lst.append(np.nanmean(alpha))

                #fixations
                unique_gaze_labels = df_s['gaze_label_number'].unique()
                fixations = [fix for fix in unique_gaze_labels if fix.startswith('fix')]

                dfcount_fix = df_s[np.logical_and(df_s['ivt_gaze_labels'].isna(), df_s['idt_gaze_label'] == 'fixation')]
                unique_gaze_labels_idt = dfcount_fix['gaze_label_number'].unique()
                fixations_idt = [fix for fix in unique_gaze_labels_idt if fix.startswith('fix')]

                for fix in fixations_idt:
                    df_fix = df_s[df_s['gaze_label_number'] == fix]

                    fix_dur = df_fix['label_duration'].iloc[0]

                    fix_dur_lst.append(fix_dur)

                mean_fix_dur_lst.append(np.nanmean(fix_dur_lst))

                if int(dim)==3:
                    try:
                        ratio  = len(fixations_idt)/len(fixations)
                        ratio_lst.append(ratio)
                    except:
                        ratio_lst.append(0)

                n_fix_lst.append(len(fixations))
                n_fix_per_sec_lst.append((len(fixations)/RT))

                saccades = [sac for sac in unique_gaze_labels if sac.startswith('sac')]
                n_sacc_lst.append(len(saccades))
                n_sacc_per_sec_lst.append((len(saccades)/RT))

            if int(dim)==3:
                ratio_mean_lst.append(np.nanmean(ratio_lst))
        mean_ratio = np.nanmean(ratio_lst)
        sd_ratio = np.nanstd(ratio_lst)
        self.df_feature = pd.DataFrame(
            {'ID': ID_lst, 'dimension': dim_lst, 'condition': cond_lst, 'stimulus': stim_lst,
             'Number of Fixations': n_fix_lst, 'Relative Number of Fixations': n_fix_per_sec_lst,
             'Number of Saccades': n_sacc_lst, 'Relative Number of Saccades': n_sacc_per_sec_lst,
             'Mean fixation duration during head movement': mean_fix_dur_lst,
             'Mean distance to figure': mean_distance_to_figure_lst,
             'Mean angle around figure':mean_angle_around_figure_lst
             })
        print(' ')


    def create_feature_dataset(self):
        ID_lst = list()
        dim_lst = list()
        cond_lst = list()
        stim_lst = list()

        repsonse_lst = list()
        RT_lst = list()

        pupil_dia_lst = list()
        pupil_dia_ampl_lst = list()

        mean_fix_dur_lst = list()
        regressive_fix_dur_lst = list()
        ratio_inner_outer_dur_lst = list()
        ratio_left_right_dur_lst = list()

        avg_sacc_dur_lst = list()
        avg_sacc_velocity_lst = list()

        head_rot_speed_lst = list()
        head_loc_speed_lst = list()

        strat_ratio_lst = list()

        for file in range(len(self.data_lst)):
            stimuli = np.arange(1,31)
            name = self.data_lst[file]
            print(name)
            df = self.dataframes[file]

            ID = df['ID'].iloc[0]
            dim = df['dimension'].iloc[0]
            cond = df['cond'].iloc[0]

            for stim in stimuli:
                df_s = df[df['stimulus_ue']==str(stim)]
                df_s['segment_part'] = df_s['segment_part'].replace(np.nan, 'None')
                df_s['segment_arm'] = df_s['segment_arm'].replace(np.nan, 'None')

                ID_lst.append(ID)
                dim_lst.append(dim)
                cond_lst.append(cond)
                stim_lst.append(stim)

                # Controller response and reaction time
                click_idx = df_s.index[-1]
                response = click_response(df, click_idx)
                RT = df_s['rel_time'].iloc[-1]
                repsonse_lst.append(response)
                RT_lst.append(RT)

                #### Pupil diameter ####
                pupil_dia_lst.append(np.nanmean(df_s['combined.pupil_diameter_corr']))

                # Pupil diameter amplitude - differences between pupil diameter min/max of 10/90% of quantile
                q_low = df_s["combined.pupil_diameter_corr"].quantile(0.1)
                q_hi = df_s["combined.pupil_diameter_corr"].quantile(0.9)

                df_quantiles = df_s[(df_s["combined.pupil_diameter_corr"] < q_hi) &
                                    (df_s["combined.pupil_diameter_corr"] > q_low)]

                pupil_min = np.min(df_quantiles['combined.pupil_diameter_corr'])
                pupil_max = np.max(df_quantiles['combined.pupil_diameter_corr'])
                pupil_dia_ampl_lst.append(pupil_max - pupil_min)

                # Select unique gaze labels
                unique_gaze_labels = df_s['gaze_label_number'].unique()

                ###################
                #### Fixations ####
                ###################
                fix_dur_lst = list()

                left_dur = 0
                right_dur = 0
                inner_dur_left = 0
                inner_dur_right = 0
                outer_dur_left  = 0
                outer_dur_right = 0

                fixations = [fix for fix in unique_gaze_labels if fix.startswith('fix')]

                for fix in fixations:
                    df_fix = df_s[df_s['gaze_label_number'] == fix]

                    fix_dur = df_fix['label_duration'].iloc[0]
                    fix_dur_lst.append(fix_dur)

                    side = h.most_frequent(list(df_fix['2Dside'].values))
                    if side == 'left':
                        left_dur += fix_dur

                    if side == 'right':
                        right_dur += fix_dur

                    segment_part = df_fix['segment_part'].iloc[0]
                    if segment_part == 'inner':
                        if side == 'left':
                            inner_dur_left += fix_dur
                        if side == 'right':
                            inner_dur_right += fix_dur

                    if segment_part == 'outer':
                        if side == 'left':
                            outer_dur_left += fix_dur
                        if side == 'right':
                            outer_dur_right += fix_dur

                ## calculate avg fixation durations
                mean_fix_dur_lst.append(np.nanmean(fix_dur_lst))

                ## Calculate fixation time ratio between inner and outer figure segments
                ratio_left = np.nan
                ratio_right = np.nan
                if np.logical_or(inner_dur_left > 0, outer_dur_left > 0):
                    ratio_left = 1 - (np.abs(outer_dur_left / (inner_dur_left + outer_dur_left) -0.5) * 2)

                if np.logical_or(inner_dur_right > 0, outer_dur_right > 0):
                    ratio_right = 1 - (np.abs(outer_dur_right / (inner_dur_right + outer_dur_right) -0.5) * 2)

                if np.logical_and(np.isnan(ratio_left),np.isnan(ratio_right)):
                    ratio_inner_outer_dur_lst.append(np.nan)

                if np.logical_or(ratio_left>=0, ratio_right>=0):
                    ratio_inner_outer_dur_lst.append(np.nanmean([ratio_left,ratio_right]))

                ## Calculate fixation time ratio between left and right figure
                if np.logical_or(left_dur > 0, right_dur > 0):
                    ratio_left_right_dur_lst.append(1 - (np.abs(left_dur / (left_dur + right_dur) - 0.5) * 2 ))
                if np.logical_and(left_dur == 0, right_dur == 0):
                    ratio_left_right_dur_lst.append(np.nan)

                ## Regressive Fixation Duration
                fixations = [fix for fix in unique_gaze_labels if fix.startswith('fix')]

                if len(fixations) >= 3:
                    reg_fix_dur = list()
                    regression_steps = np.arange(0, len(fixations)-2)
                    for j in regression_steps:
                        df_fixt0 = df_s[df_s['gaze_label_number'] == fixations[j]]
                        df_fixt2 = df_s[df_s['gaze_label_number'] == fixations[j+2]]

                        sidet0 = h.most_frequent(list(df_fixt0['2Dside'].values))
                        sidet2 = h.most_frequent(list(df_fixt2['2Dside'].values))

                        segment_partt0 = df_fixt0['segment_arm'].iloc[0]
                        segment_partt2 = df_fixt2['segment_arm'].iloc[0]

                        if segment_partt0 != 'None':
                            if np.logical_and(sidet0 == sidet2, segment_partt0 == segment_partt2):
                                fix_dur = df_fixt2['label_duration'].iloc[0]
                                reg_fix_dur.append(fix_dur)

                    if len(reg_fix_dur) > 0:
                        regressive_fix_dur_lst.append(np.nanmean(reg_fix_dur))
                    if len(reg_fix_dur) == 0:
                        regressive_fix_dur_lst.append(0)

                if len(fixations)<3:
                    regressive_fix_dur_lst.append(0)

                ## Strategy ratio
                fixations_l = fixations.copy()
                within_fix  = 0
                between_sacc = 1

                if len(fixations_l)>0:
                    current_fix = df_s[df_s['gaze_label_number'] == fixations_l[0]]
                    current_side = h.most_frequent(list(current_fix['2Dside'].values))
                    current_part = current_fix['segment_part'].iloc[0]

                    while np.logical_and(current_part == 'None', len(fixations_l)>1):
                        fixations_l.pop(0)
                        current_fix = df_s[df_s['gaze_label_number'] == fixations_l[0]]
                        current_part = current_fix['segment_part'].iloc[0]

                    if len(fixations_l)>1:
                        for fix in fixations_l[1:]:
                            df_fix = df_s[df_s['gaze_label_number'] == fix]
                            this_part = df_fix['segment_part'].iloc[0]
                            this_side = h.most_frequent(list(df_fix['2Dside'].values))

                            if np.logical_and(this_part != 'None', current_part != 'None'):
                                if current_side == this_side:
                                    within_fix +=1
                                if current_side != this_side:
                                    between_sacc +=1

                            current_side = h.most_frequent(list(df_fix['2Dside'].values))
                            current_part = df_fix['segment_part'].iloc[0]

                        strat_ratio_lst.append(within_fix/between_sacc)

                if len(fixations_l)<=1:
                    strat_ratio_lst.append(np.nan)

                ##################
                #### Saccades ####
                ##################
                saccades = [sac for sac in unique_gaze_labels if sac.startswith('sac')]

                # Get saccades
                sacc_dur = list()
                sacc_velo = list()

                for sac in saccades:
                    idx_s = df_s[df_s['gaze_label_number'] == sac].index
                    df_sac = df_s[df_s['gaze_label_number'] == sac]


                    t_sacc = df_sac['label_duration'].iloc[0]
                    sacc_dur.append(t_sacc)

                    sacc_velo.append(np.nanmean(df_sac['gaze_angle_velo']))

                # Saccades duration and velocity
                avg_sacc_dur_lst.append(np.nanmean(sacc_dur))
                avg_sacc_velocity_lst.append(np.nanmean(sacc_velo))

                ##################
                #### Head Feat ###
                ##################
                # get head rotation velocity
                head_rot_speed_lst.append(np.nanmean(df_s['head_angle_velo']))

                # get head movement velocity
                head_loc_speed_lst.append(np.nanmean(df_s['head_loc_velo']))


        self.df_feature = pd.DataFrame({'ID': ID_lst, 'dimension': dim_lst, 'condition': cond_lst, 'stimulus': stim_lst,
                                       'RT':RT_lst, 'Response':repsonse_lst, 'Mean fixation duration':mean_fix_dur_lst,
                                       'Mean regressive fixation duration': regressive_fix_dur_lst,

                                       'Equal fixation duration between figures':ratio_left_right_dur_lst,
                                       'Equal fixation duration within figure':ratio_inner_outer_dur_lst,

                                       'Mean saccade duration': avg_sacc_dur_lst,
                                       'Mean saccade velocity': avg_sacc_velocity_lst,

                                       'Mean pupil diameter': pupil_dia_lst,
                                       'Pupil diameter amplitude': pupil_dia_ampl_lst,

                                       'Mean head rotation': head_rot_speed_lst,
                                       'Mean head movement': head_loc_speed_lst,

                                       'Strategy ratio': strat_ratio_lst,

                                       })

        # Add stimulus information and correct response variable
        #
        os.chdir(self.project_path + '\\meta\\')
        df_info = pd.read_csv('All_stimulus_information.csv')
        df_info_s = df_info[['stimulus', 'Equal?', 'AngularDisp', 'DiffType']]

        self.df_feature = self.df_feature.merge(df_info_s, on='stimulus')
        self.df_feature.insert(4, 'Correct',
                          (((self.df_feature['Response'] + self.df_feature['Equal?']) % 2) + 1) % 2)

        print('')

    def get_feature_dataset(self):
        return self.df_f

    def save_feature_dataset(self, save_path):
        today = date.today()
        self.df_feature.to_csv(save_path + '{}_eye_features.csv'.format(today), index=False)
        print('Files saved to: ', save_path)


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