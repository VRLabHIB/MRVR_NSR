import os
from datetime import datetime
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np
import glob

import src.helper as h

class CubeMidpoints2D:
    def __init__(self, data_path):
        os.chdir(data_path + '\\annotation_dataframes\\')
        self.data_path = data_path
        self.stim_lst = glob.glob("E*.csv")
        self.stimuli = list()
        self.df_info = pd.read_csv(data_path + '\\annotation_dataframes\\full_stim_info.csv')

        print("Number of files: ", len(self.stim_lst))
        for i in self.df_info['index']:
            print(self.df_info['name'].iloc[i])
            print(self.stim_lst[i])
            self.stimuli.append(pd.read_csv(self.stim_lst[i], sep=',', header=0, index_col=False, low_memory=False))


    def find_cube_midpoints_2D(self):
        #clean cube names
        for i in range(len(self.stim_lst)):
            stim = self.stimuli[i]
            cube_clean_lst = list()

            for j in range(len(stim)):
                cube = stim['cube'].iloc[j]
                cube = cube.replace('A', '')
                cube = cube.replace('B', '')
                cube = cube.replace('C', '')
                cube = cube.replace('D', '')
                cube_clean_lst.append(cube)

            stim['cube_c'] = cube_clean_lst
            self.stim_lst[i] = stim

        # calculate midpoints
        side_lst = list()
        cube_lst = list()
        cube_c_lst = list()
        stim_numbers = list()
        x_lst = list()
        y_lst = list()

        for i in range(len(self.df_info)):
            index = int(self.df_info['index'].iloc[i])
            name = self.df_info['name'].iloc[i]
            stimulus = self.df_info['stimulus'].iloc[i]

            print('Name: ', name)
            print('Stimulus: ',stimulus)

            df = self.stim_lst[index]

            df_left = df[df['figure'] == 'left']
            df_right = df[df['figure'] == 'right']

            for cube in df_left['cube'].unique():
                xy_cube = df_left[df_left['cube'] == cube][['x', 'y']].to_numpy()
                poly = Polygon(xy_cube)
                p = poly.centroid
                xy = p.coords.xy
                x_lst.append(xy[0][0])
                y_lst.append(xy[1][0])
                cube_lst.append(cube)
                cube_c_lst.append(df_left[df_left['cube'] == cube]['cube_c'].iloc[0])
                side_lst.append('left')
                stim_numbers.append(stimulus)

            for cube in df_right['cube'].unique():
                xy_cube = df_right[df_right['cube'] == cube][['x', 'y']].to_numpy()
                poly = Polygon(xy_cube)
                p = poly.centroid
                xy = p.coords.xy
                x_lst.append(xy[0][0])
                y_lst.append(xy[1][0])
                cube_lst.append(cube)
                cube_c_lst.append(df_right[df_right['cube'] == cube]['cube_c'].iloc[0])
                side_lst.append('right')
                stim_numbers.append(stimulus)

        df_midpoints = pd.DataFrame({'stimulus': stim_numbers, 'side': side_lst, 'cube': cube_lst, 'cube_c': cube_c_lst,
                                     'x': x_lst, 'y': y_lst})

        # Some cubes have more than one polygon describing its surface due to perspective
        # Merge midpoints for the same cubes, if it has more than 1
        stim_numbers = list()
        side_lst = list()
        cube_lst = list()
        x_lst = list()
        y_lst = list()

        for stim in df_midpoints['stimulus'].unique():
            df_stim = df_midpoints[df_midpoints['stimulus'] == stim]

            for side in df_midpoints['side'].unique():
                df_side = df_stim[df_stim['side'] == side]

                for cube in df_side['cube_c'].unique():
                    df_cube = df_side[df_side['cube_c'] == cube]

                    xy = df_cube[['x', 'y']]
                    x, y = np.mean(xy, axis=0)

                    stim_numbers.append(stim)
                    side_lst.append(side)
                    cube_lst.append(cube)
                    x_lst.append(x)
                    y_lst.append(y)

        df_mid_agg = pd.DataFrame(
            {'stimulus': stim_numbers, 'side': side_lst, 'cube': cube_lst, 'x': x_lst, 'y': y_lst})

        df_m1 = df_mid_agg[df_mid_agg['side'] == 'left']
        df_m2 = df_mid_agg[df_mid_agg['side'] == 'right']

        df_m1 = df_m1.merge(self.df_info[['stimulus', 'left_figure']], on='stimulus')
        df_m2 = df_m2.merge(self.df_info[['stimulus', 'right_figure']], on='stimulus')

        df_mid_final = pd.concat([df_m1, df_m2])
        df_mid_final['figure'] = df_mid_final['left_figure'].fillna(df_mid_final['right_figure'])
        df_mid_final = df_mid_final.drop(columns=['left_figure', 'right_figure'])

        df_mid_final.to_csv(self.data_path + '\\meta\\2Dcube_locations.csv', index=False)

    def align_both_cube_files(self):
        cube_loc_3D = pd.read_csv(self.data_path + '\\meta\\3Dcube_locations.csv', sep=',')
        cube_loc_2D = pd.read_csv(self.data_path + '\\meta\\2Dcube_locations.csv', sep=',')

        for i in range(len(cube_loc_2D)):
            cube_loc_2D['cube'].iloc[i] = cube_loc_2D['cube'].iloc[i].replace('cube', '')

        cube_loc_3D = cube_loc_3D.rename(columns={'x': 'drop', 'y': 'x', 'z': 'y'})
        cube_loc_3D = cube_loc_3D.drop(columns=['drop'])

        for i in range(len(cube_loc_3D)):
            fig = cube_loc_3D['figure'].iloc[i]

            fig_int = int(fig.split('M')[1])
            if fig_int < 10:
                cube_loc_3D['figure'].iloc[i] = 'M0' + str(fig_int)
            if fig_int == 10:
                cube_loc_3D['figure'].iloc[i] = 'M10'

        cube_loc_2D = cube_loc_2D[['stimulus', 'side', 'figure', 'cube', 'x', 'y']]

        # transform 2D pixel into UE coordinates

        # pivot position of screen X = -34.264, Y = 85.309
        # screen size PX: X = 1200px, Y = 800px
        # screen size UE: X = 84ue, Y = 56ue
        # ratio = 14.28571428571429
        x = (cube_loc_2D['x'] / 14.28571428571429) - 34.264
        y = (cube_loc_2D['x'] / 14.28571428571429) + 85.309

        cube_loc_2D['x'] = x
        cube_loc_2D['y'] = y

        cube_loc_3D.to_csv(self.data_path + '\\meta\\3Dcube_locations.csv', sep=',', index= False)
        cube_loc_2D.to_csv(self.data_path + '\\meta\\2Dcube_locations.csv', sep=',', index= False)

    def calculate_figure_centroids(self):
        cube_loc_3D = pd.read_csv(self.data_path + '\\meta\\3Dcube_locations.csv', sep=',')
        cube_loc_2D = pd.read_csv(self.data_path + '\\meta\\2Dcube_locations.csv', sep=',')

        cube_loc_2D['mid_x'] = np.nan
        cube_loc_2D['mid_y'] = np.nan
        cube_loc_2D['max_dist'] = np.nan
        cube_loc_2D['radius'] =np.nan

        cube_loc_3D['mid_x'] = np.nan
        cube_loc_3D['mid_y'] = np.nan
        cube_loc_3D['max_dist'] = np.nan
        cube_loc_3D['radius'] =np.nan

        stim_lst = np.arange(1,31)
        side_lst = ['left', 'right']

        for stim in stim_lst:
            c2 = cube_loc_2D[cube_loc_2D['stimulus'] == stim]
            c3 = cube_loc_3D[cube_loc_3D['stimulus'] == stim]

            # left side
            c2sl = c2[c2['side'] == 'left']
            c3sl = c3[c3['side'] == 'left']

            xm2l, ym2l, dist2l = h.calculate_centroid(c2sl['x'].values, c2sl['y'].values)
            xm3l, ym3l, dist3l = h.calculate_centroid(c3sl['x'].values, c3sl['y'].values)

            cube_loc_2D['mid_x'].iloc[c2sl.index] = xm2l
            cube_loc_2D['mid_y'].iloc[c2sl.index] = ym2l
            cube_loc_2D['max_dist'].iloc[c2sl.index] = dist2l

            cube_loc_3D['mid_x'].iloc[c3sl.index] = xm3l
            cube_loc_3D['mid_y'].iloc[c3sl.index] = ym3l
            cube_loc_3D['max_dist'].iloc[c3sl.index] = dist3l

            # right side
            c2sr = c2[c2['side'] == 'right']
            c3sr = c3[c3['side'] == 'right']

            xm2r, ym2r, dist2r = h.calculate_centroid(c2sr['x'].values, c2sr['y'].values)
            xm3r, ym3r, dist3r = h.calculate_centroid(c3sr['x'].values, c3sr['y'].values)

            cube_loc_2D['mid_x'].iloc[c2sr.index] = xm2r
            cube_loc_2D['mid_y'].iloc[c2sr.index] = ym2r
            cube_loc_2D['max_dist'].iloc[c2sr.index] = dist2r

            cube_loc_3D['mid_x'].iloc[c3sr.index] = xm3r
            cube_loc_3D['mid_y'].iloc[c3sr.index] = ym3r
            cube_loc_3D['max_dist'].iloc[c3sr.index] = dist3r

            # calculate radius
            xm2mid, ym2mid, __ = h.calculate_centroid([xm2l,xm2r],[ym2l,ym2r])
            xm3mid, ym3mid, __= h.calculate_centroid([xm3l, xm3r], [ym3l, ym3r])

            leftr2 = np.sqrt((xm2l-xm2mid)**2+(ym2l-ym2mid)**2)
            rightr2 = np.sqrt((xm2r - xm2mid) ** 2 + (ym2r - ym2mid) ** 2)
            cube_loc_2D['radius'].iloc[c2sl.index] = leftr2
            cube_loc_2D['radius'].iloc[c2sr.index] = rightr2

            leftr3 = np.sqrt((xm3l-xm3mid)**2+(ym3l-ym3mid)**2)
            rightr3 = np.sqrt((xm3r - xm3mid) ** 2 + (ym3r - ym3mid) ** 2)
            cube_loc_3D['radius'].iloc[c3sl.index] = leftr3
            cube_loc_3D['radius'].iloc[c3sr.index] = rightr3

        cube_loc_3D.to_csv(self.data_path + '\\meta\\3Dcube_locations.csv', sep=',', index=False)
        cube_loc_2D.to_csv(self.data_path + '\\meta\\2Dcube_locations.csv', sep=',', index=False)


class Segments:
    def __init__(self, data_path, location):
        self.data_path = data_path
        os.chdir(data_path + location)
        self.data_lst = glob.glob("ID*.csv")
        print("Number of files: ", len(self.data_lst))
        self.dataframes = [pd.read_csv(f, sep=',', header=0, index_col=False, low_memory=False) for f in self.data_lst]

        self.d1 = dict({'M01': {'outer': [1, 2, 3, 9, 10], 'inner': [4, 5, 6, 7, 8]},
                       'M02': {'outer': [1, 2, 3, 9, 10], 'inner': [4, 5, 6, 7, 8]},
                       'M03': {'outer': [1, 2, 3, 9, 10], 'inner': [4, 5, 6, 7, 8]},
                       'M04': {'outer': [1, 2, 3, 9, 10], 'inner': [4, 5, 6, 7, 8]},
                       'M05': {'outer': [1, 2, 3, 8, 9, 10], 'inner': [4, 5, 6, 7]},
                       'M06': {'outer': [1, 2, 3, 8, 9, 10], 'inner': [4, 5, 6, 7]},
                       'M07': {'outer': [1, 2, 3, 8, 9, 10], 'inner': [4, 5, 6, 7]},
                       'M08': {'outer': [1, 2, 3, 8, 9, 10], 'inner': [4, 5, 6, 7]},
                       'M09': {'outer': [1, 2, 3, 8, 9, 10], 'inner': [4, 5, 6, 7]},
                       'M10': {'outer': [1, 2, 3, 8, 9, 10], 'inner': [4, 5, 6, 7]}
                       })

        self.d2 = dict({'M01': {'A': [1, 2, 3], 'B': [4, 5, 6, 7, 8], 'C':[9, 10]},
                        'M02': {'A': [1, 2, 3], 'B': [4, 5, 6, 7, 8], 'C':[9, 10]},
                        'M03': {'A': [1, 2, 3], 'B': [4, 5, 6, 7, 8], 'C':[9, 10]},
                        'M04': {'A': [1, 2, 3], 'B': [4, 5, 6, 7, 8], 'C':[9, 10]},
                        'M05': {'A': [1, 2, 3], 'B': [4, 5, 6, 7], 'C': [8, 9, 10]},
                        'M06': {'A': [1, 2, 3], 'B': [4, 5, 6, 7], 'C': [8, 9, 10]},
                        'M07': {'A': [1, 2, 3], 'B': [4, 5, 6, 7], 'C': [8, 9, 10]},
                        'M08': {'A': [1, 2, 3], 'B': [4, 5, 6, 7], 'C': [8, 9, 10]},
                        'M09': {'A': [1, 2, 3], 'B': [4, 5, 6, 7], 'C': [8, 9, 10]},
                        'M10': {'A': [1, 2, 3], 'B': [4, 5, 6, 7], 'C': [8, 9, 10]}
                       })
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

    def identify_closest_segment(self):
        cube_loc_3D = pd.read_csv(self.data_path + '\\meta\\3Dcube_locations.csv', sep=',')
        cube_loc_2D = pd.read_csv(self.data_path + '\\meta\\2Dcube_locations.csv', sep=',')

        for file in range(len(self.data_lst)):
            name = self.data_lst[file]
            print(name)
            df = self.dataframes[file]
            dim = df['dimension'].iloc[0]

            if dim == 2:
                df_cube_loc = cube_loc_2D.copy()
            if dim == 3:
                df_cube_loc = cube_loc_3D.copy()

            df['segment_part'] = 'None'
            df['segment_arm'] = 'None'

            stim_lst = np.arange(1,31)
            for stim in stim_lst:
                df_s = df[np.logical_and(df['stimulus']==stim,df['onset']==1)]

                cube_s = df_cube_loc[df_cube_loc['stimulus'] == stim].copy()
                cube_s = cube_s.reset_index()

                fix_lst = df_s[df_s['gaze_label_number'].str.startswith('fix')]['gaze_label_number'].unique()
                for fix in fix_lst:
                    df_f = df_s[df_s['gaze_label_number']==fix]
                    index = df_f.index

                    side = h.most_frequent(list(df_f['2Dside'].values))

                    cube_ss = cube_s[cube_s['side']==side].copy()
                    midpoints = np.array([cube_ss['x'].values, cube_ss['y'].values]).T
                    radius = cube_ss['radius'].iloc[0]
                    figure = cube_ss['figure'].iloc[0]

                    fix_mid_vec = df_f[['fixation_midpointX','fixation_midpointY']].iloc[0].to_numpy()
                    fig_mid = cube_ss[['mid_x', 'mid_y']].iloc[0].to_numpy()

                    cube_ss['cube_dist'] = np.linalg.norm(fix_mid_vec - midpoints, axis=1)
                    cubes = cube_ss.nsmallest(3, 'cube_dist')
                    cubes = cubes[~cubes['cube_dist'].isna()]['cube'].values

                    dict1 = self.d1[figure]
                    dict2 = self.d2[figure]
                    seg_part, seg_arm = get_segment(cubes, dict1, dict2)

                    # erase segment detections outside the figure radius
                    distance = (np.sqrt((fix_mid_vec[0]-fig_mid[0])**2 + (fix_mid_vec[1]-fig_mid[1])**2))
                    if distance>= radius:
                        seg_part = 'None'
                        seg_arm = 'None'

                    # check if fixations are always on one side, and if not what the segment part is
                    if len(set(df_f['2Dside'].tolist())) != 1:
                        print('not same side')
                        print(seg_part)
                        print(distance)

                    df['segment_part'].iloc[index] = seg_part
                    df['segment_arm'].iloc[index] = seg_arm

            self.dataframes[file] = df

def get_segment(cubes, dict1, dict2):
    inner = dict1['inner']
    outer = dict1['outer']

    inner_lst = list(set(cubes) & set(inner))
    outer_lst = list(set(cubes) & set(outer))

    if len(inner_lst) < len(outer_lst):
        seg_part = 'outer'
    if len(inner_lst) > len(outer_lst):
        seg_part = 'inner'

    a = dict2['A']
    b = dict2['B']
    c = dict2['C']

    arm_lst = ['A','B','C']
    a_lst = list(set(cubes) & set(a))
    b_lst = list(set(cubes) & set(b))
    c_lst = list(set(cubes) & set(c))

    lst = [len(a_lst), len(b_lst), len(c_lst)]
    idx = lst.index(max(lst))

    return seg_part, arm_lst[idx]
