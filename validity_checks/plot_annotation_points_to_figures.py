from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from pathlib import Path

import os
import glob
import pandas as pd

if __name__ == '__main__':
    val_path = os.path.abspath(os.getcwd())
    project_path = Path(val_path).parent
    data_path = os.path.abspath(project_path) + '\\data\\annotation_dataframes\\'

    df_info = pd.read_csv(data_path + 'full_stim_info.csv')

    os.chdir(data_path)
    stim_lst = glob.glob("E*.csv")
    stimuli = list()
    print("Number of files: ", len(stim_lst))
    for i in df_info['index']:
        print(df_info['name'].iloc[i])
        print(stim_lst[i])
        stimuli.append(pd.read_csv(stim_lst[i], sep=',', header=0, index_col=False, low_memory=False))

    poly_lst = list()
    for i in df_info['index'].unique():
    
        info = df_info[df_info['index'] == i].astype(str)
        name = info['name'].values[0].split('.')[0]
        stim = info['stimulus'].values[0]
        print(name)
        print(stim)
    
        stimulus = stimuli[i].copy()
    
        stim1 = stimulus[stimulus['figure'] == 'left']
        stim2 = stimulus[stimulus['figure'] == 'right']
    
        cubes1 = stim1['cube'].unique()
        cubes2 = stim2['cube'].unique()
    
        lst = list()
        cube_name_lst = list()
    
        for cube in cubes1:
            x1 = stim1[stim1['cube'] == cube]
            x1 = x1[x1['figure'] == 'left']
            x1 = x1[['x', 'y']].to_numpy()
            lst.append(x1)
            cube_name_lst.append(cube + 'left')
    
        for cube in cubes2:
            x2 = stim2[stim2['cube'] == cube]
            x2 = x2[x2['figure'] == 'right']
            x2 = x2[['x', 'y']].to_numpy()
            lst.append(x2)
            cube_name_lst.append(cube + ' right')
    
        fig = plt.figure()
        index = 0
        for p in lst:
            # define polygon based on points of a cube
    
            poly = Polygon(p)
    
            x, y = poly.exterior.xy
    
            for i in p:
                plt.plot(i[0], i[1], '+', color='black')
    
            plt.plot(x, y, label=cube_name_lst[index])
            plt.xlim(0, 1200)
            plt.ylim(0, 800)
            index += 1
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(val_path + '\\annotation_plots\\Stim{}_{}.png'.format(stim, name), bbox_extra_artists=(lgd,), bbox_inches='tight')
    
        #plt.show()
    
        plt.close(fig)
