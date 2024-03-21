import os
from datetime import date
import numpy as np
import pandas as pd

import src.P101_experiment_preprocessing_data as P101
import src.P102_experiment_fixation_detection as P102
import src.P103_experiment_segment_detection as P103
import src.P104_experiment_eye_feature_calculation as P104

#import src.P201_questionnaire_preprocessing_data as P201

import src.P301_data_analysis_statistics as P301


if __name__ == '__main__':
    # set project and data path
    project_path = os.path.abspath(os.getcwd())
    data_path = project_path + "\\data\\"
    valid_path = project_path + '\\validity_checks\\'
    os.chdir(data_path)

    save_path = project_path + "\\results\\"

    #####################################
    #### Uncomment to run the pipeline!!!
    #####################################

    ### Preprocess experiment data ###
    ####################################################################################################################
    ## First Cleaning

    #prep_exp = P101.Preprocessing(data_path, location = '\\0_experiment_raw\\')
    #prep_exp.data_cleaning()
    #prep_exp.add_experiment_condition()

    #save_path_step1 = data_path + '\\1_experiment_first_cleaning\\'
    #prep_exp.save_dataframes(save_path_step0)

    ####################################################################################################################
    ## Process pupil and select valid cases

    #pupil_exp = P101.Preprocessing(data_path, location='\\1_experiment_first_cleaning\\')
    #pupil_exp.process_pupil()

    #df_tr = pupil_exp.get_tr_data()
    #today = date.today()
    #df_tr.to_csv(valid_path + '{}_tracking_ratio.csv'.format(today), index=False)

    ## Remove invalid cases
    #valid_path = data_path + '\\2_experiment_valid_cases\\'
    #invalid_path = data_path + '\\2_experiment_removed_cases\\'
    #pupil_exp.select_cases_on_tracking_ratio(valid_path, invalid_path)

    ####################################################################################################################
    ## Add necessary variables
    #add_var = P101.Preprocessing(data_path, location='\\2_experiment_valid_cases\\')
    #add_var.calculate_and_process_variables()

    #save_path_step3 = data_path + '\\3_experiment_preprocessed\\'
    #add_var.save_dataframes(save_path_step3)

    ####################################################################################################################
    ## Fixation Detection
    #fix_det = P102.FixationDetection(data_path, valid_path, location='\\3_experiment_preprocessed\\')
    #fix_det.run_fixation_detection('IVT')
    #fix_det.calculate_fixation_midpoint()

    #fix_det.save_dataframes(data_path + '\\4_experiment_eye_events_detected\\')

    # Validate ivt idt by correlation
    #fix_val = P102.FixationDetection(data_path, valid_path, location='\\4_experiment_eye_events_detected\\')
    #fix_val.compute_detection_correlation()

    ####################################################################################################################
    ## Calculate closest segment
    #mid_2D = P103.CubeMidpoints2D(data_path)
    #mid_2D.find_cube_midpoints_2D()
    #mid_2D.align_both_cube_files()
    #mid_2D.calculate_figure_centroids()

    #seg_det = P103.Segments(data_path,location='\\4_experiment_eye_events_detected\\')
    #seg_det.identify_closest_segment()
    #seg_det.save_dataframes(data_path + '\\5_experiment_segments_detected\\')

    ####################################################################################################################
    ## Calculate Feature
    #feature = P104.Features(data_path,project_path, location='\\5_experiment_segments_detected\\' )
    #feature.create_feature_dataset()
    #feature.create_feature_dataset_count_measures()
    #feature.save_feature_dataset(data_path + '\\6_feature_dataset\\')

    ####################################################################################################################
    ## Statistical Analysis
    #stats = P301.Statistics(data_path)

    # Adjust p-values
    from statsmodels.stats import multitest



    var = ['Mean fixation duration', 'Mean fixation rate', 'Mean.regressive.fixation.duration', 'Equal.fixation.duration.between.figures',
           'Equal.fixation.duration.within.figure','Strategy.ratio', 'Mean saccade velocity', 'Mean saccade rate',
           'Mean.pupil.diameter', 'Pupil.diameter.amplitude','Mean.head.rotation', 'Mean.head.movement', 'Mean distance to figure']

    df = pd.read_csv(data_path + '\\6_feature_dataset\\2024-03-21_final_feature_dataset.csv')
    df = df[~df['stimulus'].isin([21, 24])]

    vars2 = ['Mean fixation duration', 'Relative Number of Fixations', 'Mean regressive fixation duration',
       'Equal fixation duration between figures',
       'Equal fixation duration within figure',
       'Mean saccade velocity','Relative Number of Saccades', 'Mean pupil diameter',
       'Pupil diameter amplitude', 'Mean head rotation', 'Mean head movement',
       'Strategy ratio', 'Mean distance to figure']
    dfs = df[vars2]
    dfs= dfs.rename(columns={'Pupil diameter amplitude':'Peak pupil diameter',
                             'Relative Number of Fixations': 'Mean fixation rate',
                             'Relative Number of Saccades':'Mean saccade rate'})

    dfr = dfs.corr().round(3)
    dfr = dfr.where(np.tril(np.ones(dfr.shape), k=-1).astype(np.bool))
    dfr = dfr.round(3)
    dfr = dfr.replace(np.nan, '-')
    dfr = dfr.drop(columns='Mean distance to figure')
    dfr.columns = np.arange(1, 13)
    print(dfr.to_latex(float_format="{:0.2f}".format))

    pvalsT = [0.2990102, 0.0379395,  0.000000001356195,  0.0128448, 0.000000000000000000000000000001642871,
              0.000000009310765, 0.00000001520424, 0.0002637, 0.0000000000006931733,
              0.0000000000009913028, 0.000005181929, 0.00001861239, 0.0039939]
    pvalsW =[0.2630001, 0.0404394, 0.00000006100603, 0.0225066,0.0000000001768897,
             0.00000003406438, 0.000000541066, 0.0005206, 0.000000002175876,
             0.000000001347795,0.0000008813468, 0.000000005291752, 0.0000089]
    adjT = multitest.multipletests(pvalsT, alpha=0.05, method='bonferroni', maxiter=1, is_sorted=False, returnsorted=False)
    adjW = multitest.multipletests(pvalsW, alpha=0.05, method='bonferroni', maxiter=1, is_sorted=False,
                                   returnsorted=False)
    dfx = pd.DataFrame({'vars':var, 'pvalsT':adjT[1].round(3), 'pvalsW':adjW[1].round(3)})
    print(dfx.to_latex())
    print(adjT[1].round(3))





