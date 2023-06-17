import os
from datetime import date
import numpy as np
import pandas as pd

import src.P101_experiment_preprocessing_data as P101
import src.P102_experiment_fixation_detection as P102
import src.P103_experiment_segment_detection as P103
import src.P104_experiment_eye_feature_calculation as P104

import src.P201_questionnaire_preprocessing_data as P201

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
    #feature.save_feature_dataset(data_path + '\\6_feature_dataset\\')

    ####################################################################################################################
    ## Statistical Analysis
    stats = P301.Statistics(data_path)

    ### Preprocess questionnaire ###
    ####################################################################################################################
    #prep_quest = P201.Preprocessing(data_path)
    #prep_quest.refactor_and_select_columns()
    #prep_quest.save_dataframe()

    print('')



