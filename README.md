# MRVR_NSR
 
 
## \data
**0_raw_experiment** experiement data data was 


## \src

### P001_preprocessing_experiment_data.py
Preprocessing is an object class holding all dataframes to processing. Getter functions are: get_data_lst() and get_dataframes()

**data_cleaning** 
- remove unnecessary variables
- trim datasets so that only the experiment intervals are included
- rename stimuli for 2D condition (reverse randomized stimuli order)
- insert variables necessary for further processing: onset, stimulus number and relative time

**process_pupil**
- blink detection, missing data and combined pupil diameter:
	- Using missing data in gaze directon and gaze origin to identify missing data
	- Since blinks are usually not longer than 500ms, only intervals up to 500ms were considered as blinks. 
	  We only neeeded to detect blinks to correct for artifacts and outliers around a blink event.
		- Masked 1 additional datapoint, representing at least 10 ms around the blink as missing (Ref: https://doi.org/10.3758/s13428-022-01957-7)
	- calculate a combined pupil diameter as the arthmetic mean using the pupil diameter variables of both eyes. 
		
- baseline correction pupil diameter: obtained individual baselines for each stimulus interval by caluclating the median
  over the 3 second countdown before each stimulus appeared. A substractiv baseline correction was performed using the medians.

**select_cases_on_tracking_ratio**
- Used the pupil diameter values of both eyes as an idicator of the Eye Trackers tracking ratio. Sessions with an average 
  tracking ratio below 80% during the experiment part were removed from the sample. 
- Since we always wanted to compare both conditions (2D and 3D) for each participant, sessions had to be removed in which only one of the two conditions showed a low tracking ratio.
- As a result 54 participants remained for the analysis, 12 had to be removed because either one or both conditions did not met the required threshold.  

**calculate_and_process_variables**
- 

- For 3D condition: Calculate 2D gaze points at an immaginary surface at the position of the screen in the 2D condition. 

### P102_experiment_fixation_and_gaze_target_detection.py

**ivt_fixation_detection**
- fixation detection:
	- mask blinks
	- IVT fixation detection 60°/s on 2D gaze points

To calculate features that required information gaze position on the figure, we had to apply further steps in the procedure: 
- Due to low accuracy and precision of the HMD eye tracker fixation midpoints missed the figure. Our approach ensures that fixation points with a small distance to the figure could be assigned to a corresponding part of the figure. 
- At the same time, we ensured that no fixations outside a circle around the figure were evaluated as fixation points on the figures. 
- With our apporach we were able to overcome low accuracy and low prcision in the eye tracker and control for the incorrect assignments of focal points to the figure. 

- Procedure:
	- calculate fixation midpoints
	- calculate figure midpoints and determine figure radius
	- extend figure radius and stop extension before both circle start intersecting
	- assign each gaze point to a respective figure
	- We calculated the distance between the fixation midpoints and the closest cube midpoint of the figure.
		- fixation midpoints are calculated as 2D points using #TODO
		- figure midpoints are calculated as #TODO
		- based on the closest cube, the fixation is assigned to the respective figure segment that contains this closest cube. 

### P201_questionnaire_preprocessing_data.py

- In the dataset from socey survey, we had 88 participants who filled out the questionnaire data. 22 questionnaires were from piloting the VR experiment and piloting the survey before the start of the valid experiment.
- 66 valid participants remained. 

# Results

- The results from comparing the 2D (M = , SD = ) and 3D (M = , SD = ) condition indicate an decrease in reaction time for the 3D condition 

- of the 54 participants 31 (57%) started with the 2D condition and 23 (43%) started with the 3D condition. 



 
