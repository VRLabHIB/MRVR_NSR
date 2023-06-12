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

**calculate_and_process_variables**
- blink detection, missing data and combined pupil diameter:
	- using missing data in gaze directon and gaze origin to mark missing data
	- Since blinks usually not longer than 500ms, only shorter intervals are considered as blinks. Blinks only neeeded to be dectected to correct for artifacts and outliers around the blink event.
		- mask 1 additional datapoint, representing at least 10 ms around the blink as missing (Ref: https://doi.org/10.3758/s13428-022-01957-7)
	- calculate a combined pupil diameter as the arthmetic mean using the pupil diameter variables of both eyes. 
		- For each row, only a mean is caluclated if both variables contain valid values. 
		
- baseline correction pupil diameter:
- 

- For 3D condition: Calculate 2D gaze points at an immaginary surface at the position of the screen in the 2D condition. 

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





 