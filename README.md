# EMEG-preprocessing

This script is used for pre-processing of (E)MEG data.
Version 2.4.0 (last updated by Yi Zhang (9 Nov 2023), zhang.yi-93@outlook.com)

The script sanitise_input.py should be run MANUALLY before the pre-processing.
The information on bad EEG and MEG channels should be recorded in "badEEG.csv" and "badMEG.csv" in the same directory as the script.

NOTE: Do remember to edit the event_trig() function to define the event onsets.
If you are dealing with resting-state, please do include "rs" in the task_name.

the input data should be organized as:

                                    |---/MRI/Freesurfer preprocessed MRI files
                 |---'subject 1'--- |---/MEG/task_name/raw_file_name in fif format
                                    |---/MRI/Freesurfer preprocessed MRI files
                 |---'subject 2'--- |---/MEG/task_name/raw_file_name in fif format
                                    |---/MRI/Freesurfer preprocessed MRI files
    'proj_dir'-- |---'.........'--- |---/MEG/task_name/raw_file_name in fif format
                                    |---/MRI/Freesurfer preprocessed MRI files
                 |---'subject n'--- |---/MEG/task_name/raw_file_name in fif format

The preprocessed data will be organized by the program as:

                                 |---/01_head_motion/   file created by head_motion() function
                                 |---/02_filter/        file created by filt() function
                                 |---/03_ICA_denoise/   ICA denoised files
                                 |---/04_sensor_prep/   sensor space based time series
                                 |---/05_fwd_model/     forward models
                                 |---/06_src_stc/       source space based time series (or ROI-based time series if required)
    'subj_dir' ---- |--- MEG --- |---/event/task MEG related files
                                 |---/src/volume and surface based source spaces in -src.fif format
                                 |---/fig/figures created through pre-processing                             
                    |--- MRI --- |---other Freesurfer preprocessed MRI files

update history:

    (02 Feb, 2023, v 1.0.0, by Yi Zhang) A primary version used for MILOS object task pre-processing.
    
    (12 Feb, 2023, v 1.1.0, by Yi Zhang) MRIpostprep.py file created, for registering timeseries to standard space.
    
    (14 Feb, 2023, v 2.0.0, by Yi Zhang) A modified version used for both resting and task data pre-processing. Some folder naming rules were changed.
    
    (23 Feb, 2023, v 2.0.1, by Yi Zhang) Add a separate file for defining parameters (folder names, etc).
    
    (24 Feb, 2023, v 2.0.2, by Yi Zhang) Lower the resolution(voxel 3mm -> 4mm, surface ico5 -> oct6 (3.1mm -> 4.9m)), and separate timeseries to several files.
    
    (09 Sep, 2023, v 2.1.0, by Yi Zhang) Allowing skip certain steps in the pre-processing. Capable of extracting ROI-based time series. Allowing the case where EOG/ECG channel is missing.
    
    (11 Sep, 2023, v 2.1.1, by Yi Zhang) Revise some minor problems in the script.
    
    (25 Sep, 2023, v 2.2.0, by Yi Zhang) Allowing adding empty room projectors.
    
    (10 Oct, 2023, v 2.2.1, by Yi Zhang) Repair minor mistakes.
    
    (16 Oct, 2023, v 2.3.0, by Yi Zhang) Allow SSP denoise.
    
    (9 Nov, 2023, v 2.4.0, by Yi Zhang) Allow only output surface/volume data, and modify some of the parameters. Temporarily delete all logics related to indicator files.

TODO:

    · parallel computing
    · task dictionary
    · logical judgement needed to be revised
    · putting log file into a separate file (i.e. revising folder logic)
