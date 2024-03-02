"""
This script is used for pre-processing of (E)MEG data.
Version 2.4.0 (last updated by Yi Zhang (9 Nov, 2023), yz821@cam.ac.uk)

The script sanitise_input.py should be ran MANUALLY before the pre-processing.
The information of bad EEG and MEG channels should be recorded in
"badEEG.csv" and "badMEG.csv" in the same directory with the script.

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
(02 Feb, 2023, v 1.0.0) A primary version used for MILOS object task pre-processing.
(12 Feb, 2023, v 1.1.0) MRIpostprep.py file created, for registering timeseries to standard space.
(14 Feb, 2023, v 2.0.0) A modified version used for both resting and task data pre-processing.
                        Some folder naming rules were changed.
(23 Feb, 2023, v 2.0.1) Add a separate file for defining parameters (folder names, etc).
(24 Feb, 2023, v 2.0.2) Lower the resolution(voxel 3mm -> 4mm, surface ico5 -> oct6 (3.1mm -> 4.9m)), and separate timeseries to several files.
(09 Sep, 2023, v 2.1.0) Allowing skip certain steps in the pre-processing. Capable of extracting ROI-based time series. Allowing the case where EOG/ECG channel is missing.
(11 Sep, 2023, v 2.1.1) Revise some minor problems in the script.
(25 Sep, 2023, v 2.2.0) Allowing adding empty room projectors.
(10 Oct, 2023, v 2.2.1) Repair minor mistakes.
(16 Oct, 2023, v 2.3.0) Allow SSP denoise.
(9 Nov, 2023, v 2.4.0) Allow only output surface/volume data, modify some of the parameters. Temporarily delete all logics related to signal files.
TODO: parallel computing
TODO: task dictionary
TODO: logical judgement needed to be revised
TODO: putting log file into a separate file (i.e. revising folder logic)
"""

rec_dir = "S:/MEG/milos/preproc/"  # the directory where badEEG.csv and badMEG.csv were saved
proj_dir = "S:/MEG/milos/preproc/raw"  # the project directory, containing MEG folder and MRI folder

#duplicate_channels = []
duplicate_channels = ["EEG071", "EEG072", "EEG073", "EEG074"]
# channels in same position will cause problems in interpolation, needed to be identified first. They are probably EOG/ECG channels.


task_name = "or"
# event_dict = None
# using an empty event dictionary in resting-state

#event_dict = {"matching_response": 1, "mismatching_response": 2, "cue_image": 3, "face": 4, "place": 5, "tool": 6, "body": 7,}
event_dict = {"animal": 1,"non-animal": 2}
# if it is not resting-state data, the event name should be defined first.


filt_band = [1,150]  # lower and higher frequency band for filtering
# NOTE: as recommended by mne, the ICA is sensitive to low-frequency drift, so a cutoff of 1 Hz is required.
# SSP can afford a lower frequency

raw_filename = "/obj_task_block_milos_raw.fif"  # the name of input raw file
#NOTE: do include a backslash before filename

empty_room_file_dir = "S:/MEG/fdu/Edumund_proj_emptyroom/emptyroom.fif"

do_prep = {
    "create_emptyroom_proj": False,
    "headmotion_and_filter": True, # headmotion and filter have to be skipped together as Maxfilter require head position
    "denoise": True,
    "BEM": True, # if rerun or run the program in other tasks, this term (only MRI) could be set as False to skip that
    "event": True, # if the data is resting-state, this part will be skipped automatically
    "source_localisation": True,
    "ROI_parcellation": True, # if do not wish to parcellate to ROI time series, set it as False. And only run this part if output_type of "surface" is False
}
# a parameter set whether to ignore the pre-processing procedure if some of them have already been done.
# if it is set as True, this part of pre-processing will be done, otherwise, will be skipped.


denoise_method = {
    "SSP": True,
    "ICA": True,
}
# TODO:  Do edit this part before running the program!!! All some out of expectation cuttings will be done.
cutoff_ts = None
# cutoff_ts = "/OR_cutoff.csv" # cutoff file
# NOTE: do remember to switch the cutoff_ts to None or manual edit the csv file if rerun the program!
# normally we do not need to set the parameter here.

data_type = {
    "EEG": True,
    "MEG": True,
}
# The MEG should always be true here, but EEG could be changed according to real situation
# indicator of whether the data contain EEG at the same time.

events_manual = False
# whether the time of onset of events need to be defined manually

output_type = {
    "surface": True,
    "volume":  False,
}

"""
Use command "python EMEG_datapreprocessing.py" in the terminal to execute the program after setting those parameters.
"""
