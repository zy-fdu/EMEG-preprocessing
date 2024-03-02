
"""
This script was used for (manually) check the bad EEG and MEG channels, and recording them in the csv files
the duplicate channels (most of which are ECG/EOG) are also checked and recorded, so as to avoid errors in interpolation
the sanitised data would be suitable for the later automatic preprocessing.

the input data should be organized as:
                                |---/MRI/Freesurfer preprocessed MRI files
             |---'subject 1'--- |---/MEG/task_name/raw_file_name in fif format

                                |---/MRI/Freesurfer preprocessed MRI files
             |---'subject 2'--- |---/MEG/task_name/raw_file_name in fif format

                                |---/MRI/Freesurfer preprocessed MRI files
'proj_dir'-- |---'.........'--- |---/MEG/task_name/raw_file_name in fif format

                                |---/MRI/Freesurfer preprocessed MRI files
             |---'subject n'--- |---/MEG/task_name/raw_file_name in fif format

"""

import os
import mne
import matplotlib.pyplot as plt
subj_id = "MILOS1002"
proj_dir = "X:/data/MEG/milos/data pre-processing/raw/"
task_name = "or"
raw = mne.io.read_raw_fif(proj_dir + subj_id + "/MEG/" + task_name + "/obj_task_block_milos_raw.fif", preload=True, allow_maxshield=True)

"""
EEG and MEG checks
"""
# duplicate channels
mne.channels.make_eeg_layout(raw.info) # mute this line of code after checking for duplicate sensors
# EEG
raw.pick_types(meg=False, eeg=True, eog=True, ecg = True, exclude = []).plot(block=True)
# MEG
raw.pick_types(meg=True, eeg=False, eog=True, ecg = True, exclude = []).plot(block=True)


"""
MRI bem checks
"""
# check for BEM
subjID = os.listdir(proj_dir)
conductivity = (0.3, 0.006, 0.3)
error_list = []
for subj_id in subjID:
    mriDir = proj_dir + subj_id + "/MRI/"
    try:
        model = mne.make_bem_model("", ico=4, conductivity=conductivity, subjects_dir=mriDir) # running in a lower resolution for a quicker check
    except RuntimeError:
        error_list.append(subj_id)
        ## use blender to manually fix the bem
        coords, faces = mne.read_surface(mriDir + "/bem/inner_skull.surf")
        coords[0] *= 1.1
        mne.write_surface(mriDir + "/bem/inner_skull.obj", coords, faces, overwrite=True)

        # Also convert the outer skull surface.
        coords, faces = mne.read_surface(mriDir + "/bem/outer_skull.surf")
        mne.write_surface(mriDir + "/bem/outer_skull.obj", coords, faces, overwrite=True)        

print("Please manually fix the surface for the following subjects:")
for subj_id in error_list:
    print(subj_id, end=" | ")
print()
print("Please use Blender to fix the surfaces, and run the following codes to replace the BEM files with fixed ones")
print("please refer to https://www.youtube.com/watch?v=JBIaX7VaTZk for the usage of Blender")


"""
Please use Blender to modify those surface first
The following code manually replace the BEM files with fixed files
"""
subj_id = "MILOS1003" # please manually enter the ID for subjects whose surface was not constructed properly
mriDir = proj_dir + subj_id + "/MRI"
# Read the fixed surface
coords, faces = mne.read_surface(mriDir + "/bem/inner_skull_fixed.obj")
_, _, vol_info = mne.read_surface(mriDir + "/bem/inner_skull.surf", read_metadata=True)
mne.write_surface(mriDir + "/bem/inner_skull.surf", coords, faces, volume_info=vol_info, overwrite=True)
model = mne.make_bem_model("", ico=4, conductivity=conductivity, subjects_dir=mriDir)
bem_sol = mne.make_bem_solution(model)
