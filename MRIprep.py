"""
This script pre-process the MRI data for subjects, used for source localisation later.
The script should be ran under LIUNX, IOS or WSL.


"""

# os.system("su root") #run the freesurfer under root to avoid permission issues

import os
import mne

workingDir = "/mnt/s/MEG/fdu/"
subjID = os.listdir(workingDir)

"""
TODO:
Freesurfer Recon-all command
"""

for subj_id in subjID:
    subj_dir = workingDir + subj_id + "/MRI/"
    mne.bem.make_watershed_bem("", subjects_dir=subj_dir, overwrite=True, volume="T1", atlas=False, gcaatlas=False, preflood=None, show=False, copy=True, T1=None, brainmask="ws.mgz", verbose=None)

print("BEM construction completed.")


