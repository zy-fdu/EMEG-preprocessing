import os
import numpy as np
import seaborn as sns
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import mne
from copy import deepcopy
import nibabel as nib
import scipy
import csv
from autoreject import get_rejection_threshold
import shutil
import joblib
import datetime
from EMEG_user_define_parameters import (
    rec_dir,
    proj_dir,
    duplicate_channels,
    task_name,
    event_dict,
    filt_band,
    raw_filename,
    empty_room_file_dir,
    do_prep,
    denoise_method,
    cutoff_ts,
    data_type,
    events_manual,
    output_type,
)


def main():

    if cutoff_ts != None:
        d = pd.read_csv(rec_dir + cutoff_ts)
        init_t = list(d[subj_id])[0]
        if init_t != 0:
            raw = mne.io.read_raw_fif(
                megDir + raw_filename, preload=True, allow_maxshield=True
            )
            raw = raw.crop(tmin=init_t)
            raw.save(megDir + raw_filename, overwrite=True)
        del raw

    if do_prep["create_emptyroom_proj"] == True:
        raw = mne.io.read_raw_fif(
            megDir + raw_filename, preload=True, allow_maxshield=True
        )
        empty_room_file = mne.io.read_raw_fif(empty_room_file_dir)
        empty_room_file.del_proj()
        empty_room_proj = mne.compute_proj_raw(empty_room_file, n_grad=3, n_mag=3)
        raw.add_proj(empty_room_proj, remove_existing=True)
        raw.save(megDir + raw_filename, overwrite=True)
        del raw

    if do_prep["headmotion_and_filter"] == True:
        head_motion()
    else:
        print("Head position already calculated, skip the process.")

    if do_prep["headmotion_and_filter"] == True:
        filt()  # interpolate bad channels and filter
    else:
        if not os.path.isdir(megDir + "/02_filter"):
            os.mkdir(megDir + "/02_filter")
        if not os.path.exists(megDir + "/02_filter/Maxfiltered_raw.fif"):
            shutil.copyfile(
                megDir + raw_filename, megDir + "/02_filter/Maxfiltered_raw.fif"
            )
        print("Bad channel processed, skip the process.")

    if do_prep["denoise"] == True:
        denoise()
    else:
        print("Artefacts cleaned, skip the process.")

    if not os.path.exists(mriDir + "/BEM_log.txt") or (do_prep["BEM"] == True):
        bem_src()  # create the BEM and setup source space for subject from freesurfer-processed structural image.
    else:
        print("Source space constructed, skipping the procedure")

    if (task_name.find("rs") != 0) and (do_prep["event"] == True):
        event_trig()
    elif task_name.find("rs") != 0:
        print("Event file already created, skip the process.")
    else:
        print("Resting-state data, skip the task-related process.")

    if do_prep["source_localisation"] == True:
        src_loc()  # source localisation
        with open(megDir + "/EMEG_prep_log.txt", "w") as f:
            f.write("subject has been successfully preprocessed!")
        if os.path.exists(megDir + "/01_flag.txt"):
            os.remove(megDir + "/01_flag.txt")
        if os.path.exists(megDir + "/02_flag.txt"):
            os.remove(megDir + "/02_flag.txt")
        if os.path.exists(megDir + "/03_flag.txt"):
            os.remove(megDir + "/03_flag.txt")
        if os.path.exists(megDir + "/event_flag.txt"):
            os.remove(megDir + "/event_flag.txt")
    else:
        print("Source time series has been constructed, skipping the procedure")

    if do_prep["ROI_parcellation"] == True:
        ROI_parc()
    print("EMEG pre-processing finished!")


def head_motion():

    if not os.path.isdir(megDir + "/01_head_motion"):
        os.mkdir(megDir + "/01_head_motion")

    raw = mne.io.read_raw_fif(megDir + raw_filename, preload=True, allow_maxshield=True)
    # extract the HPI coil amplitudes as a function of time
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
    # Compute time-varying HPI coil locations from these:
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
    # Compute head positions from the coil locations:
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs)
    # Visualise head positions
    head_pos_fig = mne.viz.plot_head_positions(
        head_pos,
        mode="traces",
        destination=raw.info["dev_head_t"],
        info=raw.info,
        show=False,
    )

    head_pos_fig.savefig(megDir + "/01_head_motion/head_pos.tif")
    # save files and record
    mne.chpi.write_head_pos(megDir + "/01_head_motion/head_pos.pos", head_pos)
    with open(megDir + "/01_flag.txt", "w") as f:
        f.write("head_motion calculated.")


def filt():

    if not os.path.isdir(megDir + "/02_filter"):
        os.mkdir(megDir + "/02_filter")

    raw = mne.io.read_raw_fif(megDir + raw_filename, preload=True, allow_maxshield=True)
    if data_type["EEG"] == True:
        raw_s = raw.pick_types(
            meg=True, eeg=True, eog=True, ecg=True, exclude=duplicate_channels
        )
    else:
        raw_s = raw.pick_types(
            meg=True, eeg=False, eog=True, ecg=True, exclude=duplicate_channels
        )
    # EEG
    if data_type["EEG"] == True:
        x = pd.read_csv(rec_dir + "/badEEG.csv")
        x = list(x[subj_id])
        z = [y for y in x if y == y]  # clean nans

        raw_s.info["bads"] = z
        raw_s.interpolate_bads()  # interpolate bad EEG channels

    # MEG
    x = pd.read_csv(rec_dir + "/badMEG.csv")
    x = list(x[subj_id])
    z = [y for y in x if y == y]  # clean nans
    raw_s.info["bads"].extend(
        z
    )  # MEG and GRAD bad channels will later be handled by Maxwell Filter

    # Band pass filter
    head_pos = mne.chpi.read_head_pos(megDir + "/01_head_motion/head_pos.pos")
    bp_raw = raw_s.copy().filter(l_freq=filt_band[0], h_freq=filt_band[1])

    # Notch filter (50Hz in UK/CHN)
    filt_raw = bp_raw.copy().notch_filter(np.arange(50.0, filt_band[1] + 1, 50.0))

    # Maxwell filter
    (
        auto_noisy_chs,
        auto_flat_chs,
        auto_scores,
    ) = mne.preprocessing.find_bad_channels_maxwell(
        filt_raw, head_pos=head_pos, return_scores=True, h_freq=None
    )

    bads = filt_raw.info["bads"] + auto_noisy_chs + auto_flat_chs
    filt_raw.info["bads"] = bads
    raw_maxfi = mne.preprocessing.maxwell_filter(filt_raw)

    # save files and record
    raw_maxfi.save(megDir + "/02_filter/Maxfiltered_raw.fif", overwrite=True)
    with open(megDir + "/02_flag.txt", "w") as f:
        f.write("Bad channels dealt and data filtered.")


def denoise():

    if not os.path.isdir(megDir + "/03_ICA_denoise"):
        os.mkdir(megDir + "/03_ICA_denoise")
    if not os.path.isdir(megDir + "/04_sensor_prep"):
        os.mkdir(megDir + "/04_sensor_prep")

    raw_filtered = mne.io.read_raw_fif(
        megDir + "/02_filter/Maxfiltered_raw.fif", preload=True, allow_maxshield=True
    )

    # find EOG epochs

    eog_evoked = mne.preprocessing.create_eog_epochs(raw_filtered).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))
    eogplot = eog_evoked.plot_joint(show=False)
    for i in range(len(eogplot)):
        eogplot[i].savefig(megDir + "/03_ICA_denoise/eog" + str(i + 1) + ".tif")


    # find ECG epochs
    ecg_evoked = mne.preprocessing.create_ecg_epochs(raw_filtered).average()
    ecg_evoked.apply_baseline(baseline=(None, -0.2))
    ecgplot = ecg_evoked.plot_joint(show=False)
    for i in range(len(ecgplot)):
        ecgplot[i].savefig(megDir + "/03_ICA_denoise/ecg" + str(i + 1) + ".tif")

    if denoise_method["SSP"] == True:
        ecg_projs, _ = mne.preprocessing.compute_proj_ecg(raw_filtered, n_grad=1, n_mag=1, reject=None)
        eog_projs, _ = mne.preprocessing.compute_proj_eog(raw_filtered, n_grad=2, n_mag=2, reject=None)

        raw_filtered.add_proj(ecg_projs)
        raw_filtered.add_proj(eog_projs)

    if denoise_method["ICA"] == True:
        # ICA
        ica = mne.preprocessing.ICA(n_components=50, method="picard", random_state=0)
        #reject = dict(mag=4e-12, grad=4000e-13, eog=250e-6)
        ica.fit(raw_filtered)
        icaplot = ica.plot_components(title="ICA components", show=False)
        for i in range(len(icaplot)):
            icaplot[i].savefig(megDir + "/03_ICA_denoise/ICAfig" + str(i + 1) + ".tif")

        # denoise
        ica.exclude = []

        eog_indices, eog_scores = ica.find_bads_eog(raw_filtered)
        ica.plot_scores(eog_scores, show=False).savefig(
            megDir + "/03_ICA_denoise/eogscore.tif"
        )
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_filtered)
        ica.plot_scores(ecg_scores, show=False).savefig(
            megDir + "/03_ICA_denoise/ecgscore.tif"
        )


        ica.exclude = eog_indices + ecg_indices
        raw_filtered = ica.apply(raw_filtered)

    # save file and records
    raw_filtered.save(megDir + "/04_sensor_prep/preprocessed.fif", overwrite=True)
    with open(megDir + "/03_flag.txt", "w") as f:
        f.write("Artefacts cleaned.")


def bem_src():

    if not os.path.isdir(mriDir + "/fig"):
        os.mkdir(mriDir + "/fig")
    if not os.path.isdir(mriDir + "/src"):
        os.mkdir(mriDir + "/src")

    Brain = mne.viz.get_brain_class()
    brain = Brain(
        subject="",
        views="lat",
        background="w",
        hemi="lh",
        surf="pial",
        subjects_dir=mriDir,
        title="Surface Reconstruction",
        size=(1000, 800),
    )
    brain.add_annotation("aparc.a2009s", borders=False)
    brain.save_image(mriDir + "/fig/recon.png")
    brain.close()
    # construct the BEM

    info = mne.io.read_info(megDir + "/04_sensor_prep/preprocessed.fif")
    if data_type["EEG"] == True:
        conductivity = (0.3, 0.006, 0.3)  # three layers model for EEG+MEG data
    else:
        conductivity = (0.3,)

    model = mne.make_bem_model(
        "", ico=5, conductivity=conductivity, subjects_dir=mriDir
    )
    bem_sol = mne.make_bem_solution(model)
    bem_model_fig = mne.viz.plot_bem(
        subject="", subjects_dir=mriDir, orientation="coronal", show=False
    )
    bem_model_fig.savefig(mriDir + "/fig/bem_model_fig.tif")
    mne.write_bem_solution(mriDir + "/src/bem_sol.h5", bem_sol, overwrite=True)

    # automatic coregistreation
    fiducials = "estimated"  # get fiducials from fsaverage
    coreg = mne.coreg.Coregistration(
        info, subject="", subjects_dir=mriDir, fiducials=fiducials
    )
    plot_kwargs = dict(
        subject="",
        subjects_dir=mriDir,
        surfaces="head-dense",
        dig=True,
        eeg=[],
        meg="sensors",
        show_axes=True,
        coord_frame="meg",
    )
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
    coreg.omit_head_shape_points(distance=5.0 / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
    dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
    print(
        f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
        f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
    )
    mne.write_trans(mriDir + "/src/auto-trans.fif", coreg.trans, overwrite=True)
    vol_src = mne.setup_volume_source_space(
        "",
        subjects_dir=mriDir,
        surface=mriDir + "/bem/inner_skull.surf",
        pos=4.0,
        mindist=1.0,
    )
    mne.write_source_spaces(mriDir + "/src/vol-src.fif", vol_src, overwrite=True)
    surf_src = mne.setup_source_space(
        "", subjects_dir=mriDir, spacing="oct6", add_dist="patch"
    )
    mne.write_source_spaces(mriDir + "/src/surf-src.fif", surf_src, overwrite=True)
    # plot sources
    plot_bem_kwargs = dict(
        subject="",
        subjects_dir=mriDir,
        brain_surfaces="white",
        orientation="coronal",
        slices=[50, 100, 150, 200],
    )
    src_fig = mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs, show=False)
    src_fig.savefig(mriDir + "/fig/vol_src.tif")
    src_fig = mne.viz.plot_bem(src=surf_src, **plot_bem_kwargs, show=False)
    src_fig.savefig(mriDir + "/fig/surf_src.tif")

    # save records
    with open(mriDir + "/BEM_log.txt", "w") as f:
        f.write("BEM constructed.")


def event_trig():

    if not os.path.isdir(megDir + "/event"):
        os.mkdir(megDir + "/event")

    data = mne.io.read_raw_fif(
        megDir + raw_filename, preload=True, allow_maxshield=True
    )
    
    events_all = mne.find_events(data,stim_channel="STI101",min_duration=10 / data.info["sfreq"])

    # TODO: move this part to a separate csv file.
    if events_manual == False:
        # set the min_duration > 1/frequency, so to avoid the problem caused by neuroMag
        np.savetxt(megDir + "/event/events.csv", events_all, delimiter=",")
        # Select only events corresponding to stimuli onset
        
        # Map events to animal/non animal categories
        
        events = mne.pick_events(events_all, include=list(range(1, 49)))
        merged_events = mne.merge_events(events, list(range(1, 13)) + list(range(25, 37)), 1)
        merged_events = mne.merge_events(merged_events, list(range(13, 25)) + list(range(37, 49)), 2)
        """
        events = mne.pick_events(events_all)
        merged_events = mne.merge_events(events, list(range(2, 3)), 1)
        merged_events = mne.merge_events(merged_events, list(range(4, 5)), 2)
        merged_events = mne.merge_events(merged_events, list(range(8, 9)), 3)
        merged_events = mne.merge_events(merged_events, list(range(16, 17)), 4)
        merged_events = mne.merge_events(merged_events, list(range(32, 33)), 5)
        merged_events = mne.merge_events(merged_events, list(range(64, 65)), 6)
        merged_events = mne.merge_events(merged_events, list(range(128, 129)), 7)
        """
    else:
        a1 = np.arange(12, 72, 3)
        a0 = np.arange(12, 12 + 25 * 3, 3)
        a2 = a0 + 12 + 72
        a3 = a2 + 75 + 12
        a4 = a3 + 75 + 12
        a_onset = np.concatenate((a1, a2, a3, a4))
        event_annot = mne.Annotations(
            onset=a_onset,
            duration=2.0,
            orig_time=data.info["meas_date"]
            + datetime.timedelta(seconds=(events_all[0][0]) / 1000),
            description=[task_name] * a_onset.size,
        )
        data.set_annotations(event_annot)
        global event_dict
        (merged_events, event_dict) = mne.events_from_annotations(data)
        np.savetxt(megDir + "/event/events.csv", merged_events, delimiter=",")

    # END OF TODO:

    # Plots events
    eventfig = mne.viz.plot_events(
        merged_events,
        sfreq=data.info["sfreq"],
        first_samp=data.first_samp,
        event_id=event_dict,
        show=False,
    )
    eventfig.savefig(megDir + "/event/event.tif")
    # save events and records
    mne.write_events(megDir + "/event/raw-eve.fif", merged_events, overwrite=True)

    with open(megDir + "/event_flag.txt", "w") as f:
        f.write("Event extracted.")


def src_loc():

    if not os.path.isdir(megDir + "/05_fwd_model"):
        os.mkdir(megDir + "/05_fwd_model")
    if not os.path.isdir(megDir + "/06_src_stc"):
        os.mkdir(megDir + "/06_src_stc")

    # load MEG data and compute the evoked data
    vol_src = mne.read_source_spaces(mriDir + "/src/vol-src.fif")
    surf_src = mne.read_source_spaces(mriDir + "/src/surf-src.fif")
    trans = mne.read_trans(mriDir + "/src/auto-trans.fif")
    bem_sol = mne.read_bem_solution(mriDir + "/src/bem_sol.h5")
    data = mne.io.read_raw_fif(megDir + "/04_sensor_prep/preprocessed.fif")

    if task_name.find("rs") != 0: # if it is not the resting-state data
        events = mne.read_events(megDir + "/event/raw-eve.fif")
        tmin = -0.2  # start of each epoch
        tmax = 0.8  # end of each epoch
        baseline = (None, 0)  # means from the first instant to t = 0
        epochs = mne.Epochs(
            data, events, event_dict, tmin, tmax, proj=True, baseline=baseline
        )
        reject = get_rejection_threshold(epochs)
        epochs = mne.Epochs(
            data,
            events,
            event_dict,
            tmin,
            tmax,
            proj=True,
            baseline=baseline,
            reject=reject,
        )
        cov = mne.compute_covariance(
            epochs, tmax=0.0, method=["shrunk", "empirical"], rank="info", verbose=True
        )
        fig_cov, fig_spectra = mne.viz.plot_cov(cov, data.info, show=False)
        fig_cov.savefig(megDir + "/05_fwd_model/cov.tif")
        fig_spectra.savefig(megDir + "/05_fwd_model/spect.tif")

        evoked = epochs.average()
        evokedfig = evoked.plot(time_unit="s", show=False)  # plot evoked data
        evokedfig.savefig(megDir + "/05_fwd_model/evoked.tif")
        whitening_fig = evoked.plot_white(cov, time_unit="s", show=False)
        whitening_fig.savefig(megDir + "/05_fwd_model/whitening.tif")
    else:
        if data_type["EEG"] == True:
            mne.set_eeg_reference(data, projection=True)
        cov = mne.compute_raw_covariance(
            data, method=["shrunk", "empirical"], rank="info"
        )

    # forward solution for volume
    if output_type["volume"] == True:
        if not os.path.exists(megDir + "/05_fwd_model/vol-fwd.fif"):
            if data_type["EEG"] == True:
                fwd_vol = mne.make_forward_solution(
                    megDir + "/04_sensor_prep/preprocessed.fif",
                    trans=trans,
                    src=vol_src,
                    bem=bem_sol,
                    meg=True,
                    eeg=True,
                    n_jobs=8,
                )
            else:
                fwd_vol = mne.make_forward_solution(
                    megDir + "/04_sensor_prep/preprocessed.fif",
                    trans=trans,
                    src=vol_src,
                    bem=bem_sol,
                    meg=True,
                    eeg=False,
                    n_jobs=8,
                )
            mne.write_forward_solution(
                megDir + "/05_fwd_model/vol-fwd.fif", fwd_vol, overwrite=True
            )
        else:
            fwd_vol = mne.read_forward_solution(megDir + "/05_fwd_model/vol-fwd.fif")

    # forward solution for surface
    if output_type["surface"] == True:
        if not os.path.exists(megDir + "/05_fwd_model/surf-fwd.fif"):
            if data_type["EEG"] == True:
                fwd_surf = mne.make_forward_solution(
                    megDir + "/04_sensor_prep/preprocessed.fif",
                    trans=trans,
                    src=surf_src,
                    bem=bem_sol,
                    meg=True,
                    eeg=True,
                    mindist=3.0,
                    n_jobs=8,
                )
            else:
                fwd_surf = mne.make_forward_solution(
                    megDir + "/04_sensor_prep/preprocessed.fif",
                    trans=trans,
                    src=surf_src,
                    bem=bem_sol,
                    meg=True,
                    eeg=False,
                    mindist=3.0,
                    n_jobs=8,
                )
            mne.write_forward_solution(
                megDir + "/05_fwd_model/surf-fwd.fif", fwd_surf, overwrite=True
            )
        else:
            fwd_surf = mne.read_forward_solution(megDir + "/05_fwd_model/surf-fwd.fif")

    # inverse solutions
    if task_name.find("rs") == 0:
        if output_type["volume"] == True:
            inv_vol = mne.minimum_norm.make_inverse_operator(
                data.info, fwd_vol, cov, loose="auto", depth=0.8
            )
        if output_type["surface"] == True:
            inv_surf = mne.minimum_norm.make_inverse_operator(
                data.info, fwd_surf, cov, loose=0.2, depth=0.8
            )
    else:
        if output_type["volume"] == True:       
            inv_vol = mne.minimum_norm.make_inverse_operator(
                evoked.info, fwd_vol, cov, loose="auto", depth=0.8
            )
        if output_type["surface"] == True:
            inv_surf = mne.minimum_norm.make_inverse_operator(
                evoked.info, fwd_surf, cov, loose=0.2, depth=0.8
            )

    snr = 3.0
    method = "MNE"
    lambda2 = 1.0 / snr**2

    # source localisation
    if task_name.find("rs") != 0:
        for id_event in range(1, 1 + len(event_dict)):
            # extracting trial average
            epochs = mne.Epochs(
                data,
                events,
                id_event,
                #event_dict,
                tmin,
                tmax,
                proj=True,
                baseline=baseline,
                reject=reject,
            )
            evoked = epochs.average()
            if data_type["EEG"] == True:
                mne.set_eeg_reference(evoked, projection=True)

            # volume
            if output_type["volume"] == True:
                stc, residual = mne.minimum_norm.apply_inverse(
                    evoked,
                    inv_vol,
                    lambda2,
                    method,
                    pick_ori=None,
                    return_residual=True,
                )  # volume-source TC
                stc.save_as_volume(
                    megDir
                    + "/06_src_stc/vol-TS_src_"
                    + list(event_dict.keys())[id_event - 1]
                    + ".nii.gz",
                    vol_src,
                    dest="mri",
                    format="nifti1",
                    overwrite=True,
                )  # save a nii-file
                stc.save(
                    megDir
                    + "/06_src_stc/vol-TS_src_"
                    + list(event_dict.keys())[id_event - 1],
                    ftype="stc",
                    overwrite=True,
                )  # save a mne-file

            # surface
            if output_type["surface"] == True:
                stc, residual = mne.minimum_norm.apply_inverse(
                    evoked,
                    inv_surf,
                    lambda2,
                    method,
                    verbose=True,
                    pick_ori=None,
                    return_residual=True,
                )
                if os.path.isdir(rec_dir + "/morph-maps"):
                    shutil.rmtree(rec_dir + "/morph-maps")
                morph = mne.compute_source_morph(
                    stc,
                    subject_from=mriDir,
                    subject_to="S:/MEG/milos/preproc/fsaverage",
                    src_to=surf_src,
                    subjects_dir=rec_dir,
                )
                stc_fsaverage = morph.apply(stc)
                stc_fsaverage.save(
                    megDir
                    + "/06_src_stc/surf-TS_src_"
                    + list(event_dict.keys())[id_event - 1],
                    ftype="stc",
                    overwrite=True,
                )  # save a mne-file
    else:
        # surface localisation
        # volume localisation
        # NOTE: volume files were separated into several sub-files to avoid memory error
        
        t0 = 100
        timestep = 5000 / data.info["sfreq"]
        while t0 < data.tmax:  # 5000 timepoints per file to avoid memory error
            subdata = data.copy().crop(tmin=t0, tmax=min(t0 + timestep, data.tmax))
            if output_type["surface"] == True:
                if not os.path.isdir(megDir + "/06_src_stc/surf_file"):
                    os.mkdir(megDir + "/06_src_stc/surf_file")
                stc = mne.minimum_norm.apply_inverse_raw(
                    subdata,
                    inv_surf,
                    lambda2,
                    method,
                )
                if os.path.isdir(rec_dir + "/morph-maps"):
                    shutil.rmtree(rec_dir + "/morph-maps")
                morph = mne.compute_source_morph(
                    stc,
                    subject_from=mriDir,
                    subject_to="S:/MEG/milos/preproc/fsaverage",
                    src_to=surf_src,
                    subjects_dir=rec_dir,
                )
                stc_fsaverage = morph.apply(stc)
                stc_fsaverage.save(
                    megDir
                    + "/06_src_stc/surf_file/surf-TS_src"
                    + str(t0)
                    + "to"
                    + str(t0 + timestep),
                    ftype="stc",
                    overwrite=True,
                )  # save a mne-file
                del stc  # release memory
                del stc_fsaverage

            if output_type["volume"] == True:
                if not os.path.isdir(megDir + "/06_src_stc/vol_niifile"):
                    os.mkdir(megDir + "/06_src_stc/vol_niifile")
                if not os.path.isdir(megDir + "/06_src_stc/vol_stcfile"):
                    os.mkdir(megDir + "/06_src_stc/vol_stcfile")
                stc = mne.minimum_norm.apply_inverse_raw(
                    subdata,
                    inv_vol,
                    lambda2,
                    method,
                )  # volume-source TC
                stc.save_as_volume(
                    megDir
                    + "/06_src_stc/vol_niifile/vol-TS_src"
                    + str(t0)
                    + "to"
                    + str(t0 + timestep)
                    + ".nii.gz",
                    vol_src,
                    dest="mri",
                    format="nifti1",
                    overwrite=True,
                )  # save a nii-file
                stc.save(
                    megDir
                    + "/06_src_stc/vol_stcfile/vol-TS_src_"
                    + str(t0)
                    + "to"
                    + str(t0 + timestep),
                    ftype="stc",
                    overwrite=True,
                )  # save a mne-file
            t0 += timestep
            del subdata
            del stc
    if os.path.isdir(rec_dir + "/morph-maps"):
        shutil.rmtree(rec_dir + "/morph-maps")


def ROI_parc():

    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=rec_dir, accept=True)
    if not os.path.isdir(megDir + "/06_src_stc/ROI_tc"):
        os.mkdir(megDir + "/06_src_stc/ROI_tc")

    for id_event in range(1, 1 + len(event_dict)):
        surf_src = mne.read_source_spaces(mriDir + "/src/surf-src.fif")

        # left hemisphere
        stc_fsaverage_lh = mne.read_source_estimate(
            megDir
            + "/06_src_stc/surf-TS_src_"
            + list(event_dict.keys())[id_event - 1]
            + "-lh.stc"
        )
        stc_fsaverage_lh.subject = ""
        mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=rec_dir, accept=True)
        hcp_mmp_lh = mne.read_labels_from_annot(
            subject="",
            parc="lh.HCPMMP1",
            hemi="lh",
            annot_fname=rec_dir + "/fsaverage/label/lh.HCPMMP1.annot",
            subjects_dir=rec_dir + "/fsaverage",
        )
        ts_lh = mne.extract_label_time_course(
            stcs=stc_fsaverage_lh, labels=hcp_mmp_lh[1:181], src=surf_src
        )
        scipy.io.savemat(
            megDir
            + "/06_src_stc/ROI_tc/ts_HCPmmp_lh_"
            + list(event_dict.keys())[id_event - 1]
            + ".mat",
            {"ts_lh": ts_lh},
        )
        with open(
            megDir
            + "/06_src_stc/ROI_tc/lh_ROI_label_"
            + list(event_dict.keys())[id_event - 1]
            + ".csv",
            "w",
            newline="",
        ) as csvfile:
            ROIwriter = csv.writer(
                csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            for i_ROI in range(1, 181):
                ROIwriter.writerow([hcp_mmp_lh[i_ROI]])

        # right hemisphere
        stc_fsaverage_rh = mne.read_source_estimate(
            megDir
            + "/06_src_stc/surf-TS_src_"
            + list(event_dict.keys())[id_event - 1]
            + "-rh.stc"
        )
        stc_fsaverage_rh.subject = ""
        hcp_mmp_rh = mne.read_labels_from_annot(
            subject="",
            parc="rh.HCPMMP1",
            hemi="rh",
            annot_fname=rec_dir + "/fsaverage/label/rh.HCPMMP1.annot",
            subjects_dir=rec_dir + "/fsaverage",
        )
        ts_rh = mne.extract_label_time_course(
            stcs=stc_fsaverage_rh, labels=hcp_mmp_rh[1:181], src=surf_src
        )
        scipy.io.savemat(
            megDir
            + "/06_src_stc/ROI_tc/ts_HCPmmp_rh_"
            + list(event_dict.keys())[id_event - 1]
            + ".mat",
            {"ts_rh": ts_rh},
        )
        with open(
            megDir
            + "/06_src_stc/ROI_tc/rh_ROI_label_"
            + list(event_dict.keys())[id_event - 1]
            + ".csv",
            "w",
            newline="",
        ) as csvfile:
            ROIwriter = csv.writer(
                csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            for i_ROI in range(1, 181):
                ROIwriter.writerow([hcp_mmp_rh[i_ROI]])


"""
# visualizaion
# This part could be ran manually to output the movie for (E)MEG signal on surface.

mne.viz.set_3d_backend("pyvistaqt")
mne.viz.set_3d_options(antialias=False)
stc_fsaverage.plot(views=['lat','dorsal','lat'], 
view_layout="horizontal", hemi='both', size=(1500, 500), 
subjects_dir=rec_dir, initial_time=0.8, time_viewer=True, 
show_traces=False, 
clim = {"kind":"value", "lims":[8e-14, 2e-12, 5e-12]},
#clim = {"kind":"value", "lims":[1e-12, 7.5e-12, 1.5e-11]},
)
brain = stc_fsaverage.plot(subjects_dir=rec_dir, initial_time=-0.2,
                    clim=dict(kind='value', lims=[8e-14, 2e-12, 5e-12]),
                    surface='flat', hemi='both', size=(1000, 500),
                    smoothing_steps=5, time_viewer=True,
                    add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=10)))
"""

if __name__ == "__main__":
    """
    global variables defined for the program
    """
    # to avoid problems caused by visualisation
    mne.viz.set_3d_backend("pyvistaqt")
    mne.viz.set_3d_options(antialias=False)

    # define directories as global variable
    subjIDs = os.listdir(proj_dir)
    # define error lists for log
    f_error_list = []
    r_error_list = []
    v_error_list = []
    m_error_list = []
    other_error_list = []
    fail = 0

    for subj_id in subjIDs:
        megDir = proj_dir + "/" + subj_id + "/MEG/" + task_name
        mriDir = proj_dir + "/" + subj_id + "/MRI"

        try:
            main()  # run the main function
        except RuntimeError:  #
            r_error_list.append(subj_id)
            fail += 1
            print(f"Subject {subj_id} was not preprocessed properly, please check.")
        except FileNotFoundError:
            f_error_list.append(subj_id)
            fail += 1
            print(f"Please check the folder structure for subject {subj_id}.")
        except MemoryError or np.core._exceptions._ArrayMemoryError:
            m_error_list.append(subj_id)
            fail += 1
            print(f"Not sufficient RAM when processing subject {subj_id}")
        except ValueError:
            v_error_list.append(subj_id)
            fail += 1
            print(f"Please check the folder structure for subject {subj_id}.")
        except:
            other_error_list.append(subj_id)
            fail += 1
            print(f"Other Problems with subject {subj_id}.")

    # writing log file
    with open(rec_dir + "/proc_log_" + task_name + ".txt", "w") as f:
        f.write(
            f"{len(subjIDs)-fail} / {len(subjIDs)} subjects have been pre-processed successfully!\n\n\n"
        )

        if len(r_error_list) > 0:
            f.write(
                "Following subjects were not preprocessed properly due to RuntimeError:\n\n"
            )
            for subj_error in r_error_list:
                f.write(f"{subj_error} | ")
            f.write("\n\n")
            f.write(
                "Common problems includes: wrong channel names or defects in BEM surface. Please check the MEG file for those subjects!\n\n\n"
            )

        if len(f_error_list) > 0:
            f.write(
                "Following subjects were not preprocessed properly due to FileNotFoundError:\n\n"
            )
            for subj_error in f_error_list:
                f.write(f"{subj_error} | ")
            f.write("\n\n")
            f.write("Please check the MEG/MRI file for those subjects!\n\n\n")

        if len(v_error_list) > 0:
            f.write(
                "Following subjects were not preprocessed properly due to ValueError:\n\n"
            )
            for subj_error in v_error_list:
                f.write(f"{subj_error} | ")
            f.write("\n\n")
            f.write("Please check the files for those subjects!\n\n\n")

        if len(m_error_list) > 0:
            f.write(
                "Not sufficient RAM when processing following subjects which raise a MemoryError:\n\n"
            )
            for subj_error in m_error_list:
                f.write(f"{subj_error} | ")
            f.write("\n\n")
            f.write(
                "Please rerun the pipeline for those subjects or try to allocate larger memory!\n\n\n"
            )

        if len(other_error_list) > 0:
            f.write(
                "Following subjects were not preprocessed properly due to Other type Error:\n\n"
            )
            for subj_error in other_error_list:
                f.write(f"{subj_error} | ")
            f.write("\n\n")
            f.write(
                "Please manually check the pre-processing procedure for those subjects!\n"
            )
            f.write(
                "In some occasions, rerun the pre-processing pipeline for them would magically solve the problem in this case.\n\n"
            )
            f.write("Good Luck!")

    print(f"log file saved as '{rec_dir}/proc_log_" + task_name + ".txt' ")
