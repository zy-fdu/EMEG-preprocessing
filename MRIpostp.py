import os
import sys
sys.path.append("preproc")
from EMEG_user_define_parameters import (task_name,proj_dir,rec_dir)

reff = "/mnt/z/research/ancillary/parcellation/MNI/MNI152_T1_4mm_brain.nii.gz"
reff_T1 = "/mnt/z/research/ancillary/parcellation/MNI/MNI152_T1_2mm_brain.nii.gz"
subjIDs = os.listdir(proj_dir)
err_list = []

for subj_id in subjIDs:    
    try:
        megDir = proj_dir + "/" + subj_id + "/MEG/" + task_name + "/06_src_stc"
        meg_reg_dir = proj_dir + "/" + subj_id + "/MEG/" + task_name + "/06_src_stc/vol_niifile_std"
        mriDir = proj_dir + "/" + subj_id + "/MRI"
        mriprocDir = mriDir + "/str_prep"
        if not os.path.isdir(meg_reg_dir):
            os.mkdir(meg_reg_dir)
        if not os.path.isdir(mriprocDir):
            os.mkdir(mriprocDir)
        # register
        rawT1 = mriDir + "/T1.nii.gz"
        cc_rawT1 = mriprocDir + "/cc_T1.nii.gz"
        bet_cc_rawT1 = mriprocDir + "/bet_cc_T1.nii.gz"
        wm_T1 = mriprocDir + "/bet_cc_T1_pve_2.nii.gz"
        reff_mask = "/mnt/z/research/ancillary/parcellation/MNI/MNI_Mask_4mm.nii.gz"

        mrifiles = os.listdir(mriDir)
        for file in mrifiles:
            if file.endswith(".nii.gz"):
                os.rename(mriDir + "/" + file, mriDir + "/T1.nii.gz")
        os.system("robustfov -i " + rawT1 + " -r " + cc_rawT1)
        os.system("bet " + cc_rawT1 + " " + bet_cc_rawT1 + " -m -f 0.3")
        os.system("fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o " + bet_cc_rawT1) # segmentation

        os.system("flirt -in " + bet_cc_rawT1 + " -ref " + reff_T1 + " -omat " + mriprocDir + "/linear_trans.mat")
        os.system("fnirt --in=" + bet_cc_rawT1 + " --ref=" + reff_T1 + " --aff=" + mriprocDir + "/linear_trans.mat --cout=" + mriprocDir + "/str2std_nonlinear_trans --config=T1_2_MNI152_2mm")
        os.system("applywarp -r " + reff_T1 + " -i " + bet_cc_rawT1 + " -o " + mriprocDir + "/T1_2mmStdSpace -w " + mriprocDir + "/str2std_nonlinear_trans")

        files = os.listdir(megDir)

        for file in files:
            if file.endswith(".nii.gz"):
                fname,affix1,affix2 = file.rsplit(".",maxsplit=2)
                megfile = megDir + "/" + file
                outfile = meg_reg_dir + "/" + fname + "_std." + affix1 + "." + affix2

                # BBR
                os.system("epi_reg --epi=" + megfile + " --t1=" + rawT1 + " --t1brain=" + bet_cc_rawT1 + " --wmseg=" + wm_T1 + "  --out=" + mriprocDir + "/Func2str")
                os.system("convert_xfm -omat " + mriprocDir + "/str2Func.mat -inverse " + mriprocDir + "/Func2str.mat")
                os.system("flirt -in " + cc_rawT1 + " -ref " + megfile + " -out " + mriprocDir + "/T1_in_FunImgSpace.nii.gz -init " + mriprocDir + "/str2Func.mat -applyxfm")
                ##
                os.system("convertwarp --ref=" + reff_T1 + " --warp1=" + mriprocDir + "/str2std_nonlinear_trans --premat=" + mriprocDir + "/Func2str.mat --out=" + mriprocDir + "/Func2std --relout")
                os.system("invwarp --ref=" + megfile + " --warp=" + mriprocDir + "/Func2std --out=" + mriprocDir + "/std2Func")

                os.system("applywarp --mask=" + reff_mask + " --ref=" + reff + " --in=" + megfile + " --warp=" + mriprocDir + "/Func2std --rel --out=" + outfile)

                print("file" + megfile + " has been normalized")
    except:
        err_list.append(subj_id)
with open(rec_dir + "/reg_log_" + task_name + ".txt", "w") as f:
    if len(err_list) > 0:
        f.write("Following subjects were not registered properly:\n")
        for subj_error in err_list:
            f.write(f"{subj_error} | ")