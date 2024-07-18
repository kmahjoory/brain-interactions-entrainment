import mne
import os
import matplotlib
import matplotlib.pyplot as plt
mne.viz.set_3d_options(antialias=False)

from nf_tools import fm, utils
from nf_tools import preprocessing as prep




# Set the analysis directory where all plots will be saved
analysis_dir = '/mnt/beegfs/workspace/2021-0292-NeuroFlex/prj_neuroflex/k1_analyses'


subj_id = 10 #'subj_2_JMI22'
path = utils.set_dirs(subj_id, analysis_dir=analysis_dir)
subj_name = path['subj_id']
meg_dir = path['subj_meg_dir']
subj_analysis_dir = path['subj_analysis_dir']
prep_dir = path['prep_dir']

fs_subjs_dir = path['fs_subjs_dir']
subj_fs_dir = path['subj_fs_dir']
mri_dir = path['subj_mri_dir']


## Check the following slides
# https://www.slideshare.net/mne-python/mnepython-scale-mri

## Calculate Forward solution
########################################################################################################################
# Read info

# Read transformation matrix (If str, the path to the head<->MRI transform *-trans.fif file produced during coregistration
#Can also be 'fsaverage' to use the built-in fsaverage transformation. )




# Read source files
fpath_src = os.path.join(subj_fs_dir, 'source/file-src.fif')
src = mne.read_source_spaces(fpath_src)

# Read BEM Solution
bem_solution = mne.read_bem_solution(os.path.join(mri_dir, 'bem_sol.h5'))

# Read MEG info
meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'block_1_raw.fif'))
info = meg.info

# Load the transformation matrix
trans = mne.read_trans(os.path.join(path['subj_trans_dir'], 'meg_mri-trans.fif'))

# solve the Forward problem
fwd = mne.make_forward_solution(info=info, trans=trans, src=src, bem=bem_solution, meg=True, eeg=False, mindist=0.0,
                                ignore_ref=False, n_jobs=1)

mne.write_forward_solution(os.path.join(mri_dir, 'file-fwd.fif'), fwd, overwrite=True)
#read_forward_solution



