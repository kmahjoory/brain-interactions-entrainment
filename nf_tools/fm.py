# -*- coding: utf-8 -*-
import mne
import os
import matplotlib
import matplotlib.pyplot as plt
from nf_tools.utils import set_dirs



def mk_bem_solution(subj_indx):
    
    paths = set_mri_dirs(subj_indx)
    bem_surfaces = mne.make_bem_model(paths['subj_fs_dir'], ico=4, conductivity=[0.3], 
                                      subjects_dir=paths['fs_subjs_dir'])
    bem_solution = mne.make_bem_solution(bem_surfaces, verbose=None)
    mne.write_bem_solution(os.path.join(paths['subj_mri_dir'], 'bem_solution.h5'), bem=bem_solution, overwrite=True)
    return bem_solution


def mk_fwd_model(subj_idx):
    
    paths = set_mri_dirs(subj_idx)
    
    # Read source files
    fpath_src = os.path.join(paths['subj_fs_dir'], 'source/file-src.fif')
    src = mne.read_source_spaces(fpath_src)
    
    # Read BEM Solution
    bem_solution = mne.read_bem_solution(os.path.join(paths['subj_mri_dir'], 'bem_solution.h5'))
    
    # Read MEG info
    meg = mne.io.read_raw_fif(os.path.join(paths['subj_meg_dir'], 'block_1_raw.fif'))
    info = meg.info
    
    # Load the transformation matrix    
    trans = mne.read_trans(os.path.join(paths['subj_trans_dir'], 'meg_mri-trans.fif'))
    
    # solve the Forward problem
    fwd = mne.make_forward_solution(info=info, trans=trans, src=src, bem=bem_solution, meg=True, eeg=False, mindist=0.0,
                                    ignore_ref=False, n_jobs=1)  
    mne.write_forward_solution(os.path.join(paths['subj_mri_dir'], 'file-fwd.fif'), fwd, overwrite=True)
    return fwd
    

