import os
import sys
import mne
Brain = mne.viz.get_brain_class()
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
import glob
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.time_frequency import tfr_morlet  # noqa
from nf_tools import utils, fm
from nf_tools import preprocessing as prep


subj_indx = 13
subj_id = f'subj_{subj_indx}'
paths = fm.set_mri_dirs(subj_indx)



# Load and Epoch sensor data
meg = mne.io.read_raw_fif(os.path.join(paths['subj_meg_dir'], 'after_ica_nf_meg.fif'))
epoch = prep.mk_epochs(meg.copy(), mod_freq=1., tmin=-0.3, baseline=(-0.3, 0), 
                       annot_pattern='e/1.0/', new_event_value=101)
evoked = epoch.average()

# Covariance Matrices
data_cov = mne.compute_covariance(epoch, tmin=0.005, tmax=7.5,
                                  method='empirical')
noise_cov = mne.compute_covariance(epoch, tmin=-0.3, tmax=0,
                                   method='empirical')

# Load forward model
fwd = mne.read_forward_solution(os.path.join(paths['subj_mri_dir'], 'file-fwd.fif'))

# Source Reconstruction: MNE
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, data_cov, loose=0., depth=0.8,
                            verbose=True)

snr = 3.0
lambda2 = 1.0 / snr ** 2
kwargs = dict(subject=subj_id, initial_time=0.08, hemi='lh', subjects_dir=paths['fs_subjs_dir'],
              size=(600, 600), clim=dict(kind='percent', lims=[90, 95, 99]),
              smoothing_steps=7)


stc = abs(mne.minimum_norm.apply_inverse(evoked, inv, lambda2, 'MNE', verbose=True))
brain = stc.plot(figure=1, **kwargs)
brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)



kwargs = dict(subject=subj_id, initial_time=0.08, hemi='lh', subjects_dir=paths['fs_subjs_dir'],
              size=(600, 600), clim=dict(kind='percent', lims=[90, 95, 99]),
              smoothing_steps=7)
stc = abs(mne.minimum_norm.apply_inverse(evoked, inv, lambda2, 'dSPM', verbose=True))
stc.subject = subj_id
brain = stc.plot(figure=1, **kwargs)
brain.add_text(0.1, 0.9, 'dSPM', 'title', font_size=14)


kwargs = dict(subject=subj_id, initial_time=0.08, hemi='rh', subjects_dir=paths['fs_subjs_dir'],
              size=(600, 600), clim=dict(kind='percent', lims=[90, 95, 99]),
              smoothing_steps=7)
stc = abs(mne.minimum_norm.apply_inverse(evoked, inv, lambda2, 'dSPM', verbose=True))
stc.subject = subj_id
brain = stc.plot(figure=2, **kwargs)
brain.add_text(0.1, 0.9, 'dSPM', 'title', font_size=14)






subjects_dir = mne.datasets.sample.data_path() + '/subjects'
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=paths['fs_subjs_dir'],
                                        verbose=True)
mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=paths['fs_subjs_dir'],
                                          verbose=True)

labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1', 'lh', subjects_dir=paths['fs_subjs_dir'])

brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=paths['fs_subjs_dir'],
              cortex='low_contrast', background='white', size=(800, 600))
brain.add_annotation('HCPMMP1')
aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
brain.add_label(aud_label, borders=False)





# https://github.com/ryraut/arousal-waves/tree/main/surface_files/Conte69_atlas.LR.4k_fs_LR
