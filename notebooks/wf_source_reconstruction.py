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
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.time_frequency import tfr_morlet  # noqa
from nf_tools import utils, fm
from nf_tools import preprocessing as prep


# Set Analysis Directory
analysis_dir = '/mnt/beegfs/workspace/2021-0292-NeuroFlex/prj_neuroflex/k1_analyses'


# mk_epochs
subj_id = 10
path = utils.set_dirs(subj_id=subj_id, analysis_dir=analysis_dir)
meg_dir = path['subj_meg_dir']
subj_name = path['subj_id']
prep_dir = path['prep_dir']



# Load after-ICA MEG data and make epochs
meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'after_ica_meg.fif'))
epoch = prep.mk_epochs(meg.copy(), mod_freq=2., tmin=-0.2, baseline=(-0.2, 0), 
                   annot_pattern='e/2.0/', new_event_value=101)
evoked = epoch.average()


# Covariance Matrices
data_cov = mne.compute_covariance(epoch, tmin=0.0, tmax=3.5,
                                  method='empirical')
noise_cov = mne.compute_covariance(epoch, tmin=-0.2, tmax=-0.05,
                                   method='empirical')
#mne.viz.plot_cov(noise_cov, meg.info)

#evoked.plot()
#evoked.plot_white(noise_cov, time_unit='s')

# Load forward model
fwd = mne.read_forward_solution(os.path.join(path['subj_mri_dir'], 'file-fwd.fif'))

# Source Reconstruction: MNE
inv_operator = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, data_cov, loose=0., depth=0.8,
                            verbose=True)

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked, inv_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True,  verbose=True)


fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[::10, :].T)
ax.set(xlabel='time (ms)', ylabel='%s value' % method)
ax.set_xlim(-100, 300)


# Read
label = mne.read_labels_from_annot(subject=subj_name, parc='BN_Atlas', hemi='both', 
                                   surf_name='white', annot_fname=None, regexp=None, 
                                   subjects_dir=path['fs_subjs_dir'], sort=True, verbose=None)


#stc_label = stc.in_label(label[0])
roimode = 'mean_flip'

if roimode == 'pca_flip':
    tcs = stc.extract_label_time_course(labels=label, src=inv_operator['src'], mode='pca_flip').copy()
if roimode == 'mean':
    tcs = stc.extract_label_time_course(labels=label, src=inv_operator['src'], mode='mean').copy()
if roimode == 'mean_flip':
    tcs = stc.extract_label_time_course(labels=label, src=inv_operator['src'], mode='mean_flip').copy()



"""
brain = mne.viz.Brain(subj_name, hemi='both', surf='pial', alpha=1,
                      cortex='low_contrast', subjects_dir=path['fs_subjs_dir'])
brain.add_annotation('BN_Atlas', borders=False)


brain = mne.viz.Brain(subj_name, hemi='both', surf='pial', alpha=1,
                      cortex='low_contrast', subjects_dir=path['fs_subjs_dir'])
brain.add_label(label)

brain.add_volume_labels(aseg='BN_Atlas', labels=label)
brain.show_view(azimuth=250, elevation=40, distance=400)
"""


for k in range(len(label)):
    print(k, label[k].name)
    
write_dir = f'/mnt/beegfs/workspace/2021-0292-NeuroFlex/prj_neuroflex/k1_analyses/subj_{subj_id}/source_recon/erf'
os.makedirs(write_dir, exist_ok=True)
    
atlas_labels = [label[k].name for k in range(len(label))]

stg_l = ['A38m_L-lh', 'A41/42_L-lh', 'TE1.0/TE1.2_L-lh', 'A22c_L-lh', 'A38l_L-lh', 'A22r_L-lh']    
stg_r = ['A38m_R-rh', 'A41/42_R-rh', 'TE1.0/TE1.2_R-rh', 'A22c_R-rh', 'A38l_R-rh', 'A22r_R-rh'] 
mtg_l = ['A21c_L-lh', 'A21r_L-lh', 'A37dl_L-lh', 'aSTS_L-lh' ]
mtg_r = ['A21c_R-rh', 'A21r_R-rh', 'A37dl_R-rh', 'aSTS_R-rh']

regions = ['STG L', 'STG R', 'MTG L', 'MTG R']
for jp, parcel_sel in enumerate([stg_l, stg_r, mtg_l, mtg_r]):
    nel_sel = len(parcel_sel)
    nrows = 2
    ncols = nel_sel // 2
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(10, 7))
    fig.suptitle(regions[jp], fontsize=14)
    for j, k in enumerate(parcel_sel):
        indx_roi = atlas_labels.index(k)
        roi_tcs = tcs[indx_roi]
        ax[j//ncols, j%ncols].plot(1e3 * stc.times, roi_tcs)
        ax[j//ncols, j%ncols].set_xlim(-100, 200)
        ax[j//ncols, j%ncols].set_xticks([-100, 0, 50, 100, 150])
        ax[j//ncols, j%ncols].set_title(k)
        ax[j//ncols, j%ncols].grid(True)
        ax[j//ncols, j%ncols].set_xlabel('time (ms)')
    print(f'erf_{roimode}_{regions[jp]}_{k}.jpg')
    fig.savefig(os.path.join(write_dir, f'erf_{roimode}_{regions[jp]}_{k}.jpg'))
plt.close('all')
    



    




