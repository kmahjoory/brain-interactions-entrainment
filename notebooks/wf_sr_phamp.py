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
import scipy.signal as signal
import glob
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs, read_inverse_operator
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
fs_subjs_dir = path['fs_subjs_dir']



# Load after-ICA MEG data and make epochs
meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'after_ica_meg.fif'))
epoch = prep.mk_epochs(meg.copy(), mod_freq=2., tmin=-0.2, baseline=(-0.2, 0), 
                   annot_pattern='e/2.0/', new_event_value=101)



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
inv_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, data_cov,
                                                      loose=0.2, depth=0.8, verbose=True)

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
# Change this to free orientation according to 
# https://mne.tools/stable/generated/mne.minimum_norm.apply_inverse_epochs.html
stc = apply_inverse_epochs(epoch, inv_operator, lambda2, method=method, 
                           pick_ori='normal', verbose=True)


# Read Atlas labels
label = mne.read_labels_from_annot(subject='subj_10', parc='BN_Atlas', hemi='both', 
                                   surf_name='white', annot_fname=None, regexp=None, 
                                   subjects_dir=path['fs_subjs_dir'], sort=True, verbose=None)


roimode = 'pca_flip'
tcs = np.array([stc[k].extract_label_time_course(labels=label, src=inv_operator['src'], mode=roimode) for k in range(len(stc))])

#roi_tcs = tcs.copy()
atlas_labels = [label[k].name for k in range(len(label))]


# Time Frequency analysis on ROIs' time course
freqs = np.arange(1, 40, 0.25)#np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2 #fm  # different number of cycle per frequency
import time
start_time = time.time()
power = mne.time_frequency.tfr_array_morlet(tcs, sfreq=meg.info['sfreq'], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        decim=3, n_jobs=1)
print(f'The process elapsed {time.time()-start_time} seconds')
# decimation = (33/1000)*300

power = np.swapaxes(power, axis1=0, axis2=1)
  
#write_dir = f'/mnt/beegfs/workspace/2021-0292-NeuroFlex/prj_neuroflex/k1_analyses/subj_{subj_id}/source_recon/itc'
#os.makedirs(write_dir, exist_ok=True)

seed_l = 'A41/42_L-lh'
indx_seed_l = atlas_labels.index(seed_l)

seed_r = 'A41/42_R-rh'
indx_seed_r = atlas_labels.index(seed_r)

indx_delta = (freqs>1) & (freqs<4)
indx_theta = (freqs>4) & (freqs<8)
indx_alpha = (freqs>8) & (freqs<12)
indx_beta = (freqs>16) & (freqs<30)

p_delta = np.abs(power[:, :, indx_delta, :]).mean(axis=2).reshape(212, -1).T
p_theta = np.abs(power[:, :, indx_theta, :]).mean(axis=2).reshape(212, -1).T
p_alpha = np.abs(power[:, :, indx_alpha, :]).mean(axis=2).reshape(212, -1).T
p_beta = np.abs(power[:, :, indx_beta, :]).mean(axis=2).reshape(212, -1).T

p_seed_l_delta = p_delta[:, indx_seed_l].reshape(-1, 1)
p_seed_r_delta = p_delta[:, indx_seed_r].reshape(-1, 1)


r = signal.correlate(p_seed_l_delta, p_theta, 'full')
lags = signal.correlation_lags(p_seed_l_delta, p_theta, 'full')

r = np.corrcoef(p_seed_l_delta.T, p_theta.T)


plt.imshow(r, aspect='auto')
plt.show()

"""
for jp, parcel_sel in enumerate([stg_l, stg_r, mtg_l, mtg_r]):
    nel_sel = len(parcel_sel)
    nrows = 2
    ncols = nel_sel // 2
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(10, 7))
    fig.suptitle(regions[jp], fontsize=14)
    for j, k in enumerate(parcel_sel):
        indx_roi = atlas_labels.index(k)
        roi_tcs = tcs[:, indx_roi, :]
        ax[j//ncols, j%ncols].plot(1e3 * evoked.times, roi_tcs.T)
        ax[j//ncols, j%ncols].set_xlim(-100, 200)
        ax[j//ncols, j%ncols].set_xticks([-100, 0, 50, 100, 150])
        ax[j//ncols, j%ncols].set_title(k)
        ax[j//ncols, j%ncols].grid(True)
        ax[j//ncols, j%ncols].set_xlabel('time (ms)')

  """  

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
img = ax.imshow(np.real(power.mean(axis=0)[0, :, :]), extent=[0,100,0,1], aspect='auto')
ax.set_yticklabels(freqs)
plt.show()
bar = plt.colorbar(img)

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

yticks = np.arange(16, freqs.shape[0], 20)
yticklabels = freqs[yticks]
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
img = ax.imshow(np.real(power.mean(axis=0)[0, :, :]), aspect='auto', origin='lower')
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_ylable('frequency (Hz)')
bar = plt.colorbar(img)
plt.show()
    

stc_new = stc[0].copy()
stc_new.data = stc_new.data.mean(axis=1).reshape(-1, 1)
stc_new.plot()

# Plot seed ROIs 
Brain = mne.viz.get_brain_class()
brain = Brain('subj_10', hemi='both', surf='white',
              subjects_dir=fs_subjs_dir, size=(800, 600))
brain.add_label(label[indx_seed_l], hemi='lh', color='green', borders=False)
brain.add_label(label[indx_seed_r], hemi='rh', color='green', borders=False)


src_path = os.path.join(fs_subjs_dir, 'subj_10/source/')
src = mne.read_source_spaces(os.path.join(src_path, 'file-src.fif'))
lh_tris = src[0]['tris']
rh_tris = src[1]['tris']
lh_vcs = src[0]['nn']
rh_vcs = src[1]['nn']

dat = np.random.randn(rh_vcs.shape[0])
from viz3dtools.matplotlib_surface_plotting import plot_surf
import nibabel as nb
import numpy as np

plot_surf( lh_vcs*100, lh_tris[:, [0, 2, 1]], [dat, dat], rotate=[90,270])



Brain = mne.viz.get_brain_class()
brain = Brain('subj_10', hemi='both', surf='white',
              subjects_dir=fs_subjs_dir, size=(800, 600))
brain.add_data(r[1, :], hemi='lh')

brain.add_annotation('aparc.a2009s', borders=False)
brain_kwargs = dict(alpha=0.1, background='white', cortex='low_contrast')
brain = mne.viz.Brain('subj_10', subjects_dir=fs_subjs_dir, **brain_kwargs)
kwargs = dict(fmin=stc.data.min(), fmax=stc.data.max(), alpha=0.25,
              smoothing_steps='nearest', time=stc.times)


# Plot seed ROI LH
brain_kwargs = dict(alpha=0.5, background='white', cortex='low_contrast')
brain = mne.viz.Brain('subj_10', subjects_dir=fs_subjs_dir, **brain_kwargs)
brain.add_label(label[indx_seed_l], hemi='lh', color='green', borders=True)
brain.show_view(azimuth=190, elevation=70, distance=350, focalpoint=(0, 0, 20))

# Plot seed ROI RH
brain = mne.viz.Brain('subj_10', subjects_dir=fs_subjs_dir, **brain_kwargs)
brain.add_label(label[indx_seed_r], hemi='rh', color='green', borders=True)
brain.show_view(azimuth=190, elevation=70, distance=350, focalpoint=(0, 0, 20))


brain.add_data(stc.lh_data, hemi='lh', vertices=stc.lh_vertno, **kwargs)
brain.add_data(stc.rh_data, hemi='rh', vertices=stc.rh_vertno, **kwargs)

