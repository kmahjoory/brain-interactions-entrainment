
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--subj_ids', action='store', type=int, nargs=2)
my_parser.add_argument('--dim_red', action='store', type=str, nargs='*', default=['ssd', 'pca'])
args = my_parser.parse_args()
subjects_id = range(args.subj_ids[0], args.subj_ids[1]+1)
dim_red = args.dim_red



import os, sys, glob
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
import scipy
import scipy.signal as signal
import mne
sys.path.append('../../')
import src.preprocessing as prep
from src.source_space import extract_label_tc

print('#############################################')
print('RUN specifications:')
print(subjects_id)
print(dim_red)
print('#############################################')


### Set directories
for subj_id in subjects_id:

    condition = 'encoding'#'maintenance'
    condition_acr = 'e'#'m'

    subj_name = f'subj_{subj_id}'
    print(os.listdir('../'))
    plots_path = '../../datasets/plots/'
    data_path = '../../datasets/data/'
    meg_dir = os.path.join(data_path, f'subj_{subj_id}', 'meg')
    mri_dir = os.path.join(data_path, f'subj_{subj_id}', 'mri')
    fs_subjs_dir = os.path.join(data_path, 'fs_subjects_dir')

    srcs_dir = os.path.join(plots_path, subj_name, 'source_space')
    os.makedirs(srcs_dir, exist_ok=True)
    srcs_itc_dir = os.path.join(plots_path, subj_name, 'source_space_itc', condition)
    os.makedirs(srcs_itc_dir, exist_ok=True)

    # Load after-ICA MEG data
    if os.path.isfile(os.path.join(meg_dir, f'after_ica_subj_{subj_id}_meg.fif')):
        meg = mne.io.read_raw_fif(os.path.join(meg_dir, f'after_ica_subj_{subj_id}_meg.fif'), verbose=False)
    else:
        meg = mne.io.read_raw_fif(os.path.join(meg_dir, f'after_ica_meg.fif'), verbose=False)

    # Load atlas labels
    labels = mne.read_labels_from_annot(subject=subj_name, parc='BN_Atlas', hemi='both', 
                                surf_name='white', annot_fname=None, regexp=None, 
                                subjects_dir=fs_subjs_dir, sort=True, verbose=False)
    # Load forward model
    fwd = mne.read_forward_solution(os.path.join(mri_dir, 'file-fwd.fif'))

    itpc_pca, epow_pca, xfreq_pca, itpc_ssd, epow_ssd, xfreq_ssd = [], [], [], [], [], []


    for j, fm in enumerate(['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0']):
    #for j, fm in enumerate(['1.0']):


        print(f'===> Processing subj: {subj_id}    fm= {fm} Hz')


        # Estimate noine covariance matrices from all trials, tmin=-.2 tmax=-0.05 
        _epoch = prep.mk_epochs(meg.copy(), mod_freq=float(fm), tmin=-0.2, baseline=None, 
                        annot_pattern=f'{condition_acr}/{fm}', new_event_value=101)
        noise_cov = mne.compute_covariance(_epoch, tmin=-0.2, tmax=-0.05,
                                        method='empirical')

        # Obtain separate Epochs for phase 0 and pi
        epoch0 = prep.mk_epochs(meg.copy(), mod_freq=float(fm), tmin=0, baseline=None, 
                        annot_pattern=f'{condition_acr}/{fm}/e0/', new_event_value=101)
        epochpi = prep.mk_epochs(meg.copy(), mod_freq=float(fm), tmin=0, baseline=None, 
                        annot_pattern=f'{condition_acr}/{fm}/epi/', new_event_value=101)
        print(epoch0.get_data().shape, epoch0.info['sfreq'])
        print(epochpi.get_data().shape, epochpi.info['sfreq'])

        # Shift phase of pi trials by pi
        sig_shifted = prep.shift_phase(epochpi.copy().get_data(), dphi=np.pi, f=float(fm), fs=meg.info['sfreq'])
        print(sig_shifted.shape)

        # Create new Epochs object for shifted pi trials
        epochpi_shifted = mne.EpochsArray(sig_shifted, epochpi.info, tmin=epochpi.tmin)

        #epochpi_shifted.plot_psd(fmin=0, fmax=100, average=True, spatial_colors=False, n_jobs=1, show=False)

        # Concatenate epochs with phase 0 and shifted pi trials
        epoch = mne.concatenate_epochs([epoch0, epochpi_shifted])

        # Estimate data covariance from aligned trials, tmin=0 tmax=8/fm
        data_cov = mne.compute_covariance(epoch, tmin=0.0, tmax=8/float(fm),
                                        method='empirical')

        # Visualize noise and data covariance matrices
        #with mne.viz.use_browser_backend('matplotlib'):
        #    mne.viz.plot_cov(noise_cov, meg.info)

        # Source Reconstruction: MNE
        inv_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, data_cov,
                                                            loose=0.2, depth=0.8, verbose=True)
        method = "dSPM"
        snr = 3.
        lambda2 = 1. / snr ** 2
        stc = mne.minimum_norm.apply_inverse_epochs(epoch, inv_operator, lambda2, method=method, 
                                pick_ori='normal', verbose=False)
        
        # Read Atlas labels for subject
        labels = mne.read_labels_from_annot(subject=subj_name, parc='BN_Atlas', hemi='both', 
                                        surf_name='white', annot_fname=None, regexp=None, 
                                        subjects_dir=fs_subjs_dir, sort=True, verbose=False)
        
        ### PCA based calculation of ITC
        # Create label to index dictionary

        label2index = {label.name: i for i, label in enumerate(labels)}
        if 'pca' in dim_red:
            # Time courses for all epochs and all labels (pca across dipoles)
            stc_labels = np.array([k.extract_label_time_course(labels=labels, src=inv_operator['src'], mode='pca_flip', verbose=False) for k in stc])
            stc_labels.shape # (num_epochs, num_labels, num_times)

            _itpc, _xfreq, _epow, *_ = prep.calc_itc(stc_labels, meg.info['sfreq'])
            itpc_pca.append(_itpc)
            epow_pca.append(_epow)
            xfreq_pca.append(_xfreq)

        ### SSD based calculation of ITC
        if 'ssd' in dim_red:
            franges = [float(fm)-.1*float(fm), float(fm)+.1*float(fm)]
            stc_labels_ssd = np.array([extract_label_tc(stc, label_name=l.name, labels=labels,
                                                franges=franges, sfreq=meg.info['sfreq'], mode='broad')[0] for l in labels])

            _itpc, _xfreq, _epow, *_ = prep.calc_itc(np.swapaxes(stc_labels_ssd.squeeze(), 0, 1), meg.info['sfreq'])
            itpc_ssd.append(_itpc)
            epow_ssd.append(_epow)
            xfreq_ssd.append(_xfreq)

        print(f"===> {fm} Hz done!")


    srcs_itc_dir_files = os.path.join(srcs_itc_dir, 'files')
    os.makedirs(srcs_itc_dir_files, exist_ok=True)

    if 'ssd' in dim_red:
        for j, _fm in enumerate(np.arange(1, 4.4, 0.5)):
            np.save(os.path.join(srcs_itc_dir_files, f'itpc_{condition}_{_fm}_ssd.npy'), itpc_ssd[j], allow_pickle=True)
            np.save(os.path.join(srcs_itc_dir_files, f'xfreq_{condition}_{_fm}_ssd.npy'), xfreq_ssd[j], allow_pickle=True)
            np.save(os.path.join(srcs_itc_dir_files, f'epow_{condition}_{_fm}_ssd.npy'), epow_ssd[j], allow_pickle=True)

    if 'pca' in dim_red:
        for j, _fm in enumerate(np.arange(1, 4.4, 0.5)):
            np.save(os.path.join(srcs_itc_dir_files, f'itpc_{condition}_{_fm}_pca.npy'), itpc_pca[j], allow_pickle=True)
            np.save(os.path.join(srcs_itc_dir_files, f'xfreq_{condition}_{_fm}_pca.npy'), xfreq_pca[j], allow_pickle=True)
            np.save(os.path.join(srcs_itc_dir_files, f'epow_{condition}_{_fm}_pca.npy'), epow_pca[j], allow_pickle=True)