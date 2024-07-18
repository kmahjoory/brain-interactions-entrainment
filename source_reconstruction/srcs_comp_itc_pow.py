

import os, sys, glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.signal as signal
import mne
sys.path.append('../../')
import src.preprocessing as prep

### Set directories
for subj_id in range(1, 2):

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
        meg = mne.io.read_raw_fif(os.path.join(meg_dir, f'after_ica_subj_{subj_id}_meg.fif'))
    else:
        meg = mne.io.read_raw_fif(os.path.join(meg_dir, f'after_ica_meg.fif'))

    # Load atlas labels
    labels = mne.read_labels_from_annot(subject=subj_name, parc='BN_Atlas', hemi='both', 
                                surf_name='white', annot_fname=None, regexp=None, 
                                subjects_dir=fs_subjs_dir, sort=True, verbose=False)
    # Load forward model
    fwd = mne.read_forward_solution(os.path.join(mri_dir, 'file-fwd.fif'))

    itpc, xfreq, epow, tpow, F = [], [], [], [], []


    for j, fm in enumerate(['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0']):


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
        # Create label to index dictionary
        label2index = {label.name: i for i, label in enumerate(labels)}

        # Time courses for all epochs and all labels (pca across dipoles)
        stc_labels = np.array([k.extract_label_time_course(labels=labels, src=inv_operator['src'], mode='pca_flip', verbose=False) for k in stc])
        stc_labels.shape # (num_epochs, num_labels, num_times)

        _itpc, _xfreq, _epow, _tpow, _F = prep.calc_itc(stc_labels, meg.info['sfreq'])
        itpc.append(_itpc)
        xfreq.append(_xfreq)
        epow.append(_epow)
        tpow.append(_tpow)
        F.append(_F)


    ## EPOW
    stg_l = ['A38m_L-lh', 'A41/42_L-lh', 'TE1.0/TE1.2_L-lh', 'A22c_L-lh',  'A22r_L-lh']    #'A38l_L-lh',
    stg_r = ['A38m_R-rh', 'A41/42_R-rh', 'TE1.0/TE1.2_R-rh', 'A22c_R-rh', 'A22r_R-rh'] # , 'A38l_R-rh'

    fig, ax = plt.subplots(2, 4, sharex=False, sharey=True, figsize=(14, 6))
    fig.suptitle(f'{subj_name}, evoked power, {condition}', fontsize=16)
    plt.subplots_adjust(hspace=0.4)
    for j, fm in enumerate(['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0']):
        for label_name in stg_r+stg_l:
            ix_roi = label2index[label_name]
            ax[j//4, j%4].plot(xfreq[j], epow[j][ix_roi, :].T)
        
        ax[j//4, j%4].set_title(f'fm= {fm} Hz')
        ax[j//4, j%4].set_xlim(0.5, 4.3)
        ax[j//4, j%4].set_ylim(0, 2000)
        ax[j//4, j%4].set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4])
        #ax[j//4, j%4].set_yticks([])
        ax[j//4, j%4].spines['top'].set_visible(False)
        ax[j//4, j%4].spines['right'].set_visible(False)
        #ax[j//4, j%4].spines['left'].set_visible(False)
        #ax[j//4, j%4].grid(True)
        ax[0, 0].set_ylabel('ePow')
        ax[1, 0].set_ylabel('ePow')
        if j==5:
            fig.legend(stg_r+stg_l, loc='lower right');
        
    plt.delaxes(ax[1, 3]) ;
    plt.savefig(os.path.join(srcs_itc_dir, f'epow_seeds_{condition}.png'), dpi=300)
    #plt.savefig(os.path.join(srcs_itc_dir, f'epow_seeds_{condition}.eps'), dpi=300)
    plt.close()


    ## ITPC
    stg_l = ['A38m_L-lh', 'A41/42_L-lh', 'TE1.0/TE1.2_L-lh', 'A22c_L-lh',  'A22r_L-lh']    #'A38l_L-lh',
    stg_r = ['A38m_R-rh', 'A41/42_R-rh', 'TE1.0/TE1.2_R-rh', 'A22c_R-rh', 'A22r_R-rh'] # , 'A38l_R-rh'

    fig, ax = plt.subplots(2, 4, sharex=False, sharey=True, figsize=(14, 6))
    fig.suptitle(f'{subj_name}, ITPC,  {condition}', fontsize=16)
    plt.subplots_adjust(hspace=0.4)
    for j, fm in enumerate(['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0']):
        for label_name in stg_r+stg_l:
            ix_roi = label2index[label_name]
            ax[j//4, j%4].plot(xfreq[j], itpc[j][ix_roi, :].T)
        
        ax[j//4, j%4].set_title(f'fm= {fm} Hz')
        ax[j//4, j%4].set_xlim(0.5, 4.3)
        #ax[j//4, j%4].set_ylim(0, 2000)
        ax[j//4, j%4].set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4])
        #ax[j//4, j%4].set_yticks([])
        ax[j//4, j%4].spines['top'].set_visible(False)
        ax[j//4, j%4].spines['right'].set_visible(False)
        #ax[j//4, j%4].spines['left'].set_visible(False)
        #ax[j//4, j%4].grid(True)
        ax[0, 0].set_ylabel('ITPC')
        ax[1, 0].set_ylabel('ITPC')
        if j==5:
            fig.legend(stg_r+stg_l, loc='lower right');
        
    plt.delaxes(ax[1, 3]) ;
    plt.savefig(os.path.join(srcs_itc_dir, f'itpc_seeds_{condition}.png'), dpi=300)
    #plt.savefig(os.path.join(srcs_itc_dir, f'itpc_seeds_{condition}.eps'), dpi=300)
    plt.close()

    # Save data
    if 0:
        srcs_itc_dir_files = os.path.join(srcs_itc_dir, 'files')
        os.makedirs(srcs_itc_dir_files, exist_ok=True)
        for j, _fm in enumerate(np.arange(1, 4.4, 0.5)):
            np.save(os.path.join(srcs_itc_dir_files, f'itpc_{condition}_{_fm}.npy'), itpc[j], allow_pickle=True)
            np.save(os.path.join(srcs_itc_dir_files, f'xfreq_{condition}_{_fm}.npy'), xfreq[j], allow_pickle=True)
            np.save(os.path.join(srcs_itc_dir_files, f'epow_{condition}_{_fm}.npy'), epow[j], allow_pickle=True)