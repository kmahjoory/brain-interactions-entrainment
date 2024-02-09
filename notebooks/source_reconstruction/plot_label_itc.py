

import os, sys, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.signal as signal
import mne
sys.path.append('../../')
import src.preprocessing as prep
from src.source_space import extract_label_tc




plots_path = '../../datasets/plots/'
os.makedirs(plots_path, exist_ok=True)
data_path = '../../datasets/data/'
condition = 'encoding'



subjs_list = [10, 11, 12, 13]
itpc_group, xfreq_group, epow_group = [], [], []

# Set subject ID

for j, _fm in enumerate(np.arange(1, 4.4, 0.5)):

    itpc, xfreq, epow = [], [], []
    for subj_id in subjs_list:

        subj_name = f'subj_{subj_id}'
        meg_dir = os.path.join(data_path, f'subj_{subj_id}', 'meg')
        mri_dir = os.path.join(data_path, f'subj_{subj_id}', 'mri')
        fs_subjs_dir = os.path.join(data_path, 'fs_subjects_dir')
        srcs_dir = os.path.join(plots_path, subj_name, 'source_space')
        srcs_itc_dir = os.path.join(plots_path, subj_name, 'source_space_itc')
        srcs_itc_dir_files = os.path.join(srcs_itc_dir, 'files')

        itpc.append(np.load(os.path.join(srcs_itc_dir_files, f'itpc_{condition}_{_fm}.npy'), allow_pickle=True))
        xfreq.append(np.load(os.path.join(srcs_itc_dir_files, f'xfreq_{condition}_{_fm}.npy'), allow_pickle=True))
        epow.append(np.load(os.path.join(srcs_itc_dir_files, f'epow_{condition}_{_fm}.npy'), allow_pickle=True))

    itpc_group.append(itpc)
    xfreq_group.append(xfreq)
    epow_group.append(epow)



# Read individual labels
labels_indv, label2index_indv = [], []

for subj_id in subjs_list:

        subj_name = f'subj_{subj_id}'
        data_path = '../data'
        fs_subjs_dir = os.path.join(data_path, 'fs_subjects_dir')
        
        labels = mne.read_labels_from_annot(subject=subj_name, parc='BN_Atlas', hemi='both', 
                                        surf_name='white', annot_fname=None, regexp=None, 
                                        subjects_dir=fs_subjs_dir, sort=True, verbose=False)
        # Create label to index dictionary
        label2index = {label.name: i for i, label in enumerate(labels)}

        labels_indv.append(labels)
        #label2index_indv.append(label2index)



stg_l = ['A38m_L-lh', 'A41/42_L-lh', 'TE1.0/TE1.2_L-lh', 'A22c_L-lh',  'A22r_L-lh']    #'A38l_L-lh',
stg_r = ['A38m_R-rh', 'A41/42_R-rh', 'TE1.0/TE1.2_R-rh', 'A22c_R-rh', 'A22r_R-rh'] # , 'A38l_R-rh'

fig, ax = plt.subplots(2, 4, sharex=False, sharey=True, figsize=(14, 6))
fig.suptitle(f' Mean, ITPC,  {condition}', fontsize=16)
plt.subplots_adjust(hspace=0.4)
for j, fm in enumerate(['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0']):
    for label_name in stg_r+stg_l:
        ix_roi = label2index[label_name]
        _itpc = np.stack(itpc_group[j], axis=0).mean(axis=0)
        _xfreq = np.stack(xfreq_group[j], axis=0).mean(axis=0)
        ax[j//4, j%4].plot(_xfreq, _itpc[ix_roi, :].T)
        #ax[j//4, j%4].plot(_xfreq, _itpc[ix_roi, :].mean(axis=0), linewidth=2, color='k')
    
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
plt.savefig(os.path.join(plots_path, f'itpc_seeds_{condition}.png'), dpi=300)
plt.savefig(os.path.join(plots_path, f'itpc_seeds_{condition}.eps'), dpi=300)