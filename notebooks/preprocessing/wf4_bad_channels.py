import os
import numpy as np
import scipy.io as scpio
import mne
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
import glob
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from nf_tools import preprocessing as prep
from nf_tools import utils

#os.environ['ETS_TOOLKIT'] = 'qt4'
#os.environ['QT_API'] = 'pyqt5'


# Set the analysis directory where all plots will be saved
analysis_dir = '/mnt/beegfs/workspace/2021-0292-NeuroFlex/prj_neuroflex/k1_analyses'


# Specify subject index (A number between 1-27)
# #############################################################################
subj_indx = 12
path = utils.set_dirs(subj_id=subj_indx, analysis_dir=analysis_dir)
meg_dir = path['subj_meg_dir']
prep_dir = path['prep_dir']


### Specify block number for Visual inspection of time series
# #############################################################################
block = 9

meg_block_name = f'block_{block}_meg.fif'
raw = mne.io.read_raw_fif(os.path.join(meg_dir, meg_block_name))

%matplotlib qt

#raw.plot_psd(fmax=40, n_fft=int(raw.info['sfreq']*2), n_overlap=int(raw.info['sfreq']))

fig = raw.plot(duration=60, n_channels=40)
fig.canvas.key_press_event('a')


raw.plot_psd()
raw.plot_psd(fmax=40)


### Save channel/time span rejected data and its PSD plot
# #############################################################################
if 0:
    fig1 = raw.plot_psd(fmax=150)
    os.makedirs(os.path.join(prep_dir, 'psd_filtered_downsampled'), exist_ok=True)
    fig1.savefig(os.path.join(prep_dir, 'psd_filtered_downsampled', f'psd_block_{block}_tsrej.jpg'))
    plt.close('all')
    
    
    write_name = f"block_{block}" + "_meg_tsrej.fif"  
    raw.save(os.path.join(meg_dir, write_name), overwrite=False)






