import os
import numpy as np
import scipy.io as scpio
import mne
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
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
subj_id = 10

# This will go through all block files and will load them, set the grad compensation
# to 3, notch-filter range(50, 300, 30) and firlter in 0.5-130, and then 
# down-sample it
prep.preprocess_blocks(subj_id=subj_id, analysis_dir=analysis_dir)

