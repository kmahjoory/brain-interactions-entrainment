import os, sys
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

sys.path.append("../../")
from src import preprocessing as prep

#%matplotlib qt

#os.environ['ETS_TOOLKIT'] = 'qt4'
#os.environ['QT_API'] = 'pyqt5'


# Set the analysis directory where all plots will be saved
subj_id = 28
subjs_dir = f'../../datasets/data/'
plots_dir = f'../../datasets/plots/'


# Specify subject index (A number between 1-27)
# #############################################################################
prep.concat_blocks(subj_id=subj_id, subjs_dir=subjs_dir, plots_path=plots_dir)
