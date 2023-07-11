
import numpy as np
from scipy import signal
from scipy import linalg
import mne
from .dsp import ssd


def extract_label_tc(stc, labels, label_name, franges, sfreq, mode='broad'):
    """
    This function extracts time course from source estimate within a label.
    Args:
        stc: mne source estimate object (list of epochs)
        label: mne label (string)
        frange: frequency range of interest 
                    list of two floats where signal band is [frange[0], frange[1]], and noise band is broad band
                    or list of three lists of two floats: e.g. [[9, 11], [6, 15], [8, 12]] According to SSD paper
        sfreq: sampling frequency (float)
        mode: 'broad' or 'narrow' (string)
    """
    # Create label to index dictionary
    label2index = {label.name: i for i, label in enumerate(labels)}
    ix_label = label2index[label_name]
    tcs_label = [stc[e].in_label(labels[ix_label]).data for e in range(len(stc))]
    # Broadband covariance matrix
    if len(franges) == 2:
        c_noise = np.array([np.cov(tcs_label[e]) for e in range(len(tcs_label))]).mean(axis=0)
        tcs_label_sig = mne.filter.filter_data(tcs_label, sfreq=sfreq, l_freq=franges[0], h_freq=franges[1], method='iir', verbose=False)
        c_sig = np.array([np.cov(tcs_label_sig[e]) for e in range(len(tcs_label_sig))]).mean(axis=0)

    # Narrowband covariance matrix
    elif len(franges) == 3 and all([len(franges[k]) == 2 for k in range(len(franges))]):
        tcs_label_sig = mne.filter.filter_data(tcs_label, sfreq=sfreq, l_freq=franges[0][0], h_freq=franges[0][1], method='iir', verbose=False)
        c_sig = np.array([np.cov(tcs_label_sig[e]) for e in range(len(tcs_label_sig))]).mean(axis=0)
        tcs_label_widesig = mne.filter.filter_data(tcs_label, sfreq=sfreq, l_freq=franges[1][0], h_freq=franges[1][1], method='iir', verbose=False)
        #  Band stop filtering, According to MNE impl. of filter l_freq > h_freq: band-stop filter
        tcs_label_noise = mne.filter.filter_data(tcs_label_widesig, sfreq=sfreq, l_freq=franges[2][1], h_freq=franges[2][0], method='iir', verbose=False)
        c_sig = np.array([np.cov(tcs_label_noise[e]) for e in range(len(tcs_label_noise))]).mean(axis=0)    
    
    # spatial spectral decomposition (W is the spatial filter along which to project the label time courses)
    W = ssd(c_sig, c_noise)
    if np.all(np.isreal(W)):
        W = np.real(W)
    # Single time course per label
    tc_label = W[:, [0]].T @ np.array(tcs_label)

    return tc_label, W, tcs_label