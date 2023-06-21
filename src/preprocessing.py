import os
import numpy as np
import scipy.io as scpio
import mne
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
from scipy.fft import fft, ifft, fftfreq
import scipy
import scipy.signal as signal
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

from .utils import *


def fix_ch_types(raw):
    """ This function fixes the channel types issue, where gradiometers are detected as magnetomenter!
    Arguments:
        raw: raw object in mne format"""
    ch_names = raw.info['ch_names']
    ch_types = raw.get_channel_types()
    mapping = dict(zip(raw.info['ch_names'], raw.get_channel_types()))
    for k, v in mapping.items():
        if v == 'mag':
            mapping[k] = 'grad'
        if k == 'UPPT001':
            mapping[k] = 'stim'
        if k == 'UADC004-3609':
            mapping[k] = 'stim'
    raw.set_channel_types(mapping)
    mssg = f"\t N-Grads={raw.get_channel_types().count('grad')} \t N-Ref-Mags={raw.get_channel_types().count('ref_meg')} \t "
    mssg += f"N-other= {raw.get_channel_types().count('misc')}"
    print(mssg)
    return raw


def filter_block(raw):
    """"""

    # Remove line Noise only from Grad channels
    freqs_notch = (50, 100, 150, 200, 250, 300)
    raw.load_data()
    raw = raw.notch_filter(freqs=freqs_notch)
    # raw.plot_psd()

    # Apply HP and LP filters at 0.5Hz and 130Hz
    raw = raw.filter(l_freq=0.5, h_freq=130, h_trans_bandwidth=130/8)
    # raw.plot_psd(fmin=0, fmax=150)
    return raw


def preprocess_blocks(subj_id, subjs_dir, plots_path, write=False):
    """ 
    This function loads all block files and notch filters at range(50, 300, 50), HP
    filters at 1 Hz and LP filters at 130 
    
    Args:
        subj_id (int): ID of the subject to be preprocessed 
        subjs_dir (str):  Path to the directory where all subjects data are read from and stored in. 
        plots_path: all output plots will be saved here.
        
    Returns: 
        No output. Preprocessed files are saved in the following directory
     
    """

    subj_name = f"subj_{subj_id}"
    meg_dir = os.path.join(subjs_dir, subj_name, 'meg')
    prep_dir = os.path.join(plots_path, subj_name)
    os.makedirs(meg_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)

    # load blocks info
    meg_blocks_info = pd.read_csv(os.path.join(meg_dir, 'nf_blocks_info.csv'), header=0)
    n_meg_blocks = meg_blocks_info.shape[0]
    for block in range(1, n_meg_blocks+1):  # to include the last block as well
        meg_block_name = f"block_{block}_raw.fif"  # meg_blocks_info.origname[block-1]
        raw = mne.io.read_raw_fif(os.path.join(meg_dir, meg_block_name))

        # plot and save
        if write:
            fig = raw.plot_psd(fmax=160)
            os.makedirs(os.path.join(prep_dir, 'psd_raw'), exist_ok=True)
            fig.savefig(os.path.join(prep_dir, 'psd_raw', f'psd_block_{block}.jpg'))
        
        # Apply 3rd gradient compensation
        raw = raw.apply_gradient_compensation(grade=3, verbose=None)
        
        # plot and save
        if write:
            fig = raw.plot_psd(fmax=160)
            os.makedirs(os.path.join(prep_dir, 'psd_3rd_grad'), exist_ok=True)
            fig.savefig(os.path.join(prep_dir, 'psd_3rd_grad', f'psd_block_{block}.jpg'))

        # Filter block raw
        raw = filter_block(raw)

        # plot and save
        if write:
            fig = raw.plot_psd(fmax=160)
            os.makedirs(os.path.join(prep_dir, 'psd_filtered'), exist_ok=True)
            fig.savefig(os.path.join(prep_dir, 'psd_filtered', f'psd_block_{block}.jpg'))

        # Pick MEG (Gradiometers) channels & Trigger channels.
        #indx_good_channels = np.where(np.isin(raw.get_channel_types(), ['mag', 'stim']))[0]
        indx_good_channels = np.where(np.isin(raw.get_channel_types(), ['mag', 'ref_meg']))[0]
        good_channels = np.array(raw.ch_names)[indx_good_channels]
        meg = raw.pick_channels(good_channels)

        # Resample data to 300 Hz
        meg = meg.resample(300)

        if write:
            fig = raw.plot_psd(fmax=150)
            os.makedirs(os.path.join(prep_dir, 'psd_filtered_downsampled'), exist_ok=True)
            fig.savefig(os.path.join(prep_dir, 'psd_filtered_downsampled', f'psd_block_{block}.jpg'))
            plt.close('all')
        
            fig = raw.plot_psd(fmax=40)
            os.makedirs(os.path.join(prep_dir, 'psd_filtered_downsampled'), exist_ok=True)
            fig.savefig(os.path.join(prep_dir, 'psd_filtered_downsampled', f'psd_block_{block}_40hz.jpg'))
            plt.close('all')

        # Save the data for block
        if write:
            write_name = f"block_{block}" + "_meg.fif"  # common meg files should be saved in this format
            meg.save(os.path.join(meg_dir, write_name), overwrite=True)
    # Print list of files in meg directory
    os.system(f"ls -lh {meg_dir}*/")


def concat_blocks(subj_id, subjs_dir, plots_path):
    """This function concatenates blocks"""

    subj_name = f"subj_{subj_id}"
    meg_dir = os.path.join(subjs_dir, subj_name, 'meg')
    prep_dir = os.path.join(plots_path, subj_name)
    os.makedirs(prep_dir, exist_ok=True)
    
    # Search for clean (filtered, down-sampled, and noisy time span removed) data
    # and sort them based on their names, to get right order for concatenation
    block_fnames = sorted(glob.glob(os.path.join(meg_dir, "block*_meg.fif")))
    print("--------------------------")
    print(block_fnames)
    print("--------------------------")
    
    # For concatenation use 'block_*_meg_tsrej.fif' if exists, otherwise use 'block_*meg.fif'
    blocks_all, blocks_all_fnames = [], []
    
    for jblock, block_fname in enumerate(block_fnames):
        block_tsrej_fname = block_fname[:-4] + '_tsrej.fif'
        if os.path.isfile(block_tsrej_fname):
            blocks_all.append(mne.io.read_raw_fif(block_tsrej_fname))
            blocks_all_fnames.append(os.path.basename(block_tsrej_fname))
        else:
            blocks_all.append(mne.io.read_raw_fif(block_fname))
            blocks_all_fnames.append(os.path.basename(block_fname))
    
    # Change head trans matrix of all blocks to the 1st
    for jblock in range(1, len(blocks_all)):
        blocks_all[jblock].info['dev_head_t'] = blocks_all[0].info['dev_head_t']      
    
    # Concatenate blocks
    meg = mne.concatenate_raws(blocks_all)
    
    # Write concatenated MEG
    write_path = meg_dir
    write_name = "concat_meg.fif"
    meg.save(os.path.join(write_path, write_name))
    
    # Save blocks name
    with open(os.path.join(meg_dir, 'concatenated_blocks_order.txt'), 'w') as f:
        f.write(' '.join(blocks_all_fnames))
        
    # Save PSD of concatenated data
    fig1 = meg.plot_psd(fmax=150)
    fig1.savefig(os.path.join(prep_dir, 'psd_concatenated.jpg'))
    plt.close('all')




def find_events_frequency(trig_sig):
    """This function finds the events and their frequency on Trigger signal"""
    if isinstance(trig_sig, (np.ndarray, np.generic)):
        trig_sig = trig_sig.ravel().tolist()  # convert to list
    events = list(set(trig_sig))
    events.remove(0)  # Ignore zero values
    event_freqs = []
    for event in events:
        cnt = 0
        for indx, val in enumerate(trig_sig):
            if (val == event) and (trig_sig[indx - 1] != event):
                cnt += 1
        event_freqs.append(cnt)
    return(dict(zip(events, event_freqs)))


def mk_epochs(meg, tmin, baseline, mod_freq, annot_pattern, new_event_value):
    """This function creates epochs based on specified mod_freq and annotation_pattern
    Arguments:
        meg: annotated MNE object, where bad time spans are annotated as BAD_*
        tmin: start time of the epoch in seconds. This parameter should match with baseline.
        baseline: to specify baseline correction. e.g. tmin=0, baseline=(0, 0) applies no baseline correction.
        mod_freq: to specify the modulating frequency of interest. This parameter should match with the annot_pattern
        annot_pattern: The annotation pattern, based on which epochs are created.
        new_event_value: to specify new label for events. The default is 100. Optional argument.
    Returns:
        epoch: Epoch MNE object.
    Example1:
        The following call will create epochs from mod_freq=1 and encoding events, and will apply baseline correction
        between -0.5 and 0. The length of created epochs would be 8/mod_freq + abs(tmin) = 8 + 0.5 = 8.5 seconds or
        8.5 * sfreq = 8.5 * 300 samples
        mk_epochs(meg.copy(), mod_freq=1., tmin=-0.5, baseline=(-0.5, 0), annot_pattern='e/1.0/')
    Example2:
        create epochs from maintenance events with mod_freq=3.5
        mk_epochs(meg.copy(), mod_freq=3.5, tmin=0, baseline=(0, 0), annot_pattern='m/3.5/')
    Notes:
        tmin and annotation patterns should match.
        baseline = (None, 0) sets baseline to MNE defaults.
        baseline = (0, 0), tmin = 0 sets to no baseline.
        baseline = (-0.5, 0), tmin = -0.5
    """
    if not new_event_value:
        new_event_value = 100
    events = mne.events_from_annotations(meg)
    annot = list(events[1].keys())
    indx_pattern = np.where([annot_pattern in k for k in annot])[0].tolist()
    event_vals_pattern = np.array(list(events[1].values()))[indx_pattern].tolist()
    indx_events = np.where(np.isin(events[0][:, 2], event_vals_pattern))[0].tolist()
    events4epoch = events[0][indx_events, :]
    events4epoch[:, 2] = new_event_value
    annot_epoch = mne.annotations_from_events(events4epoch, meg.info['sfreq'])
    meg.set_annotations(annot_epoch)
    epoch = mne.Epochs(meg, events=events4epoch, tmin=tmin, tmax=(8/mod_freq)-1/meg.info['sfreq'], baseline=baseline)
    return epoch


def run_ica(subj_id, subjs_dir, write=True, overwrite=False):
    
    subj_name = f"subj_{subj_id}"
    meg_dir = os.path.join(subjs_dir, subj_name, 'meg')
    meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'concat_meg.fif'))
    
    # Apply 1Hz HP filter before ICA
    meg = meg.load_data().filter(l_freq=1, h_freq=None)
    
    ica = ICA(n_components=40, method='infomax', fit_params=dict(extended=True))
    ica.fit(meg)
    if write:
        ica.save(os.path.join(meg_dir, "matrix_ica.fif"))

            
    # Print list of files in meg directory
    os.system(f"ls -lh {meg_dir}*/")
    return ica



def pick_topchans_erf(evoked):
    # Creat channel mask for right and left hemispheres
    mask_rh = [k[1]=='R' for k in evoked.ch_names]
    mask_lh = [k[1]=='L' for k in evoked.ch_names]
    
    
    # Compute mean-amplitude between 95-110 ms
    sample_min = evoked.time_as_index(0.095).item()
    sample_max = evoked.time_as_index(0.11).item()
    erf100 = evoked.data[:, sample_min:sample_max].mean(axis=1)
    
    #
    indx30_largest = np.argsort(erf100)[-30:]
    indx10lh_largest = np.argsort(erf100 * mask_lh)[-10:]
    indx10rh_largest = np.argsort(erf100 * mask_rh)[-10:]
    
    return {'indx30_chan': indx30_largest, 'indx10lh_chan': indx10lh_largest, 'indx10rh_chan': indx10rh_largest}

    
    
def plot_erf(evoked, indx30, indx10lh, indx10rh, prep_dir=None):
    # Based on mean-amplitude, pick 30 channels with max amplitude
    n_chans = evoked.data.shape[0]
    n_tps = evoked.data.shape[1]
    mask = np.array([False] * n_chans)
    mask[indx30] = True
    mask = np.tile(mask.reshape((n_chans, 1)), n_tps)
    
    mask_10lh = np.array([False] * n_chans)
    mask_10lh[indx10lh] = True
    mask_10lh = np.tile(mask_10lh.reshape((n_chans, 1)), n_tps)
    
    mask_10rh = np.array([False] * n_chans)
    mask_10rh[indx10rh] = True
    mask_10rh = np.tile(mask_10rh.reshape((n_chans, 1)), n_tps)
    

    # Plot evoked data of 30 channels with largest amplitude
    plt.rcParams['axes.grid'] = True
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 7))
    evoked.plot(xlim=(-.2, 1), picks=indx30, axes=ax[0])
    ax[0].set_title('30 top')
    ax[0].set_xlabel('')
    evoked.plot(xlim=(-.2, 1), picks=indx10lh, axes=ax[1])
    ax[1].set_title('10 top LH')
    ax[1].set_xlabel('')
    evoked.plot(xlim=(-.2, 1), picks=indx10rh, axes=ax[2])
    ax[2].set_title('10 top RH')
    ax[2].set_xlabel('Time (ms)')
    custom_xticks = np.array([-.1, 0, .05, .1, .2, .4, .6, .8])
    plt.setp(ax, xticks=custom_xticks, xticklabels=(custom_xticks*1000).astype(int))
    if prep_dir:
        write_path = os.path.join(prep_dir, 'erf')
        os.makedirs(write_path, exist_ok=True)
        fig.savefig(os.path.join(write_path, 'erf_time.jpg'))
        plt.close('all')
    
    # Topo plots
    times = np.arange(0.09, 0.13, 0.005)
    fig1 = evoked.plot_topomap(times, ch_type='mag', average=0.05, time_unit='s', mask=mask, mask_params=dict(markersize=7, markerfacecolor='y'))
    fig2 = evoked.plot_topomap(times, ch_type='mag', average=0.05, time_unit='s', mask=mask_10lh, mask_params=dict(markersize=7, markerfacecolor='y'))
    fig3 = evoked.plot_topomap(times, ch_type='mag', average=0.05, time_unit='s', mask=mask_10rh, mask_params=dict(markersize=7, markerfacecolor='y'))
    if prep_dir:
        fig1.savefig(os.path.join(write_path, 'erf_topo.jpg'))
        fig2.savefig(os.path.join(write_path, 'erf_topo_lh.jpg'))
        fig3.savefig(os.path.join(write_path, 'erf_topo_rh.jpg'))
        plt.close('all')
    
    

def shift_phase(sig, dphi, f, fs):
    """
    This function shifts the phase at a certain frequency by pi degree
    (works well only for dphi=pi, not tested for other dphi)
    Arguments:
        sig: A 2d or 3d signal, where time is the last dimension
        dphi: phase shift e.g. np.pi, np.pi/2
        f: the frequecy on which phase shift is applied.
        fs: sampling frequency
    Returns:
        phase shifted signal
    Example 1:
        fs = 300
        timestep = 1/fs
        t = np.arange(0, 2, timestep).reshape(1, -1)
        ntps = t.shape[0]
        f1, f2, f3 = .7, 1, 125
        amp = 1
        phi = 0
        dphi = np.pi
        f = 1
        time_span_shift = round((fs/f) * dphi/(2*np.pi))
        sig = amp * np.sin(2*np.pi*f1*t + np.pi) + np.sin(2*np.pi*f2*t + np.pi/6) + np.sin(2*np.pi*f3*t + phi)
        sig_shifted = shift_phase(sig, dphi=dphi, f=f, fs=fs)
        
        plt.rcParams['axes.grid'] = True
        fig, ax = plt.subplots(2, 1, figsize=(13, 4), sharex=True, sharey=True)
        ax[0].plot(t.T, sig.T)
        ax[1].plot(t.T, sig_shifted.T, label='Phase Shifted', alpha=.5, linewidth=2)
        ax[1].plot(t[:, :-time_span_shift].T, sig[:, time_span_shift:].T, label='Time Shifted', color=[1, 0, 0], alpha=.6)
        xticks = np.arange(0, 2, 0.1)
        plt.setp(ax, xlim=(0, 2), xticks=xticks)
        plt.legend()
        plt.show()
    """
    ndim = np.ndim(sig)
    if ndim == 1:
        sig = sig.reshape(1, sig.shape[0])
    if ndim == 3:
        nd1, nd2, nd3 = sig.shape
        sig = sig.reshape(-1, sig.shape[2])
        
    ntps = sig.shape[1]
    timestep = 1/fs
    sigfft = fft(sig)
    freqs = fftfreq(ntps, d=timestep)
    sigfft_shifted = sigfft * np.exp(np.array([1j])*dphi*freqs/f)
    sig_shifted = ifft(sigfft_shifted)
    if ndim == 3:
        sig_shifted = sig_shifted.reshape(nd1, nd2, nd3)
        
    return np.real(sig_shifted)
        

def mk_epochs_new(meg,  mod_freq=None, tmin=None, tmax=None, baseline=None, annot_pattern='', new_event_value=100):
    """This function creates epochs based on specified mod_freq and annotation_pattern
    Arguments:
        meg: annotated MNE object, where bad time spans are annotated as BAD_*
        tmin: start time of the epoch in seconds. This parameter should match with baseline.
        baseline: to specify baseline correction. e.g. tmin=0, baseline=(0, 0) applies no baseline correction.
        mod_freq: to specify the modulating frequency of interest. This parameter should match with the annot_pattern
        annot_pattern: The annotation pattern, based on which epochs are created.
        new_event_value: to specify new label for events. The default is 100. Optional argument.
    Returns:
        epoch: Epoch MNE object.
    Example1:
        The following call will create epochs from mod_freq=1 and encoding events, and will apply baseline correction
        between -0.5 and 0. The length of created epochs would be 8/mod_freq + abs(tmin) = 8 + 0.5 = 8.5 seconds or
        8.5 * sfreq = 8.5 * 300 samples
        mk_epochs(meg.copy(), mod_freq=1., tmin=-0.5, baseline=(-0.5, 0), annot_pattern='e/1.0/')
    Example2:
        create epochs from maintenance events with mod_freq=3.5
        mk_epochs(meg.copy(), mod_freq=3.5, tmin=0, baseline=(0, 0), annot_pattern='m/3.5/')
    Notes:
        tmin and annotation patterns should match.
        baseline = (None, 0) sets baseline to MNE defaults.
        baseline = (0, 0), tmin = 0 sets to no baseline.
        baseline = (-0.5, 0), tmin = -0.5
    """
    if not tmax:
        tmax = (8 / mod_freq) - 1 / meg.info['sfreq']
    events = mne.events_from_annotations(meg)
    annot = list(events[1].keys())
    indx_pattern = np.where([annot_pattern in k for k in annot])[0].tolist()
    event_vals_pattern = np.array(list(events[1].values()))[indx_pattern].tolist()
    indx_events = np.where(np.isin(events[0][:, 2], event_vals_pattern))[0].tolist()
    events4epoch = events[0][indx_events, :]
    events4epoch[:, 2] = new_event_value
    annot_epoch = mne.annotations_from_events(events4epoch, meg.info['sfreq'])
    meg.set_annotations(annot_epoch)
    epoch = mne.Epochs(meg, events=events4epoch, tmin=tmin, tmax=tmax, baseline=baseline)
    return epoch


def calc_itc(sig, sfreq):
    """
    This function computes ITPC from epoched time series

    Args:
        sig: epochs x channels x time points
        sfreq: sampling frequency
        
    Returns:
        itpc: Inter-trial phase coherence.
    """
    nepoch, nchan, ntp = sig.shape
    sig = np.reshape(sig, (-1, ntp))  # convert epochs @ channels @ time-points to  (epochs . channels) @ time-points
    nfft = ntp
    xwin = signal.windows.hann(ntp).reshape(1, -1)
    #plt.plot(xwin)
    sig = sig * xwin
    #plt.plot(sig)

    # Compute FFT
    F = scipy.fft.fft(sig)
    xfreq = np.arange(nfft/2) * (sfreq/nfft)
    #xf = fftfreq(ntp, 1/sfreq)[:ntp//2]
    #plt.plot(xfreq, np.abs(F[0,:ntp//2]))

    # Compute power, amplitude, and phase spectra
    AS = 2/xwin.sum() * np.sqrt(F*np.conjugate(F))  # amplitude spectrum
    S = F * np.conjugate(F) / xwin.sum()  # Power spectrum
    P = np.arctan(np.imag(F) / np.real(F))  # Phase spectrum

    # shorten matrix by half
    AS = AS[:, :ntp//2]
    S = S[:, :ntp//2]
    P = P[:, :ntp//2]
    F = F[:, :ntp//2]

    # Reshape Fourier matrix to its to initial shape
    F = F.reshape((nepoch, nchan, ntp//2))

    # inter-trial phase coherence
    itpc = np.squeeze(np.abs(np.mean(F/np.abs(F), axis=0)))  # Channel 68
    # evoked power
    epow = np.squeeze(np.abs(np.mean(F, axis=0)) ** 2)
    # total power
    tpow = np.squeeze(np.mean(np.abs(F) ** 2, axis=0))

    return itpc, xfreq, epow, tpow, F


# Unity tests
if __name__ == '__main__':
    
    
    # mk_epochs
    subj_id = 13
    path = utils.set_dirs(subj_id=subj_id, analysis_dir='')
    meg_dir = path['subj_meg_dir']

    # Load after-ICA MEG data
    meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'after_ica_meg.fif'))
    #epoch = mk_epochs(meg.copy(), mod_freq=1., tmin=-0.3, baseline=(-0.3, 0), 
     #                  annot_pattern='e/1.0/', new_event_value=101)
    #evoked = epoch.average()
    

    
    