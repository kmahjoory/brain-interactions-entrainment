import os
import datetime
import shutil
import numpy as np
import subprocess
import glob
import mne
import pandas as pd
from .preprocessing import *



def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')


def mk_subj_dir(subj_id, subjs_dir=""):
    """This function creates the subject directory and "meg", "mri", "smt_meg", and "behavior" directories."""
    if not subjs_dir:
        subjs_dir = "/hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/data/"
    subj_ids = get_subj_ids(subj_id, subjs_dir='')
    subj_name = "_".join(subj_ids)
    subj_shortname = subj_ids[0]
    for jdir in ['meg', 'mri', 'meg_smt', 'behavior', 'other_meg', 'trigger_mismatch_meg']:
        os.makedirs(os.path.join(subjs_dir, subj_shortname, jdir), exist_ok=True)
    with open(os.path.join(subjs_dir, subj_shortname, 'subject_ids.txt'), 'w') as f:
        f.write(subj_name)
    return subj_ids, subj_shortname
        
        
def set_dirs(subj_id, analysis_dir=''):
    """This function sets all directories for a subject and returns all as a dictionary"""
    
    subj_name = f'subj_{subj_id}'

    subjs_dir = '/hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/data/'
    subj_dir = os.path.join(subjs_dir, subj_name)
    subj_meg_dir = os.path.join(subj_dir, 'meg')
    
    subj_mri_dir = os.path.join(subj_dir, 'mri')
    subj_trans_dir = os.path.join(subj_mri_dir, 'trans')
    fs_subjs_dir = '/hpc/workspace/2021-0292-NeuroFlex/packages/freesurfer/subjects'
    subj_fs_dir = os.path.join(fs_subjs_dir, subj_name)
    
    all_dirs = {'subj_dir': subj_dir, 'subj_meg_dir': subj_meg_dir,
        'subj_mri_dir': subj_mri_dir, 'subj_trans_dir': subj_trans_dir, 
        'fs_subjs_dir': fs_subjs_dir, 'subj_fs_dir': subj_fs_dir,
        }
        
    if analysis_dir:
        subj_analysis_dir = os.path.join(analysis_dir, subj_name)
        prep_dir = os.path.join(subj_analysis_dir, 'preprocessing')
        all_dirs['analysis_dir'] = analysis_dir
        all_dirs['subj_analysis_dir'] = subj_analysis_dir
        all_dirs['prep_dir'] = prep_dir

            
    for vals in all_dirs.values():
        os.makedirs(vals, exist_ok=True)
    all_dirs['subj_id'] = subj_name
           
    return all_dirs

        

def mv_file(src=None, dest=None):
    """ This function moves a file to an existing directory.
    It can also be used for renaming files in a directory"""

    if not os.path.isfile(src):
        print("Specified source file is not available!")
    else:
        if os.path.isfile(dest):
            d = datetime.datetime.now()
            dtxt = f"{d.year}.{d.month}.{d.day}_{d.hour}:{d.minute}"
            shutil.move(dest, f"{dest[:-4]}_{dtxt}.csv")
        shutil.move(src, dest)
    print("List of files in the destination directory:")
    print(os.listdir(os.path.dirname(dest)))


def get_subj_ids(subj_id, subjs_dir=''):
    """This function creates the subject directory and "meg", "mri", "smt_meg", and "behavior" directories."""
    if not subjs_dir:
        subjs_dir = "/hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/data/"
    # Read the participant IDs file
    # Get all IDs (subject_id, bic_id, morla_id, and morla neurflex id) for a participant
    part_ids_file = "/hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/data/partcipant_ids/participant_ids.csv"
    part_ids = pd.read_csv(part_ids_file, header=0)  # The first row in our csv file is the header
    # Make sure that IDs or in integer/string format
    part_ids["subj_id"] = part_ids["subj_id"].astype(int)  # subject IDs should be integer values
    part_ids[["bic_id", "bic_id2", "morla_id", "morla_nfid"]] = part_ids[["bic_id", "bic_id2", "morla_id", "morla_nfid"]].astype(str)
    # find the participant of interest
    indx_subj = np.where(part_ids.subj_id == subj_id)[0].item()
    return f"subj_{subj_id}", part_ids.bic_id[indx_subj].strip(), part_ids.bic_id2[indx_subj].strip(), part_ids.morla_id[indx_subj].strip(), part_ids.morla_nfid[indx_subj].strip()


def cat_cp_meg_blocks(src, dest, name_pattern):
    """This function is to categorize raw meg files to one of the following categories: 1) nf meg 2) smt, 3) resting state
    4) other useless. Saves the output in a CSV file. and Copies the raw block files to corresponding category."""

    fnames = glob.glob(os.path.join(src, name_pattern))
    if not fnames:
        print('No relevant file found in  /mnt/prjekte directory')
    fnames.sort(key=os.path.getmtime)
    n_files = len(fnames)
    files_size = []
    meg_nf_blocks_origname, meg_nf_blocks_name = [], []
    meg_nf_blocks_dur, meg_nf_blocks_events = [], []
    meg_nf_blocks_acqtime = []
    meg_smt_blocks_origname, meg_smt_blocks_name = [], []
    meg_smt_blocks_dur, meg_smt_blocks_events = [], []
    meg_smt_blocks_acqtime = []

    acq_times, acq_durs, acq_types = [], [], []
    cnt_nf, cnt_smt, cnt_other = 0, 0, 0
    for jfile in range(n_files):
        fbasename = os.path.basename(fnames[jfile])
        fsize = du(fnames[jfile])
        if fsize[-1] == 'G':
            files_size.append(fsize)
            raw = mne.io.read_raw_ctf(directory=fnames[jfile])
            trig_sig = raw['UPPT001'][0]
            trigger_vals = find_events_frequency(trig_sig)
            is_smt = False
            is_nf = False
            if float('200.0') in trigger_vals.keys():
                if trigger_vals[200] > 50:
                    is_smt = True

            if 56 in trigger_vals.values():
                if 28 in trigger_vals.values():
                    is_nf = True
            dur = np.round(raw.n_times/(1200*60), 1)
            if (dur > 10) and is_nf:
                cnt_nf += 1
                acq_type_ = f'nf_b{cnt_nf}'
                acq_types.append(acq_type_)
                meg_nf_blocks_origname.append(fbasename)
                meg_nf_blocks_name.append(f'nf_b{cnt_nf}')
                meg_nf_blocks_dur.append(dur)
                meg_nf_blocks_events.append(str(trigger_vals))
                meg_nf_blocks_acqtime.append(raw.info['meas_date'])
                # Copy NF block files to subject directory
                shutil.copytree(src=fnames[jfile], dst=os.path.join(dest, 'meg', fbasename))
            elif (dur > 1.4) and (dur < 5) and is_smt:
                cnt_smt += 1
                acq_type_ = f'smt_{cnt_smt}'
                acq_types.append(acq_type_)
                meg_smt_blocks_origname.append(fbasename)
                meg_smt_blocks_name.append(f'smt_{cnt_nf}')
                meg_smt_blocks_dur.append(dur)
                meg_smt_blocks_events.append(str(trigger_vals))
                meg_smt_blocks_acqtime.append(raw.info['meas_date'])
                # Copy SMT block files to subject directory
                shutil.copytree(src=fnames[jfile], dst=os.path.join(dest, 'meg_smt', fbasename))
            else:
                cnt_other += 1
                acq_types.append(f'other_b{cnt_other}')
                shutil.copytree(src=fnames[jfile], dst=os.path.join(dest, 'trigger_mismatch_meg', fbasename))
        else:
            shutil.copytree(src=fnames[jfile], dst=os.path.join(dest, 'other_meg', fbasename))
    # Categorize Neurflex MEG blocks files
    meg_nf_blocks_info = pd.DataFrame(list(zip(meg_nf_blocks_origname, meg_nf_blocks_name, meg_nf_blocks_dur,
                                              meg_nf_blocks_events, meg_nf_blocks_acqtime)),
                                      columns=['origname', 'name', 'dur', 'events', 'acqtime'])
    meg_nf_blocks_info.to_csv(os.path.join(dest, 'meg', 'nf_blocks_info.csv'))
    # Categorize SMT MEG blocks files
    meg_smt_blocks_info = pd.DataFrame(list(zip(meg_smt_blocks_origname, meg_smt_blocks_name, meg_smt_blocks_dur,
                                                meg_smt_blocks_events, meg_smt_blocks_acqtime)),
                                       columns=['origname', 'name', 'dur', 'events', 'acqtime'])
    meg_smt_blocks_info.to_csv(os.path.join(dest, 'meg_smt', 'smt_blocks_info.csv'))
    print("##################################################")
    print("list of copied/renamed files:")
    os.system(f"ls -lh {dest}*/")


def status_subj(subj_id):
    """
    
    """
    subj_name = f"subj_{subj_id}"
    subjs_path = "datasets/data/"
    subj_path = os.path.join(subjs_path, subj_name)

    meg_raw_ds, meg_rawfif, mri_raw, meg_blocks_fif, meg_concat_fif, meg_after_ica_fif = ([] for i in range(6))
    meg_raw_other_ds, meg_raw_smt, meg_raw_triggermismatch = [], [], []
    #print(os.listdir(subj_path))
    meg_raw_ds.append(len(glob.glob(os.path.join(subj_path, 'meg', '*.ds'))))
    meg_raw_other_ds.append(len(glob.glob(os.path.join(subj_path, 'other_meg', '*.ds'))))
    meg_raw_smt.append(len(glob.glob(os.path.join(subj_path, 'meg_smt', '*.ds'))))
    meg_raw_triggermismatch.append(len(glob.glob(os.path.join(subj_path, 'trigger_mismatch_meg', '*.ds'))))

    meg_rawfif.append(len(glob.glob(os.path.join(subj_path, 'meg', '*raw.fif'))))
    meg_blocks_fif.append(len(glob.glob(os.path.join(subj_path, 'meg', 'block_*meg.fif'))))
    meg_concat_fif.append(len(glob.glob(os.path.join(subj_path, 'meg', 'concat_meg*fif'))))
    meg_after_ica_fif.append(len(glob.glob(os.path.join(subj_path, 'meg', 'after_ica*fif'))))
    mri_raw.append(len(glob.glob(os.path.join(subj_path, 'mri', '*/*.dcm'))))

    T = {'megRawds': meg_raw_ds, 'megRawfif': meg_rawfif, 
         'megBlocksfif': meg_blocks_fif, 'megConcat': meg_concat_fif, 'megAfterIca': meg_after_ica_fif,
         'meg_smt': meg_raw_smt, 'megRawOther': meg_raw_other_ds,'megRawTrigger':meg_raw_triggermismatch, 
         'mri_raw': mri_raw}
    return pd.DataFrame(T)


def status(subj_ids, subjs_dir):
    """
    This function will print the status of analysis. e.g. what files are ready or which analysis done!

    args:
        subj_id
        subjs_dir

    Returns:
        A pandas data frame containing the information about the analysis
    """
    if not subjs_dir:
        subjs_dir = '/hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/data/'

    if not isinstance(subj_ids, list) | isinstance(subj_ids, range):
        subj_ids = [subj_ids]
    ID = []
    shortname, bic_id_mri, bic_id, morla_id, morla_fid = [], [], [], [], []
    meg_rawds, meg_rawfif, mri_raw = [], [], []
    meg_raw_other_ds, meg_raw_smt, meg_raw_triggermismatch = [], [], []
    for subject in subj_ids:
        _shortname, _bic_id_mri, _bic_id, _morla_id, _morla_fid = get_subj_ids(subject, subjs_dir)
        ID.append(subject)
        shortname.append(_shortname)
        bic_id_mri.append(_bic_id_mri)
        bic_id.append(_bic_id)
        morla_id.append(_morla_id)
        morla_fid.append(_morla_fid)

        meg_rawds.append(len(glob.glob(os.path.join(subjs_dir, _shortname, 'meg', '*.ds'))))
        meg_raw_other_ds.append(len(glob.glob(os.path.join(subjs_dir, _shortname, 'other_meg', '*.ds'))))
        meg_raw_smt.append(len(glob.glob(os.path.join(subjs_dir, _shortname, 'meg_smt', '*.ds'))))
        meg_raw_triggermismatch.append(len(glob.glob(os.path.join(subjs_dir, _shortname, 'trigger_mismatch_meg', '*.ds'))))

        meg_rawfif.append(len(glob.glob(os.path.join(subjs_dir, _shortname, 'meg', '*raw.fif'))))
        mri_raw.append(len(glob.glob(os.path.join(subjs_dir, _shortname, 'mri', '*/*.dcm'))))

    T = {'ID': ID, 'shortName': shortname, 'megRawds': meg_rawds, 'megRawfif': meg_rawfif, 'megRawOther': meg_raw_other_ds,
         'meg_smt': meg_raw_smt, 'megRawTrigger':meg_raw_triggermismatch, 'mri_raw': mri_raw}
    TT = pd.DataFrame(T)
    print(TT)

    return TT


def cp_raw_mri(subj_ids, src=None, dst=None):
    """
    This function looks up a RAW MRI based on participant ID and copies it to subject folder
    args:
        subj_ids: a python range e.g. range(1, 2) or range(1, 5)
    Return:

    """
    if not src:
        src = '/mnt/projekte/2021-0292-NeuroFlex/rawdata/'

    if not dst:
        dst = '/hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/data/'

    for ID in subj_ids:

        _shortname, _bic_id_mri, _bic_id, _morla_id, _morla_fid = get_subj_ids(ID, dst)
        name_pattern = f"*{_bic_id}_0292"
        if not glob.glob(os.path.join(dst, _shortname, 'mri', name_pattern)):
            fname_0 = glob.glob(os.path.join(src, name_pattern))
            if len(fname_0) == 1:
                readname = fname_0[0]
                shutil.copytree(src=readname, dst=os.path.join(dst, _shortname, 'mri', os.path.basename(readname)))
            else:
                fname_1 = glob.glob(os.path.join('/mnt/projekte/2021-0292-NeuroFlex/MRI_Re-use/', name_pattern))
                if len(fname_1) == 1:
                    readname = fname_1[0]
                    shutil.copytree(src=readname, dst=os.path.join(dst, _shortname, 'mri', os.path.basename(readname)))
                else:
                    fname_2 = glob.glob(os.path.join('/mnt/projekte/2021-0292-NeuroFlex/MRI_new/', name_pattern))
                    if len(fname_2) == 1:
                        readname = fname_2[0]
                        shutil.copytree(src=readname,
                                        dst=os.path.join(dst, _shortname, 'mri', os.path.basename(readname)))
                    else:
                        print(f"MRI file for subj_{ID} not found!")

        else:
            print(f"   ***  MRI file for {_shortname} is already copied!  ***")


    
def annotate_blocks(subjs_dir, subj_id, stim_channel = 'UPPT001', write=True):
    """
    This function annotates the MEG data for each block using the behavioral data. 
    It compares the modulating frequencies, correct responses, and the time of recording between the MEG trigger and behavioral data.

    Args:
        subjs_dir (str): path to the directory containing the subjects data. This directory should contain a folder named "behav_data"
          containing the behavioral data for all subjects. And a folder for each subject containing the MEG data.
        subj_id (int): subject id

    Returns:
        None
    """
    subj_name = f'subj_{subj_id}'
    
    meg_dir = os.path.join(subjs_dir, subj_name, 'meg')
    behav_dir = os.path.join(subjs_dir, "behav_data", subj_name)
    # load blocks info
    meg_blocks_info = pd.read_csv(os.path.join(meg_dir, 'nf_blocks_info.csv'), header=0)
    n_meg_blocks = meg_blocks_info.shape[0]
    for block in range(1, n_meg_blocks+1):  # to include the last block as well
        meg_block_name = meg_blocks_info.origname[block-1]
        raw = mne.io.read_raw_ctf(os.path.join(meg_dir, meg_block_name))
        events = mne.find_events(raw, stim_channel=stim_channel)

        event_initial = np.array([0, 0, 1]).reshape((1, 3))  # to specify the time period before the first trigger!
        events_new = np.concatenate((event_initial, events), axis=0)
        events_onset_in_sec = events_new[:, 0] / raw.info['sfreq']  # Not included the last event

        event_ending = np.array([raw.last_samp, 0, 2]).reshape((1, 3))  # to specify the time period after the last trigger!
        events_new_appended = np.concatenate((events_new, event_ending), axis=0)
        events_dur_in_sec = (events_new_appended[1:, 0] - events_new_appended[:-1, 0]) / raw.info[
            'sfreq']  # The last events halps find the duration of events
        events_correct_response = events[2::4, 2]
        annots_correct_response = ['faster' if jev==24 else 'slower' for jev in events_correct_response]

        # obtain fm values from MEG data
        fms_trials_from_trigger_not_rounded = 8 / events_dur_in_sec[1::4]
        fms_trials_from_trigger = np.round(fms_trials_from_trigger_not_rounded * 2)/2  # To round values to 1, 1.5, 2, ...
        fms_from_trigger = np.tile(fms_trials_from_trigger.reshape((-1, 1)), (1, 4)).reshape((-1, 1)).ravel()
        labels_fm = np.insert(fms_from_trigger, 0, float('nan'))

        # obtain fm values from behav data
        behav_block_info = pd.read_csv(os.path.join(behav_dir, f'block_{block}.csv'), header=0)
        behav_correct_response = behav_block_info.correct_response.to_list()
        fms_from_behav = behav_block_info.mod_freq.to_numpy()

        assert np.all(fms_trials_from_trigger == fms_from_behav), f'Modulating frequencies do NOT match for block {block}'
        assert_correct_response = [annots_correct_response[jev]==behav_correct_response[jev] for jev in range(len(annots_correct_response
                                                                                                                  ))]
        assert np.all(np.array(assert_correct_response)), f'The event for the correct response does not match  between behavioral and MEG'
        # TO DO: Add recording time from Psychtoolbox and MEG and compare ...
        #behav_block_info.block_start_datetime[0]
        #meg_blocks_info.acqtime[0]

        # Annotate using behavioral data
        fms = behav_block_info.mod_freq.astype(str).tolist()
        phie = behav_block_info.phase_enc.astype(int).astype(str).replace('3', 'pi')
        phit = behav_block_info.phase_targ.astype(int).astype(str).replace('3', 'pi')
        resp = behav_block_info.eval_response.astype(int).astype(str)
        freq_match = behav_block_info.freq_match.astype(str)
        events_ = ["/".join([fms[k],f'e{phie[k]}', f't{phit[k]}', f"{freq_match[k][0]}",f"r{resp[k]}"]) for k in range(len(fms))]
        event_descs = [f"{j}/{events_[k]}" for k in range(len(events_)) for j in ['e', 'm', 't', 'bad_']]
        event_descs.insert(0, 'bad_start')
        # Apply annotations to block meg data
        block_annots = mne.Annotations(onset=events_onset_in_sec,
                                       duration=events_dur_in_sec,
                                       description=event_descs,
                                       orig_time=raw.info['meas_date'])
        raw = raw.set_annotations(block_annots)

        # Save the data for block
        if write:
            write_path = meg_dir
            write_name = f"block_{block}" + "_raw.fif"  # common meg files should be saved in this format
            raw.save(os.path.join(write_path, write_name), overwrite=True)
    # Print list of files in meg directory
    os.system(f"ls -lh {meg_dir}*/")






if __name__ == '__main__':
    import sys
    print(sys.path)
    #  Add unit test