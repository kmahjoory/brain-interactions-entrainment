import os
import datetime
import shutil
import numpy as np
import subprocess
import glob
import mne
import pandas as pd
from nf_tools import preprocessing as prep


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
            trigger_vals = prep.find_events_frequency(trig_sig)
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

def status(subj_ids=range(1, 29), subjs_dir=None):
    """
    This function will print the status of analysis. e.g. what files are ready or which analysis done!

    args:
        subjs_dir: Default is '/hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/data/'

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



#if __name__ == '__main__':
    #  Add unit test