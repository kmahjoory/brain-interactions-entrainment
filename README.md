# Neuroflex: Investigating Brain Interactions during Entrainment to Frequency Modulated Sounds

## Overview
Neuroflex is a Magnetoencephalography (MEG) study project focused on exploring the interactions between various brain regions when entrained to frequency-modulated sounds..

## Data

## Stimuli

## Methods

## Objectives
- To map the brain's response patterns to frequency-modulated sounds.
- To identify specific brain regions involved in the processing and entrainment to these sounds.
- To understand the interaction and communication between different brain areas during auditory entrainment.

## Getting Started

### check the analysis status
from nf_tools import utils  
utils.status(subj_ids=range(1, 29))

### Copy data from the server to the local machine and organize them in the right folder structure
from src import utils
utils.cat_cp_meg_blocks(src='',
                        dest='', 
                        name_pattern='')

### Annotate blocks based on Behavioral data and MEG stimulus channel
from src import utils
utils.annotate_blocks(subjs_dir='datasets/data/', subj_id=25, write=True)

### Filter and downsample the data 
from src import preprocessing as prep
prep.preprocess_blocks(subj_id=25, subjs_dir='datasets/data/', plots_path='datasets/plots/', write=True)


### Reject bad channels or epochs
wf4_bad_channels.ipynb


### Concatenate blocks
from src import preprocessing as prep
prep.concat_blocks(subj_id=25, subjs_dir='datasets/data/', plots_path='datasets/plots/')

### Run ICA analysis and save the ICA solution
from src import preprocessing as prep
prep.run_ica(subj_id=25, subjs_dir='datasets/data/', write=True, overwrite=False)

### Visually inspect the ICA components
ica_analysis.ipynb



### Useful terminal commands
rm -r subj_20/meg/block_*meg.fif
rm -r $(find subj_23/meg/ -name "block_*meg.fif" ! -name "block_1_meg.fif")
rsync -aux --include="*meg.fif" --exclude='*' --exclude='*/' keyvan.mahjoory@hpc-login://hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/neuroflex_analysis/datasets/data/subj_23/meg/ ~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/subj_23/meg/


rsync -aux --include="*" ~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/subj_28/meg/ keyvan.mahjoory@hpc-login://hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/neuroflex_analysis/datasets/data/subj_28/meg/ 


for i in {15..20}; do rsync -aux --include="*" ~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/subj_${i}/meg/ keyvan.mahjoory@hpc-login://hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/neuroflex_analysis/datasets/data/subj_${i}/meg/; done;



### SET SUBJECTS_DIR for MNE
export SUBJECTS_DIR=~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/fs_subjects_dir 
mne coreg

### Download Freesufer SUBJECTS_DIR from the server
rsync -aux --include="subj*" keyvan.mahjoory@hpc-login://hpc/workspace/2021-0292-NeuroFlex/packages/freesurfer/subjects/ ~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/fs_subjects_dir/

rsync -aux --include="subj*" ~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/fs_subjects_dir/ keyvan.mahjoory@hpc-login://hpc/workspace/2021-0292-NeuroFlex/packages/freesurfer/subjects/ 

/hpc/workspace/2021-0292-NeuroFlex/packages/freesurfer/subjects


### Forward model 

# set up the Freesurfer home and subject directory
python
from src import fm
import matplotlib.pyplot as plt

# estimate BEM solution from Freesurfer output
for i in range(14, 15):
    fm.calc_bem_solution(subj_id=i, 'datasets'); plt.close('all')




# Check BEM surfaces and coregistration
for i in range(3,14):
    fm.viz_head_model(i, 'datasets')

# Make forward model
for i in range(15, 29):
    fm.mk_fwd_model(i, 'datasets')


### Add atlas labels to the source space
- Read the documentation
cd notebooks/forward_modeling
# Note that this schell script takes subject name as an argument
./fm_5_add_brainnetome_parcelation.sh subj_16
# Or write a for loop
for i in {19..28}; do ./fm_5_add_brainnetome_parcelation.sh subj_${i}; done;
# Check the results
ls -l datasets/data/fs_subjects_dir/subj_*/label/*.BN_Atlas*

# Plot atlas parcelation
python fm_6_visualize_atlas.py


### SOURCE RECONSTRUCTION

- Apply source recon separately on silence and entrainment, and each of fms
- Implement both LCMV and MNE
- WHICH SSD
- Sensor space ITC and Source space ITC


### Make report
conda activate mne
cd reports
for i in {26..28}; do python mk_report.py --id ${i};done;




### Sync data with the server
rsync -aux --include="subj*" keyvan.mahjoory@hpc-login://hpc/workspace/2021-0292-NeuroFlex/packages/freesurfer/subjects/ ~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/fs_subjects_dir/

rsync -aux --include="subj*" ~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/fs_subjects_dir/ keyvan.mahjoory@hpc-login://hpc/workspace/2021-0292-NeuroFlex/packages/freesurfer/subjects/ 


/Users/keyvan.mahjoory/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/subj_4/mri/file-fwd.fif

for i in {20..28}; do rsync -aux ~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/subj_${i}/mri/file-fwd.fif keyvan.mahjoory@hpc-login://hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/neuroflex_analysis/datasets/data/subj_${i}/mri/; echo  subj_${i} Uploaded!; done;

rsync -aux ~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/subj_${i}/mri/file-fwd.fif keyvan.mahjoory@hpc-login://hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/neuroflex_analysis/datasets/data/subj_${i}/mri/

# MEG files
rsync -aux --include="after_ica*" ~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/subj_${i}/meg/ keyvan.mahjoory@hpc-login://hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/neuroflex_analysis/datasets/data/subj_${i}/meg/

for i in {20..28}; do echo subj_${i} Uploaded!; echo subj Uploaded!; done;


# ITC for labels

for