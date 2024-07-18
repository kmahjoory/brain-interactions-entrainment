
# Brain Interactions during Entrainment to Frequency Modulated Sounds

This is a Magnetoencephalography (MEG) study exploring the interactions between various brain regions when entrained to frequency-modulated sounds.

## Objectives

- To map the brain's response patterns to frequency-modulated sounds.
- To identify specific brain regions involved in the processing and entrainment to these sounds.
- To understand the interaction and communication between different brain areas during auditory entrainment.

## Data

## Behavioral Paradigm

## Stimuli

## Methods


## MEG Proprecessing Pipeline
```python
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

```

## MRI Proprecessing Pipeline



### Alighning MEG and MRI data
```shell
export SUBJECTS_DIR=~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/fs_subjects_dir 
mne coreg
```

### Forward model 

```python
# set up the Freesurfer home and subject directory
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

```

### SOURCE RECONSTRUCTION

- Apply source recon separately on silence and entrainment, and each of fms
- Implement both LCMV and MNE
- WHICH SSD
- Sensor space ITC and Source space ITC
