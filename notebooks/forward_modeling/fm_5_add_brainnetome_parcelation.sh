

# Read the documentation. The file BN_Atlas_freesurfer_Usage.pdf 

Subject=$1


# Initialize FreeSurfer
export FREESURFER_HOME=$HOME/freesurfer   
export SUBJECTS_DIR=$HOME/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/fs_subjects_dir
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Unzip BN_atlas_freesurfer.zip and copy the four files, lh.BN_Atlas.gcs , rh.BN_Atlas.gcs , BN_Atlas_subcortex.gca , and BN_Atlas_246_LUT.txt , to the Freesurfer SUBJECTS_DIR
# folder.


### mapping BN_atlas cortex to subjects

mris_ca_label -l $SUBJECTS_DIR/$Subject/label/lh.cortex.label $Subject lh $SUBJECTS_DIR/$Subject/surf/lh.sphere.reg $SUBJECTS_DIR/lh.BN_Atlas.gcs $SUBJECTS_DIR/$Subject/label/lh.BN_Atlas.annot

mris_ca_label -l $SUBJECTS_DIR/$Subject/label/rh.cortex.label $Subject rh $SUBJECTS_DIR/$Subject/surf/rh.sphere.reg $SUBJECTS_DIR/rh.BN_Atlas.gcs $SUBJECTS_DIR/$Subject/label/rh.BN_Atlas.annot

