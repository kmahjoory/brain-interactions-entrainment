
#! /usr/bin/bash
path='pwd'
SUBJECTS_DIR=../../datasets/data/fs_subjects_dir
Subject="subj_28"

### Set Freesurfer
export FREESURFER_HOME=/Users/keyvan.mahjoory/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

### mapping BN_atlas cortex to subjects
mris_ca_label -l $SUBJECTS_DIR/$Subject/label/lh.cortex.label $Subject lh $SUBJECTS_DIR/$Subject/surf/lh.sphere.reg $SUBJECTS_DIR/lh.BN_Atlas.gcs $SUBJECTS_DIR/$Subject/label/lh.BN_Atlas.annot
mris_ca_label -l $SUBJECTS_DIR/$Subject/label/rh.cortex.label $Subject rh $SUBJECTS_DIR/$Subject/surf/rh.sphere.reg $SUBJECTS_DIR/rh.BN_Atlas.gcs $SUBJECTS_DIR/$Subject/label/rh.BN_Atlas.annot

### check the result in Freeview
freeview -f $SUBJECTS_DIR/$Subject/surf/lh.pial:annot=$SUBJECTS_DIR/$Subject/label/lh.BN_Atlas.annot
freeview -f $SUBJECTS_DIR/$Subject/surf/rh.pial:annot=$SUBJECTS_DIR/$Subject/label/rh.BN_Atlas.annot

### Parcellation Stats
mris_anatomical_stats -mgz -cortex $SUBJECTS_DIR/$Subject/label/lh.cortex.label -f $SUBJECTS_DIR/$Subject/stats/lh.BN_Atla.stats -b -a $SUBJECTS_DIR/$Subject/label/lh.BN_Atlas.annot -c $SUBJECTS_DIR/BN_Atlas_210_LUT.txt $Subject lh white
aparcstats2table -s $Subject --hemi lh --parc BN_Atlas --meas thickness --tablefile ./$Subject/lh.thickness.txt

mris_anatomical_stats -mgz -cortex $SUBJECTS_DIR/$Subject/label/rh.cortex.label -f $SUBJECTS_DIR/$Subject/stats/rh.BN_Atlas.stats -b -a $SUBJECTS_DIR/$Subject/label/rh.BN_Atlas.annot -c $SUBJECTS_DIR/BN_Atlas_210_LUT.txt $Subject rh white
aparcstats2table -s $Subject --hemi rh --parc BN_Atlas --meas thickness --tablefile ./$Subject/rh.thickness.txt