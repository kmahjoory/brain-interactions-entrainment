#CONVERT DICOM to IMAGE/NIFTI


#RUN FREESURFER ON MRI IMAGE CREATEd
my_subject='subj_13'

my_NIfTI=/hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/data/subj_13/mri/GDA04_0292/1_002_T1_mprage_sag_1_0iso_20220120/GDA04_0292_20220120_001_002_T1_mprage_sag_1_0iso.img

recon-all -i $my_NIfTI -s $my_subject -all