
subj_id=$1

export dst=~/k1_analyses/prj_neuroflex/neuroflex_analysis/datasets/data/subj_${subj_id}/meg/
export src=keyvan.mahjoory@hpc-login:/hpc/workspace/2021-0292-NeuroFlex/prj_neuroflex/neuroflex_analysis/datasets/data/subj_${subj_id}/meg/block_*_meg.fif

mkdir -p $dst

rsync -aux $src $dst