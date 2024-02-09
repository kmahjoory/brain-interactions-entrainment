import mne
import os
import matplotlib
import matplotlib.pyplot as plt


""" 
Load Freesurfer and set the environment variables before running this script:
export SUBJECTS_DIR=$HOME/k1_analyses/prj_neuroflex/neuroflex_analysis/data/fs_subjects_dir
export FREESURFER_HOME=$HOME/freesurfer   
source $FREESURFER_HOME/SetUpFreeSurfer.sh
then run:
python
"""


datasets_path = 'datasets'
subj_id = 28


subj_name = f'subj_{subj_id}'
meg_dir = os.path.join(datasets_path, 'data', subj_name, 'meg')
mri_dir = os.path.join(datasets_path, 'data', subj_name, 'mri')
fs_subjs_dir = mri_dir = os.path.join(datasets_path, 'data/fs_subjects_dir')
plots_dir = os.path.join(datasets_path, 'plots', subj_name)

### Make source surface and write it
src = mne.setup_source_space(subject=subj_name, spacing='oct6', surface='white', 
                             subjects_dir=fs_subjs_dir, add_dist=True)

src_path = os.path.join(fs_subjs_dir, f'{subj_name}/source')
os.makedirs(src_path, exist_ok=True)
mne.write_source_spaces(fname=os.path.join(src_path, 'file-src.fif'), src=src, overwrite=True)

# Read the saved source space surface
src = mne.read_source_spaces(os.path.join(src_path, 'file-src.fif'))


### Make brain_surface, inner_skull_surface, outer_skull_surface, outer_skin_surface
mne.bem.make_watershed_bem(subject=subj_name, subjects_dir=fs_subjs_dir, overwrite=True)

# visualize bem surfaces
fig = mne.viz.plot_bem(subject=subj_name, subjects_dir=fs_subjs_dir, orientation='coronal',
                 slices=None, brain_surfaces=None, src=src, show=True, show_indices=True, 
                 mri='T1.mgz', show_orientation=True)
os.makedirs(os.path.join(plots_dir, 'fm'), exist_ok=True)
fig.savefig(os.path.join(plots_dir, 'fm', 'bem_surfs.jpg'))


### Make BEM model
bem_surfaces = mne.make_bem_model(subj_name, ico=4, conductivity=[0.3], subjects_dir=fs_subjs_dir)
bem_solution = mne.make_bem_solution(bem_surfaces, verbose=None)

if 1:
    plot_bem_kwargs = dict(
        subject=subj_name, subjects_dir=fs_subjs_dir,
        brain_surfaces='white', orientation='axial',
        slices=[50, 100, 120, 150, 200])
    fig = mne.viz.plot_bem(**plot_bem_kwargs)
    os.makedirs(os.path.join(plots_dir, 'fm'), exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'fm', 'bem_surfs_2.jpg'))

# write bem solution
mne.write_bem_solution(os.path.join(mri_dir, 'bem_sol.h5'), bem_solution, overwrite=True, verbose=None)


