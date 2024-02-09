import mne
import os
import matplotlib
import matplotlib.pyplot as plt

mne.viz.set_3d_options(antialias=False)
from nf_tools import fm, utils
from nf_tools import preprocessing as prep
"""
# In Spyder to do visualization, run %matplotlib qt alone (No together with other lines)
# If it doesn't work, run it again before the visualization
"""
#os.environ['ETS_TOOLKIT'] = 'qt4'
#os.environ['QT_API'] = 'pyqt5'
#matplotlib.use('TkAgg')
#matplotlib.use("Qt5Agg")

# Set the analysis directory where all plots will be saved
analysis_dir = '/mnt/beegfs/workspace/2021-0292-NeuroFlex/prj_neuroflex/k1_analyses'

""" 
export SUBJECTS_DIR=$HOME/k1_analyses/prj_neuroflex/neuroflex_analysis/data/fs_subjects_dir
export FREESURFER_HOME=$HOME/freesurfer   
source $FREESURFER_HOME/SetUpFreeSurfer.sh
"""


subj_id = 2 #'subj_2_JMI22'
subj_name = f'subj_{subj_id}'
meg_dir = os.path.join('../data', subj_name, 'meg')
mri_dir = os.path.join('../data', subj_name, 'mri')
fs_subjs_dir = os.path.join('../data/fs_subjects_dir')
subj_output_dir = os.path.join('../outputs', subj_name)


fs_subjs_dir = '/Users/keyvan.mahjoory/k1_analyses/prj_neuroflex/neuroflex_analysis/data/fs_subjects_dir'

# Make source surface and write it
########################################################################################################################
src = mne.setup_source_space(subject=subj_name, spacing='oct6', surface='white', 
                             subjects_dir=fs_subjs_dir, add_dist=True)

src_path = os.path.join(fs_subjs_dir, f'{subj_name}/source')
os.makedirs(src_path, exist_ok=True)
mne.write_source_spaces(fname=os.path.join(src_path, 'file-src.fif'), src=src, overwrite=True)


src = mne.read_source_spaces(os.path.join(src_path, 'file-src.fif'))

## Make brain_surface, inner_skull_surface, outer_skull_surface, outer_skin_surface
########################################################################################################################
mne.bem.make_watershed_bem(subject=subj_name, subjects_dir=fs_subjs_dir, overwrite=True)

# visualize bem surfaces
fig = mne.viz.plot_bem(subject=subj_name, subjects_dir=fs_subjs_dir, orientation='coronal',
                 slices=None, brain_surfaces=None, src=src, show=True, show_indices=True, 
                 mri='T1.mgz', show_orientation=True)
os.makedirs(os.path.join(subj_output_dir, 'fm'), exist_ok=True)
fig.savefig(os.path.join(subj_output_dir, 'fm', 'bem_surfs.jpg'))

## Make BEM model
########################################################################################################################
bem_surfaces = mne.make_bem_model(subj_name, ico=4, conductivity=[0.3], subjects_dir=fs_subjs_dir)
bem_solution = mne.make_bem_solution(bem_surfaces, verbose=None)


if 0:


    # mne.write_bem_surfaces(fname, surfs, overwrite=False, verbose=None)
    # mne.write_head_bem(fname, rr, tris, on_defects='raise', overwrite=False, verbose=None)

    plot_bem_kwargs = dict(
        subject=subj_name, subjects_dir=fs_subjs_dir,
        brain_surfaces='white', orientation='axial',
        slices=[50, 100, 120, 150, 200])
    fig = mne.viz.plot_bem(**plot_bem_kwargs)
    os.makedirs(os.path.join(subj_output_dir, 'fm'), exist_ok=True)
    fig.savefig(os.path.join(subj_output_dir, 'fm', 'bem_surfs_2.jpg'))


# write bem solution
#mne.write_bem_solution(os.path.join(mri_dir, 'bem_surfs.h5'), bem_surfaces, overwrite=True, verbose=None)
mne.write_bem_solution(os.path.join(mri_dir, 'bem_sol.h5'), bem_solution, overwrite=True, verbose=None)

# Read bem solution
# mne.read_bem_solution(fname, verbose=None)


## Calculate Forward solution
########################################################################################################################
# Read info

# Read transformation matrix (If str, the path to the head<->MRI transform *-trans.fif file produced during coregistration
#Can also be 'fsaverage' to use the built-in fsaverage transformation. )




# Read source files
fpath_src = os.path.join(fs_subjs_dir, subj_name, 'source/file-src.fif')
src = mne.read_source_spaces(fpath_src)

# Read BEM Solution
bem_solution = mne.read_bem_solution(os.path.join(mri_dir, 'bem_sol.h5'))

# Read MEG info
meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'block_1_raw.fif'))
info = meg.info

# Load the transformation matrix

trans = mne.read_trans(os.path.join(mri_dir, 'trans' , 'meg_mri-trans.fif'))

# solve the Forward problem
fwd = mne.make_forward_solution(info=info, trans=trans, src=src, bem=bem_solution, meg=True, eeg=False, mindist=0.0,
                                ignore_ref=False, n_jobs=1)

mne.write_forward_solution(os.path.join(mri_dir, 'file-fwd.fif'), fwd)
#read_forward_solution

# Plot field distribution

plot_kwargs = dict(subject=subj_name, subjects_dir=fs_subjs_dir,
                   surfaces=["brain", 'head'], dig=True, bem=bem_solution, src=src,
                   meg=['helmet', 'sensors'], show_axes=True,
                   coord_frame='meg')
fig = mne.viz.plot_alignment(info, trans=trans, **plot_kwargs)
mne.viz.set_3d_view(fig, 180, 90, distance=0.6, focalpoint=(0., 0., 0.))
_, im = mne.viz.snapshot_brain_montage(fig, info, hide_sensors=True)
plt.imshow(im)
plt.show()

fig.savefig(os.path.join(subj_output_dir, 'fm', 'coregistration_1.jpg'))

# Plot to check
fig, az = plt.subplots(1, 1, figsize=(10, 10))
plot_kwargs = dict(subject=subject, subjects_dir=subjects_dir,
                   surfaces="head", dig=True, eeg=[],
                   meg='sensors', show_axes=True,
                   coord_frame='meg')
fig = mne.viz.plot_alignment(info, trans=trans, **plot_kwargs)
mne.viz.set_3d_view(fig, 180, 90, distance=0.6, focalpoint=(0., 0., 0.))


view_kwargs = dict(azimuth=45, elevation=90, distance=0.6,
                   focalpoint=(0., 0., 0.))


fig = mne.viz.plot_alignment(info, trans=trans, subject=subject, dig=False,
                       meg=['sensors'], subjects_dir=subjects_dir,
                       surfaces='head', fwd=fwd, interaction='trackball')
mne.viz.set_3d_view(fig, 180, 90, distance=0.6, focalpoint=(0., 0., 0.))



mne.viz.set_3d_view(fig, 90, 90, distance=0.6, focalpoint=(0., 0., 0.))








# To do:
# try to visualize MEG sensors toghether with bem surfaces using mne.viz.plot_alignment
# try to find the trans automatcially and plot


## Relevant MNE pages
# https://mne.tools/stable/overview/implementation.html#bem-model
# https://www.fieldtriptoolbox.org/faq/how_can_i_convert_an_anatomical_mri_from_dicom_into_ctf_format/


# Source alignment and coordinate frames
# https://mne.tools/dev/auto_tutorials/forward/20_source_alignment.html

#mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)


#t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
#t1 = nibabel.load(t1_fname)
#t1.orthoview()