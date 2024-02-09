import mne
import os
import matplotlib
import matplotlib.pyplot as plt

mne.viz.set_3d_options(antialias=False)

from nf_tools import fm, utils
from nf_tools import preprocessing as prep
#matplotlib.use("Qt5Agg")
#os.environ['ETS_TOOLKIT'] = 'qt4'
#os.environ['QT_API'] = 'pyqt5'
#matplotlib.use('TkAgg')


# Set the analysis directory where all plots will be saved
analysis_dir = '/mnt/beegfs/workspace/2021-0292-NeuroFlex/prj_neuroflex/k1_analyses'


subj_id = 18 #'subj_2_JMI22'
subj_name = f'subj_{subj_id}'
meg_dir = os.path.join('datasets/data', subj_name, 'meg')
mri_dir = os.path.join('datasets/data', subj_name, 'mri')
fs_subjs_dir = os.path.join('datasets/data/fs_subjects_dir')
plots_dir = os.path.join('datasets/plots', subj_name)



# Read source files
fpath_src = os.path.join(fs_subjs_dir, subj_name, 'source/file-src.fif')
src = mne.read_source_spaces(fpath_src)

# Read BEM Solution
bem_solution = mne.read_bem_solution(os.path.join(mri_dir, 'bem_sol.h5'))

# Read MEG info
meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'block_1_meg.fif'))
info = meg.info

# Load the transformation matrix
trans = mne.read_trans(os.path.join(mri_dir, 'trans' , 'meg_mri-trans.fif'))


# Read Forward solution
#fwd = mne.read_forward_solution(os.path.join(mri_dir, 'file-fwd.fif'))




# Plot to check
plot_kwargs = dict(subject=subj_name, subjects_dir=fs_subjs_dir,
                   surfaces=["head", 'white'], dig=True, eeg=[], src=src,
                   meg='sensors', show_axes=True,
                   coord_frame='meg')
fig1 = mne.viz.plot_alignment(info, trans=trans, **plot_kwargs)
fig1.background_color = (1, 1, 1)
mne.viz.set_3d_view(fig1, 180, 90, distance=0.6, focalpoint=(0., 0., 0.))
img1 = fig1.plotter.image


plot_kwargs = dict(subject=subj_name, subjects_dir=fs_subjs_dir,
                   surfaces=["head", 'white'], dig=True, eeg=[], src=src,
                   meg='sensors', show_axes=True,
                   coord_frame='meg')
fig2 = mne.viz.plot_alignment(info, trans=trans, **plot_kwargs)
fig2.background_color = (1, 1, 1)
mne.viz.set_3d_view(fig2, 270, 90, distance=0.6, focalpoint=(0., 0., 0.))
img2 = fig2.plotter.image

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(img1)
ax[1].imshow(img2)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].tick_params(axis='both', which='both', length=0)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].tick_params(axis='both', which='both', length=0)
os.makedirs(os.path.join(plots_dir , 'fm'), exist_ok=True)
fig.savefig(os.path.join(plots_dir , 'fm', 'fwd_sens_surfs.jpg'))



# Plot Helmet
plot_kwargs = dict(subject=subj_name, subjects_dir=fs_subjs_dir,
                   surfaces=["head", 'white'], dig=True, eeg=[], src=src,
                   meg=['sensors', 'helmet'], show_axes=True,
                   coord_frame='meg')
fig1 = mne.viz.plot_alignment(info, trans=trans, **plot_kwargs)
fig1.background_color = (1, 1, 1)
mne.viz.set_3d_view(fig1, 90, 90, distance=0.6, focalpoint=(0., 0., 0.))
img1 = fig1.plotter.image


plot_kwargs = dict(subject=subj_name, subjects_dir=fs_subjs_dir,
                   surfaces=["head", 'white'], dig=True, eeg=[], src=src,
                   meg=['sensors', 'helmet'], show_axes=True,
                   coord_frame='meg')
fig2 = mne.viz.plot_alignment(info, trans=trans, **plot_kwargs)
fig2.background_color = (1, 1, 1)
mne.viz.set_3d_view(fig2, 180, 90, distance=0.6, focalpoint=(0., 0., 0.))
img2 = fig2.plotter.image

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(img1)
ax[1].imshow(img2)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].tick_params(axis='both', which='both', length=0)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].tick_params(axis='both', which='both', length=0)
os.makedirs(os.path.join(plots_dir , 'fm'), exist_ok=True)
fig.savefig(os.path.join(plots_dir , 'fm', 'fwd_sens_surfs_helmet.jpg'))


# Plot dipoles on the cortex
if 0:
    fig = mne.viz.plot_alignment(info, trans=trans, subject=subj_name, dig=False,
                           meg=['sensors'], subjects_dir=fs_subjs_dir,
                           surfaces='white', interaction='trackball')
    mne.viz.set_3d_view(fig, 180, 90, distance=0.2, focalpoint=(0., 0., 0.))
    
    
    fig = mne.viz.plot_alignment(info, trans=trans, subject=subj_name, dig=False,
                           meg=['sensors'], subjects_dir=fs_subjs_dir,
                           surfaces='white', fwd=fwd, interaction='trackball')
    mne.viz.set_3d_view(fig, 0, 90, distance=0.2, focalpoint=(0., 0., 0.))


# https://www.fieldtriptoolbox.org/faq/how_can_i_convert_an_anatomical_mri_from_dicom_into_ctf_format/


# Source alignment and coordinate frames
# https://mne.tools/dev/auto_tutorials/forward/20_source_alignment.html

#mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)


#t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
#t1 = nibabel.load(t1_fname)
#t1.orthoview()