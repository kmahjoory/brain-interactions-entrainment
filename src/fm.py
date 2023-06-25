# -*- coding: utf-8 -*-
import mne
import os
import matplotlib
import matplotlib.pyplot as plt


def calc_bem_solution(subj_id, datasets_path):
    """
    Calculate BEM solution for a given subject.
    """

    subj_name = f'subj_{subj_id}'
    mri_dir = os.path.join(datasets_path, 'data', subj_name, 'mri')
    os.makedirs(mri_dir, exist_ok=True)
    fs_subjs_dir = os.path.join(datasets_path, 'data/fs_subjects_dir')
    plots_dir = os.path.join(datasets_path, 'plots', subj_name)

    if os.path.exists(os.path.join(mri_dir, 'bem_sol.h5')) == True:
        print("BEM solution already exists. Skipping the calculation.")
        return

    ### Make source surface and write it
    src_path = os.path.join(fs_subjs_dir, f'{subj_name}/source')
    if os.path.isfile(os.path.join(src_path, 'file-src.fif')) == False:
        print("=====>  Source space calculation:")
        src = mne.setup_source_space(subject=subj_name, spacing='oct6', surface='white', 
                                    subjects_dir=fs_subjs_dir, add_dist=True) 
        os.makedirs(src_path, exist_ok=True)
        mne.write_source_spaces(fname=os.path.join(src_path, 'file-src.fif'), src=src, overwrite=True)
    else:
        print("=====>  Source space already exists. Skipping the calculation.")

    # Read the saved source space surface
    src = mne.read_source_spaces(os.path.join(src_path, 'file-src.fif'))


    ### Make brain_surface, inner_skull_surface, outer_skull_surface, outer_skin_surface
    if os.path.isfile(os.path.join(fs_subjs_dir, subj_name, 'bem/watershed/ws.mgz')) == False:
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



def viz_head_model(subj_id, datasets_path):
    """
    """

    subj_name = f'subj_{subj_id}'
    mri_dir = os.path.join(datasets_path, 'data', subj_name, 'mri')
    meg_dir = os.path.join(datasets_path, 'data', subj_name, 'meg')
    fs_subjs_dir = os.path.join(datasets_path, 'data/fs_subjects_dir')
    plots_dir = os.path.join(datasets_path, 'plots', subj_name)

    # Read source files
    fpath_src = os.path.join(fs_subjs_dir, subj_name, 'source/file-src.fif')
    src = mne.read_source_spaces(fpath_src)

    # Read BEM Solution
    bem_solution = mne.read_bem_solution(os.path.join(mri_dir, 'bem_sol.h5'))

    # Read MEG info
    if os.path.isfile(os.path.join(meg_dir, 'block_1_meg.fif')) == True:
        meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'block_1_meg.fif'))
    else:
        meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'after_ica_meg.fif'))
    info = meg.info

    # Load the transformation matrix
    trans = mne.read_trans(os.path.join(mri_dir, 'trans' , 'meg_mri-trans.fif'))

    # Vizualize the head model after coregistration (source space, BEM surfaces, MEG sensors)
    plot_kwargs = dict(subject=subj_name, subjects_dir=fs_subjs_dir,
                    surfaces=["head", 'white'], dig=True, eeg=[], src=src,
                    meg='sensors', show_axes=True,
                    coord_frame='meg')
    fig1 = mne.viz.plot_alignment(info, trans=trans, **plot_kwargs)
    fig1.background_color = (1, 1, 1)
    mne.viz.set_3d_view(fig1, 180, 90, distance=0.6, focalpoint=(0., 0., 0.))
    img1 = fig1.plotter.image

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
    fig.savefig(os.path.join(plots_dir , 'fm', 'fm_sens_surfs.jpg'))

    # Plot Helmet
    plot_kwargs = dict(subject=subj_name, subjects_dir=fs_subjs_dir,
                    surfaces=["head", 'white'], dig=True, eeg=[], src=src,
                    meg=['sensors', 'helmet'], show_axes=True,
                    coord_frame='meg')
    fig1 = mne.viz.plot_alignment(info, trans=trans, **plot_kwargs)
    fig1.background_color = (1, 1, 1)
    mne.viz.set_3d_view(fig1, 180, 90, distance=0.6, focalpoint=(0., 0., 0.))
    img1 = fig1.plotter.image

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
    fig.savefig(os.path.join(plots_dir , 'fm', 'fm_sens_surfs_helmet.jpg'))



def mk_fwd_model(subj_id, datasets_path):
    """
    """
    subj_name = f'subj_{subj_id}'
    mri_dir = os.path.join(datasets_path, 'data', subj_name, 'mri')
    meg_dir = os.path.join(datasets_path, 'data', subj_name, 'meg')
    fs_subjs_dir = os.path.join(datasets_path, 'data/fs_subjects_dir')
    plots_dir = os.path.join(datasets_path, 'plots', subj_name)
    
    # Read source space file
    fpath_src = os.path.join(fs_subjs_dir, subj_name, 'source/file-src.fif')
    src = mne.read_source_spaces(fpath_src)
    
    # Read BEM Solution
    bem_solution = mne.read_bem_solution(os.path.join(mri_dir, 'bem_sol.h5'))

    # Read MEG info
    if os.path.isfile(os.path.join(meg_dir, 'block_1_meg.fif')) == True:
        meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'block_1_meg.fif'))
    else:
        meg = mne.io.read_raw_fif(os.path.join(meg_dir, 'after_ica_meg.fif'))
    info = meg.info

    # Load the transformation matrix
    trans = mne.read_trans(os.path.join(mri_dir, 'trans' , 'meg_mri-trans.fif'))
    
    # solve the Forward problem
    fwd = mne.make_forward_solution(info=info, trans=trans, src=src, bem=bem_solution, meg=True, eeg=False, mindist=0.0,
                                    ignore_ref=False, n_jobs=1)  
    mne.write_forward_solution(os.path.join(mri_dir, 'file-fwd.fif'), fwd, overwrite=True)

    return fwd
    

