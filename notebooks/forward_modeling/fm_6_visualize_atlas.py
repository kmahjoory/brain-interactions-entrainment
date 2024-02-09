

# Import Packages
import os, sys, glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.signal as signal

import mne


    # Set subject ID
for subj_id in range(2, 10):

    print(os.listdir('../'))
    subj_name = f'subj_{subj_id}'
    outputs_path = '../../datasets/plots/'
    data_path = '../../datasets/data/'
    meg_dir = os.path.join(data_path, f'subj_{subj_id}', 'meg')
    mri_dir = os.path.join(data_path, f'subj_{subj_id}', 'mri')
    fs_subjs_dir = os.path.join(data_path, 'fs_subjects_dir')


    # Read Atlas labels
    labels = mne.read_labels_from_annot(subject=subj_name, parc='BN_Atlas', hemi='both', 
                                    surf_name='white', annot_fname=None, regexp=None, 
                                    subjects_dir=fs_subjs_dir, sort=True, verbose=False)
    label2index = {label.name: i for i, label in enumerate(labels)}
    label2index['A1/2/3ll_L-lh']

    Brain = mne.viz.get_brain_class()

    # Vsualize atlas rois filled with colors
    atlas_dir = os.path.join(outputs_path, subj_name, 'atlas')
    os.makedirs(atlas_dir, exist_ok=True)   

    atlas_view_names, atlas_view_angles = ['right', 'left'], [0, 180]
    atlas_view_tesselations =  'colored' #colored'#border'

    for k in range(2):
        # visualize atlas labels
        brain = Brain(subj_name, 'both', 'inflated', subjects_dir=fs_subjs_dir, 
                    cortex='low_contrast', background='white', size=(800, 600))
        brain.add_annotation('BN_Atlas', borders=False)
        #brain.add_head() needs head surface in subj_10/surf/lh.seghead  or /subj_10/bem/subj_10-head-dense.fif
        # brain.add_sensors(meg.info, trans='fsaverage') the transformation file needed
        brain.show_view(azimuth=atlas_view_angles[k], distance=450)
        brain.save_image(filename=os.path.join(atlas_dir, f'atlas_{atlas_view_names[k]}_{atlas_view_tesselations}.png'), mode='rgb')
        #brain.save_image(filename=os.path.join(atlas_dir, f'atlas_{atlas_view_names[k]}_{atlas_view_tesselations}.eps'), mode='rgb')

    # Vsualize atlas rois filled with colors
    atlas_dir = os.path.join(outputs_path, subj_name, 'atlas')
    os.makedirs(atlas_dir, exist_ok=True)   

    atlas_view_names, atlas_view_angles = ['right', 'left'], [0, 180]
    atlas_view_tesselations =  'colored' #colored'#border'

    for k in range(2):
        # visualize atlas labels
        brain = Brain(subj_name, 'both', 'pial', subjects_dir=fs_subjs_dir, 
                    cortex='low_contrast', background='white', size=(800, 600))
        brain.add_annotation('BN_Atlas', borders=False)
        #brain.add_head() needs head surface in subj_10/surf/lh.seghead  or /subj_10/bem/subj_10-head-dense.fif
        # brain.add_sensors(meg.info, trans='fsaverage') the transformation file needed
        brain.show_view(azimuth=atlas_view_angles[k], distance=450)
        brain.save_image(filename=os.path.join(atlas_dir, f'atlas_pial_{atlas_view_names[k]}_{atlas_view_tesselations}.png'), mode='rgb')
        #brain.save_image(filename=os.path.join(atlas_dir, f'atlas_{atlas_view_names[k]}_{atlas_view_tesselations}.eps'), mode='rgb')

        # Vsualize atlas rois filled with colors
    atlas_dir = os.path.join(outputs_path, subj_name, 'atlas')
    os.makedirs(atlas_dir, exist_ok=True)   

    atlas_view_names, atlas_view_angles = ['right', 'left'], [0, 180]
    atlas_view_tesselations =  'colored' #colored'#border'

    for k in range(2):
        # visualize atlas labels
        brain = Brain(subj_name, 'both', 'white', subjects_dir=fs_subjs_dir, 
                    cortex='low_contrast', background='white', size=(800, 600))
        brain.add_annotation('BN_Atlas', borders=False)
        #brain.add_head() needs head surface in subj_10/surf/lh.seghead  or /subj_10/bem/subj_10-head-dense.fif
        # brain.add_sensors(meg.info, trans='fsaverage') the transformation file needed
        brain.show_view(azimuth=atlas_view_angles[k], distance=450)
        brain.save_image(filename=os.path.join(atlas_dir, f'atlas_white_{atlas_view_names[k]}_{atlas_view_tesselations}.png'), mode='rgb')
        #brain.save_image(filename=os.path.join(atlas_dir, f'atlas_{atlas_view_names[k]}_{atlas_view_tesselations}.eps'), mode='rgb')


    # Vsualize atlas rois, borders only
    atlas_dir = os.path.join(outputs_path, subj_name, 'atlas')
    os.makedirs(atlas_dir, exist_ok=True)   

    atlas_view_names, atlas_view_angles = ['right', 'left'], [0, 180]
    atlas_view_tesselations =  'border' #colored'#border'

    for k in range(2):
        # visualize atlas labels
        brain = Brain(subj_name, 'both', 'inflated', subjects_dir=fs_subjs_dir, 
                    cortex='low_contrast', background='white', size=(800, 600))
        brain.add_annotation('BN_Atlas', borders=True)
        #brain.add_head() needs head surface in subj_10/surf/lh.seghead  or /subj_10/bem/subj_10-head-dense.fif
        # brain.add_sensors(meg.info, trans='fsaverage') the transformation file needed
        brain.show_view(azimuth=atlas_view_angles[k], distance=450)
        brain.save_image(filename=os.path.join(atlas_dir, f'atlas_{atlas_view_names[k]}_{atlas_view_tesselations}.png'), mode='rgb')
        #brain.save_image(filename=os.path.join(atlas_dir, f'atlas_{atlas_view_names[k]}_{atlas_view_tesselations}.eps'), mode='rgb')


    # Vsualize Auditory SEEDs and atlas rois, LEFT
    stg_l = ['A38m_L-lh', 'A41/42_L-lh', 'TE1.0/TE1.2_L-lh', 'A22c_L-lh',  'A22r_L-lh']    #'A38l_L-lh',

    atlas_seed_dir = os.path.join(outputs_path, subj_name, 'atlas_seed')
    os.makedirs(atlas_seed_dir, exist_ok=True)   

    atlas_view_names, atlas_view_angles = 'left', 180
    atlas_view_tesselations =  'border' #colored'#border'


    # visualize atlas labels
    brain = Brain(subj_name, 'both', 'inflated', subjects_dir=fs_subjs_dir, 
                cortex='low_contrast', background='white', size=(800, 600))
    brain.add_annotation('BN_Atlas', borders=True)
    for seed_l in stg_l:
        brain.add_label(labels[label2index[seed_l]], hemi='lh', color='green', borders=False, alpha=0.8)
    #brain.add_head() needs head surface in subj_10/surf/lh.seghead  or /subj_10/bem/subj_10-head-dense.fif
    # brain.add_sensors(meg.info, trans='fsaverage') the transformation file needed
    brain.show_view(azimuth=atlas_view_angles, distance=450)
    brain.save_image(filename=os.path.join(atlas_seed_dir, f'atlas_{atlas_view_names}_{atlas_view_tesselations}.png'), mode='rgb')
    brain.save_image(filename=os.path.join(atlas_seed_dir, f'atlas_{atlas_view_names}_{atlas_view_tesselations}.eps'), mode='rgb')
    brain.close()

    # Vsualize Auditory SEEDs and atlas rois, RIGHT

    stg_r = ['A38m_R-rh', 'A41/42_R-rh', 'TE1.0/TE1.2_R-rh', 'A22c_R-rh', 'A22r_R-rh'] # , 'A38l_R-rh'

    atlas_seed_dir = os.path.join(outputs_path, subj_name, 'atlas_seed')
    os.makedirs(atlas_seed_dir, exist_ok=True)   

    atlas_view_names, atlas_view_angles = 'right', 0
    atlas_view_tesselations =  'border' #colored'#border'


    # visualize atlas labels
    brain = Brain(subj_name, 'both', 'inflated', subjects_dir=fs_subjs_dir, 
                cortex='low_contrast', background='white', size=(800, 600))
    brain.add_annotation('BN_Atlas', borders=True)
    for seed_r in stg_r:
        brain.add_label(labels[label2index[seed_r]], hemi='rh', color='green', borders=False, alpha=0.8)
    #brain.add_head() needs head surface in subj_10/surf/lh.seghead  or /subj_10/bem/subj_10-head-dense.fif
    # brain.add_sensors(meg.info, trans='fsaverage') the transformation file needed
    brain.show_view(azimuth=atlas_view_angles, distance=450)
    brain.save_image(filename=os.path.join(atlas_seed_dir, f'atlas_{atlas_view_names}_{atlas_view_tesselations}.png'), mode='rgb')
    brain.save_image(filename=os.path.join(atlas_seed_dir, f'atlas_{atlas_view_names}_{atlas_view_tesselations}.eps'), mode='rgb')
    brain.close()

    # Vsualize Auditory SEEDs and atlas rois, borders only

    stg_r = ['A38m_R-rh', 'A41/42_R-rh', 'TE1.0/TE1.2_R-rh', 'A22c_R-rh', 'A22r_R-rh'] # , 'A38l_R-rh'

    atlas_seed_dir = os.path.join(outputs_path, subj_name, 'atlas_seed')
    os.makedirs(atlas_dir, exist_ok=True)   

    atlas_view_names, atlas_view_angles = ['right', 'left'], [0, 180]
    atlas_view_tesselations =  'border' #colored'#border'


    # visualize atlas labels
    brain = Brain(subj_name, 'both', 'inflated', subjects_dir=fs_subjs_dir, 
                cortex='low_contrast', background='white', size=(800, 600))
    brain.add_annotation('BN_Atlas', borders=True)
    brain.add_label(labels[label2index[seed_l]], hemi='lh', color='green', borders=False)
    #brain.add_head() needs head surface in subj_10/surf/lh.seghead  or /subj_10/bem/subj_10-head-dense.fif
    # brain.add_sensors(meg.info, trans='fsaverage') the transformation file needed
    brain.show_view(azimuth=atlas_view_angles[k], distance=450)
    brain.save_image(filename=os.path.join(atlas_dir, f'atlas_{atlas_view_names[k]}_{atlas_view_tesselations}.png'), mode='rgb')
    #brain.save_image(filename=os.path.join(atlas_dir, f'atlas_{atlas_view_names[k]}_{atlas_view_tesselations}.eps'), mode='rgb')
    brain.close()

