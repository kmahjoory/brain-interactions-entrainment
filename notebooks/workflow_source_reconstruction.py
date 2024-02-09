import matplotlib.pyplot as plt
import mne
from mne.datasets import sample, fetch_fsaverage
from mne.beamformer import make_lcmv, apply_lcmv


data_cov = mne.compute_covariance(epoch, tmin=0.005, tmax=1,
                                  method='empirical')
noise_cov = mne.compute_covariance(epoch, tmin=-0.3, tmax=0,
                                   method='empirical')
#data_cov.plot(epoch.info)







# Spatial filter
filters = make_lcmv(evoked.info, fwd, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power',
                    weight_norm='unit-noise-gain', rank=None)

# Apply spatial filter
stc = apply_lcmv(evoked, filters, max_ori_out='signed')

vertno_max, time_max = stc.get_peak(hemi='rh')
surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
brain = stc.plot(**surfer_kwargs)


# MNE
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, data_cov, loose=0., depth=0.8,
                            verbose=True)


snr = 3.0
lambda2 = 1.0 / snr ** 2
kwargs = dict(initial_time=0.08, hemi='lh', subjects_dir=subjects_dir,
              size=(600, 600), clim=dict(kind='percent', lims=[90, 95, 99]),
              smoothing_steps=7)


stc = abs(mne.minimum_norm.apply_inverse(evoked, inv, lambda2, 'MNE', verbose=True))
brain = stc.plot(figure=1, **kwargs)
brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)


stc = abs(mne.minimum_norm.apply_inverse(evoked, inv, lambda2, 'dSPM', verbose=True))
brain = stc.plot(figure=2, **kwargs)
brain.add_text(0.1, 0.9, 'dSPM', 'title', font_size=14)