import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.baseline import rescale
from mne.stats import _bootstrap_ci

data_path = '/Users/ilamiheev/Downloads/eeg_data/boys_7_9/'
raw_edf_name = data_path + '7_9_1.edf'
iter_freqs = [('Alpha', 8, 12)]
# ('Theta', 4, 7)
# [('Alpha', 8, 12)]
# ('Beta', 13, 25),
# ('Gamma', 30, 45)]
event_id, tmin, tmax = 1, 0., 1.500
baseline = None
raw = mne.io.read_raw_edf(raw_edf_name, preload=False)
events = np.array([[42 * 60, 0, 1],
                   [69 * 60, 0, 1],
                   [140 * 60, 0, 1],
                   [160 * 60, 0, 1],
                   [258 * 60, 0, 1],
                   [378 * 60, 0, 1],
                   [428 * 60, 0, 1],
                   [644 * 60, 0, 1],
                   [876 * 60, 0, 1],
                   [978 * 60, 0, 1],
                   [1018 * 60, 0, 1],
                   [1161 * 60, 0, 1],
                   [1176 * 60, 0, 1],
                   [1198 * 60, 0, 1],
                   [1300 * 60, 0, 1]])
# events = mne.make_fixed_length_events(raw, id=1, duration=.250)
times = events[:, 0] / 125
print(np.shape (times))
frequency_map = list()
for band, fmin, fmax in iter_freqs:
    raw = mne.io.read_raw_edf(raw_edf_name, preload=True)
    # raw.pick_types(meg='grad', eog=True)
    raw.filter(fmin, fmax, n_jobs=1,
               l_trans_bandwidth=1,
               h_trans_bandwidth=1,
               fir_design='firwin')
    raw.apply_hilbert(n_jobs=1, envelope=False)
    epochs = mne.Epochs(raw, events, tmin=0., tmax=5.500, baseline=baseline, preload=True)
    epochs.subtract_evoked()
    # epochs = mne.EpochsArray(
    # data=np.abs(epochs.get_data()), info=epochs.info, tmin=epochs.tmin)
    # frequency_map.append(((band, fmin, fmax), epochs.average()))
    data = np.abs(epochs.get_data())
t=[[]]
gfp=[]
for i in range(np.shape(times)[0]):
    t = np.sum(data[i, ...] ** 2, axis=0)
    gfp.append(np.mean(t,axis=0))
gfp=np.array(gfp)
fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
axes.ravel()[::-1])
    # times = average.times*1000
    # times = average.times * 1e3
    # times = events
    # times = raw.times[epochs.events[:, 0]]
    # gfp= epochs.get_data().var(axis=2)[:, 0]
    # gfp = np.sum(average.data ** 2, axis=0)
    #gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
freq_name ='alpha'
ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
ax.axhline(0, linestyle='--', color='grey', linewidth=2)
# ci_low, ci_up = _bootstrap_ci(average.data, random_state=0,
# stat_fun=lambda x: np.sum(x ** 2, axis=0))
# ci_low = rescale(ci_low, average.times, baseline=(None, 0))
# ci_up = rescale(ci_up, average.times, baseline=(None, 0))
# ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
ax.grid(True)
ax.set_ylabel('GFP')
fmin=8
fmax=12
ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax))
# xy=(0.95, 0.8),
# horizontalalignment='right',
# xycoords='axes fraction')
ax.set_xlim(0, 700)