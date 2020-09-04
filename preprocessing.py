import scipy.signal
import numpy as np
from scipy.interpolate import interp1d
import mne


# add function for set eeg files
# correlation with temperature in case of helium pump off!!!!!!
def eeg_transform(sig):
    f, _, ft = scipy.signal.stft(sig, fs=1000, padded=False, nperseg=400)
    ft = ft[:, 2:19, :]
    ft = np.log1p(np.abs(ft))
    ft1 = scipy.signal.resample(ft, 60, axis=-1)
    return ft1


def fmri_transform(time, fmri, delay, fmri_end):
    k = time
    t_list = []
    for roi in range(np.shape(fmri)[1]):
        fint = interp1d(np.linspace(delay, fmri_end, 140), fmri[0:140, roi], kind='cubic')
        mmm = fint(k)
        t_list.append(mmm)
    result = (np.array(t_list))
    return result


def find_volume_times(eeg_set):
    raw = mne.io.read_raw_eeglab(eeg_set)
    k = mne.events_from_annotations(raw)
    jj = k[1]['mri']
    a, b = np.where(k[0] == jj)
    times = k[0][a[0:, 0]]
    return times


def find_delay_and_end(eeg_set):
    raw = mne.io.read_raw_eeglab(eeg_set)
    k = mne.events_from_annotations(raw)
    jj = k[1]['mri']
    a, b = np.where(k[0] == jj)
    delay = k[0][a[0], 0]
    fmri_end = k[0][a[-1], 0]
    return delay, fmri_end
