import mne
from preprocessing import eeg_transform, find_volume_times
from nilearn import image
import numpy as np
from fMRI_Regions import get_masked_fmri


def create_dataset(self, start_time, end_time, num_frames, eeg_path, eeg_set_path, fmri_path):
    vector_exclude = ['EOG', 'ECG', 'CW1', 'CW2', 'CW3', 'CW4', 'CW5', 'CW6', 'Status']
    raw = mne.io.read_raw_edf(eeg_path, exclude=vector_exclude)
    eeg = raw.get_data()
    eeg = mne.filter.filter_data(eeg, sfreq=1000, l_freq=5, h_freq=100)
    eeg_flip = np.fliplr(eeg)
    fmri_im = image.smooth_img(fmri_path, fwhm=6)
    fmri = get_masked_fmri(fmri_im, "sub")
    times = find_volume_times(eeg_set_path)
    start = start_time
    end = start_time + self.segment_length
    x_list = []
    x_fl_list = []
    while end < eeg.shape[1] and end <= times[num_frames] and end < end_time:
        signal = eeg[..., start:end]
        signal_flip = eeg_flip[..., start:end]
        x = eeg_transform(signal)
        x1 = eeg_transform(signal_flip)
        x_list.append(x)
        x_fl_list.append(x1)
        start += self.step
        end += self.step
    x_list = np.array(x_list)
    x_fl_list = np.array(x_fl_list)
    y_list = np.array(fmri)
    return x_list, y_list, x_fl_list
