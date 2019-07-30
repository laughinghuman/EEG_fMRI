import mne
from fMRI import fMRI_transform
from EEG import EEG_transform
from settings import settings
from nilearn import image
import numpy as np
from fMRI_Regions import atlas_masker


def get_masked_fMRI(y):
    m = atlas(y)
    return m

def atlas(k):
    masker, labels = atlas_masker('sub')
    time_series = masker.fit_transform(k)
    return time_series


class Sampler:
    def __init__(self, root_dir, random_seed=42,segment_length=16384, eeg_nperseg=100, eeg_padded=False,
                 eeg_scale=1e+5,
                 fmri_scale=4095 ** -1, num_slices=settings.num_slices,
                 frame_creation_time=settings.frame_creation_time,
                 step=50, epi_repeat=1950):
        self.eeg_nperseg = eeg_nperseg
        self.eeg_padded = eeg_padded
        self.eeg_scale = eeg_scale
        self.segment_length = segment_length
        self.num_slices = num_slices
        self.frame_creation_time = frame_creation_time
        self.fmri_scale = fmri_scale
        self.step = step
        self.root_dir = root_dir
        self.random_state = np.random.RandomState(random_seed)
        self.epi_repeat = epi_repeat
        self.patient_list = ['35']

    def create_one_man_dataset(self, patient, start_time, end_time, delay, fmri_end, eeg_path, fmri_path, forw_delay):
        eeg = mne.io.read_raw_edf(eeg_path, exclude=['EOG', 'ECG', 'CW1', 'CW2', 'CW3', 'CW4', 'CW5', 'CW6', 'Status'])
        eeg = eeg.get_data()
        eeg = mne.filter.filter_data(eeg, sfreq=1000, l_freq=5, h_freq=60)
        eeg_flip = np.fliplr(eeg)
        fmri_im = image.smooth_img(fmri_path, fwhm=6)
        fmri = get_masked_fMRI(fmri_im)
        eegHandler = EEG_transform(nperseg=self.eeg_nperseg)
        fmriHandler = fMRI_transform(num_slices=self.num_slices, fmri_scale=self.fmri_scale)
        start = start_time - self.segment_length
        #start = start_time
        end = start_time
        x_list = []
        y_list = []
        x_fl_list = []
        while end < eeg.shape[1] and end <= (fmri_end) and end < end_time:
            signal = eeg[..., start:end]
            signal_flip = eeg_flip[..., start:end]
            x = eegHandler.transform(signal)
            y = fmriHandler.get_fmri(end, fmri, delay, fmri_end)
            x1 = eegHandler.transform(signal_flip)
            x_list.append(x)
            y_list.append(y)
            x_fl_list.append(x1)
            start += self.step
            end += self.step
        x_list = np.array(x_list)
        x_fl_list = np.array(x_fl_list)
        y_list = np.array(y_list)
        return x_list, y_list, x_fl_list
