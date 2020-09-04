import mne
from preprocessing import eeg_transform, fmri_transform
from nilearn import image
import numpy as np
from fMRI_Regions import get_masked_fmri


# make new attribute with labels of fmri. Call new attribute in methods
# make something with marks (now we need two files - set and edf. Problem with eeglab that converts set to edf
# and change marks(events) with letters to numbers - it affects generalisation of our program. I need to extract events
# file from eeglab and extract events
# make filtration from 5 to 100 (i want to find delay in 5 seconds)
# Why positive coefficients (positive correlation) in undershoot peak?
# Draw on one picture coefficients, patterns and predict time series
# Make with pandas two tables with results with helium pump on and off
# Draw left and right amigdalla and hypothalamus
# Add number of dimensions of input
# Create new dataset for CCA (1-channel and voxels in cortex)
# shift instead of inverse


class Dataset:
    def __init__(self, random_seed=42, segment_length=16384, eeg_padded=False,
                 frame_creation_time=1950, step=195):
        self.eeg_padded = eeg_padded
        self.segment_length = segment_length
        self.frame_creation_time = frame_creation_time
        self.step = step
        self.random_state = np.random.RandomState(random_seed)

    def create_dataset(self, start_time, end_time, delay, fmri_end, eeg_path, fmri_path):
        vector_exclude = ['EOG', 'ECG', 'CW1', 'CW2', 'CW3', 'CW4', 'CW5', 'CW6', 'Status']
        raw = mne.io.read_raw_edf(eeg_path, exclude=vector_exclude)
        eeg = raw.get_data()
        eeg = mne.filter.filter_data(eeg, sfreq=1000, l_freq=5, h_freq=100)
        eeg_flip = np.fliplr(eeg)
        fmri_im = image.smooth_img(fmri_path, fwhm=6)
        fmri = get_masked_fmri(fmri_im, "sub")
        start = start_time
        end = start_time + self.segment_length
        x_list = []
        y_list = []
        x_fl_list = []
        while end < eeg.shape[1] and end <= fmri_end and end < end_time:
            signal = eeg[..., start:end]
            signal_flip = eeg_flip[..., start:end]
            x = eeg_transform(signal)
            y = fmri_transform(end, fmri, delay, fmri_end)
            x1 = eeg_transform(signal_flip)
            x_list.append(x)
            y_list.append(y)
            x_fl_list.append(x1)
            start += self.step
            end += self.step
        x_list = np.array(x_list)
        x_fl_list = np.array(x_fl_list)
        y_list = np.array(y_list)
        return x_list, y_list, x_fl_list
