import numpy as np
from scipy import signal


def eeg_transform(self, sig):
    f, t, ft = signal.stft(sig, fs=1000, padded=self.padded, nperseg=400)
    ft = ft[:, 2:19, :]
    ft = np.log1p(np.abs(ft))
    ft1 = signal.resample(ft, 60, axis=-1)
    return ft1
