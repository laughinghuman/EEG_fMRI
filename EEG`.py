import numpy as np
from scipy import signal
import sklearn.decomposition
from fMRI_Regions import atlas_masker
from sklearn.preprocessing import StandardScaler
import pywt
class EEG_transform:
    def __init__(self, nperseg=63, padded=False, scale=1e+5):
        self.nperseg = nperseg
        self.padded = padded
        self.scale = scale
    def transform(self, sig, waveletname='morl'):
        masker, labels = atlas_masker('sub')
        f, t, ft = signal.stft(sig,fs=250, padded=self.padded, nperseg=80)
        ft = ft[:,2:20,:] 
        ft=np.abs(ft)
        #ft = np.log1p(np.abs(ft))
        ft1=signal.resample(ft,60,axis=-1)
        return ft1
   
