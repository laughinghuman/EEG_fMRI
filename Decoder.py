import glob
import mne
import numpy as np
import scipy
import scipy.stats
import sklearn.decomposition
from matplotlib import pyplot as plt
from sampler1 import Sampler
from settings import settings
from sklearn.linear_model import Ridge
# from fMRI_Regions import atlas
from fMRI_Regions import atlas_masker

sampler = Sampler('/Users/ilamiheev/Desktop/data/')
from scipy import signal
masker, labels = atlas_masker('sub')
path_new_set='/Users/ilamiheev/Downloads/coredf.set'
path_new='/Users/ilamiheev/Downloads/corrr.edf'
eeg = mne.io.read_raw_edf('/Users/ilamiheev/Downloads/eeg_fmri_data/eeg_off/eeg_off_1.edf',exclude=['EOG', 'ECG','CW1', 'CW2','CW3','CW4','CW5','CW6','Status'])
#eeg = mne.io.read_raw_edf(path_new,exclude=['EOG', 'ECG','CW1', 'CW2','CW3','CW4','CW5','CW6','Status'])
data = eeg.get_data()
channels=eeg.ch_names
eeg1=data
f, t, ft = signal.stft(eeg1[:,0:2000],fs=1000, nperseg=400)
ix_chs = f[2:19]
num_frames = 140
num_train_frames = 95
num_test_frames = 20
nmse_scores_off=[]
pearson_scores_off=[]
r2_scores_off=[]
Coefs=[]
index_dif=[]
eeg_set_off_paths=glob.glob("/Users/ilamiheev/Downloads/eeg_fmri_data/eeg_set_off/*.set")
eeg_edf_off_paths=glob.glob("/Users/ilamiheev/Downloads/eeg_fmri_data/eeg_off/*.edf")
fmri_off_paths=glob.glob("/Users/ilamiheev/Downloads/eeg_fmri_data/fmri_off/*.nii")
forw_delay=0
patient_list=[1,2,3,4]
#find min number of volumes in all dataset and use it as end in interp1d
#plot the components of channels on topomap sculp
        #sto=patient_list[tau]
#bb=mne.io.read_raw_eeglab(path_new_set)
eeg_path='/Users/ilamiheev/Downloads/eeg_fmri_data/eeg_off/eeg_off_1.edf'
#eeg_path=path_new
fmri_path='/Users/ilamiheev/Downloads/eeg_fmri_data/fmri_off/fmri_off_1.nii'
#fmri_path='/Users/ilamiheev/Downloads/eeg_fmri_data/CWL_Data/mri/epi_normalized/rwatrio1_eoec_in-scan_hpump-on.nii'
#another function for finding delay.
delayf = settings.frame_creation_time * 5 + delay
train_start = delayf
train_end = settings.frame_creation_time * (num_train_frames)+delay
test_start = settings.frame_creation_time * (num_frames - num_test_frames)+ delay
test_end = settings.frame_creation_time * (num_frames)+ delay
current_patient = 35
random_state = 42
x_train, y_train, x_fl_train = sampler.create_one_man_dataset(patient=current_patient, start_time=train_start,
                                                          end_time=train_end, delay=delay,fmri_end=fmri_end,eeg_path=eeg_path,fmri_path=fmri_path, forw_delay=forw_delay)
x_test, y_test, x_fl_test = sampler.create_one_man_dataset(patient=current_patient, start_time=test_start,
                                                       end_time=test_end, delay=delay, fmri_end=fmri_end,eeg_path=eeg_path,fmri_path=fmri_path, forw_delay=forw_delay)
alphas = np.logspace(-3, 3, 7)
scores= np.zeros(((len(labels)-1),np.shape(x_test)[1]))
scoresr2 = np.zeros(((len(labels)-1),np.shape(x_test)[1]))
scoresmse1 = np.zeros(((len(labels)-1),np.shape(x_test)[1],len(alphas)))
scoresr21 = np.zeros(((len(labels)-1),np.shape(x_test)[1],len(alphas)))
mm=[]
bbb=[]
x_train=x_train*1e+4
x_test=x_test*1e+4
#change to nested cross validation
#x_train1, x_val1, y_train1, y_val1 = sklearn.model_selection.train_test_split(x_train,y_train, train_size=0.8,test_size=0.2)
x_train1=x_train[0:1500,...]
x_val1=x_train[1600:2000,...]
y_train1=y_train[0:1500,...]
y_val1=y_train[1600:2000,...]
#y_val=
best_alpha=[]
for zone in range(len(labels)-1):
    for a in range((np.shape(x_test)[1])):
        for ii, alpha in enumerate(alphas):
            ridgereg = Ridge(alpha, normalize=False,fit_intercept=True)
            ridgereg.fit(x_train1[:,a,:,:].reshape(np.shape(x_train1)[0],-1), y_train1[:,zone])
            predicted_y_test = ridgereg.predict(x_val1[:,a,:,:].reshape(np.shape(x_val1)[0],-1))
            scoresmse1[zone,a,ii] = sklearn.metrics.mean_squared_error(y_val1[:, zone], predicted_y_test)
            scoresr21[zone,a,ii] = sklearn.metrics.r2_score(y_val1[:, zone], predicted_y_test)
            mm.append(ridgereg)
mmm=np.array(mm)
#mmm=mmm.reshape(scoresmse.shape)
coefs=[]
scores_max1=[]
scores_max2=[]
scores_max3=[]
R=[]
best_param=[]
for zone in range(len(labels)-1):
    ind = np.unravel_index((np.argmax((scoresr21[zone,...]))), scoresr21[zone,...].shape)
    best_param.append(ind)
