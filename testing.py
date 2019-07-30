import nibabel as nib
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import nilearn.image
import sklearn.decomposition
from sklearn.linear_model import Ridge
from fMRI_Regions import atlas
from sampler import Sampler
from settings import settings
import scipy
import time
sampler = Sampler('/Users/ilamiheev/Desktop/data/')
num_frames = 300
num_train_frames = 210
num_test_frames = 60
train_start = settings.frame_creation_time * 5
train_end = settings.frame_creation_time * (num_train_frames + 5)
test_start = settings.frame_creation_time * (num_frames - num_test_frames)
test_end = settings.frame_creation_time * num_frames
patient_list = sampler.patient_list
current_patient = 35
random_state = 42
x_train, y_train, x_fl_train = sampler.create_one_man_dataset(patient=current_patient, start_time=train_start,
                                                                  end_time=train_end)