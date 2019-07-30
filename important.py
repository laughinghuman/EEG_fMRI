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

b = nib.load('/Users/ilamiheev/Desktop/fmri /FUNC/sub-36_task-rest_bold.nii')


def get_masked_fMRI(y):
    ol = nilearn.image.new_img_like(b, y)
    m, labels = atlas(ol)
    return m, labels


def main():
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
    x_test, y_test, x_fl_test =  sampler.create_one_man_dataset(patient=current_patient, start_time=test_start,
                                                               end_time=test_end)
    y_train = np.stack(y_train, axis=3)
    x_train[np.isinf(x_train)] = 0
    x_fl_train[np.isinf(x_fl_train)] = 0
    y_test = np.stack(y_test, axis=3)
    x_test[np.isinf(x_test)] = 0
    x_fl_test[np.isinf(x_fl_test)] = 0
    preprocessed_train_y, labels = get_masked_fMRI(y_train)
    preprocessed_test_y = get_masked_fMRI(y_test)[0]
    mean_train_y = preprocessed_train_y.mean(0)
    pca = sklearn.decomposition.PCA(n_components=len(labels), random_state=random_state)
    pca.fit(x_train)
    preprocessed_train_x = pca.transform(x_train)
    pca.fit(x_fl_train)
    preprocessed_train_x_fl = pca.transform(x_fl_train)
    ridgereg = Ridge(alpha=1, normalize=True)
    ridgereg.fit(preprocessed_train_x, preprocessed_train_y)
    coefs=[]
    coefs.append(ridgereg.coef_)
    pca = sklearn.decomposition.PCA(n_components=len(labels), random_state=random_state)
    mean_score = (preprocessed_test_y - mean_train_y[None, ...]) ** 2
    pca.fit(x_test)
    preprocessed_test_x = pca.transform(x_test)
    pca.fit(x_fl_test)
    preprocessed_test_x_fl = pca.transform(x_fl_test)
    predicted_test_y = ridgereg.predict(preprocessed_test_x)
    ridgereg.fit(preprocessed_train_x_fl, preprocessed_train_y)
    predicted_test_y_fl = ridgereg.predict(preprocessed_test_x_fl)
    prediction_score = (predicted_test_y - preprocessed_test_y) ** 2
    prediction_score_fl = (predicted_test_y_fl - preprocessed_test_y) ** 2
    zone_score = np.sqrt(np.sum(prediction_score, 0) / np.sum(mean_score, 0))
    zone_score_fl = np.sqrt(np.sum(prediction_score_fl, 0) / np.sum(mean_score, 0))
    l_list = []
    l_fl_list = []
    lo_list=[]
    to_list=[]
    for zone in range(0, len(labels) - 1):
        v = np.mean(abs(predicted_test_y[..., zone] - preprocessed_test_y[..., zone]))
        lo = scipy.stats.pearsonr(preprocessed_test_y[..., zone], predicted_test_y[..., zone])[0]
        j = np.mean(abs(predicted_test_y_fl[..., zone] - preprocessed_test_y[..., zone]))
        to = scipy.stats.pearsonr(preprocessed_test_y[..., zone], predicted_test_y_fl[..., zone])[0]
        lo_list.append(lo)
        to_list.append(to)
        l_list.append(v)
        l_fl_list.append(j)
        plt.plot(preprocessed_test_y[..., zone], label=labels[zone])
        plt.plot(predicted_test_y[..., zone], label='prediction')
        plt.legend()
        plt.show()
    return l_list, l_fl_list, zone_score, zone_score_fl, lo_list, to_list, coeffs
