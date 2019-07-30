import numpy as np
from scipy.interpolate import interp1d


def fmri_transform(self, time, fmri_tensor, delay, fmri_end):
    k=time
    t_list=[]
    for bnm in range(np.shape(fmri_tensor)[1]):
        fint = interp1d(np.linspace(delay,fmri_end,140), fmri_tensor[0:140,bnm], kind='cubic')
        mmm=fint(k)
        t_list.append(mmm)
    result=(np.array(t_list))
    return result
