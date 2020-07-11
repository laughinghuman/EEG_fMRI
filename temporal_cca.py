# kernel solution of CCA
#pyrcca copy for solving eigenvalues_problem
#make switch between linear kernel in case n_samples <
# use for review  amigdalla activity
# use for review
# correlation between correlation
# why white matter?
# try to estimate
# everything lower then 1Hertz component of bold signal
# maybe default brain networks????????? Is solution ?

import numpy as np


# evaluate linear kernel as dot product
def kernel_evaluate(a):
    kernel = np.dot(a.T, a)
    return kernel

def solving_eigenvalues_problem:
    if kernelcca:
        kernel = [_make_kernel(d, ktype=ktype, gausigma=gausigma,
                               degree=degree) for d in data]
    else:
        kernel = [d.T for d in data]

    nDs = len(kernel)
    nFs = [k.shape[0] for k in kernel]
    numCC = min([k.shape[1] for k in kernel]) if numCC is None else numCC
    crosscovs = [np.dot(ki, kj.T) for ki in kernel for kj in kernel]

    # Allocate left-hand side (LH) and right-hand side (RH)
    # make cca for eigenvalue problem and estimate test part by allocation and fragile end estimate by temporal part
    #make several kernels (linear and differential)
    # make forward solution]
    # units g - t
    # make comments for interpolation  of z - spline
    # After data is collected, classifiers will be trained to discriminate between individual phonemes (e.g., /b/, /d/, /p/, and /t/) as well as place of articulation (labial and dental) and voicing.
    # Evaluation will be performed using 10-fold cross validation and performance will be measured based on accuracy in the classification task.
    # The models trained on TMS-EEG data will then be tested using the random catch trials to evaluate whether the learned models identify features that are applicable to EEG in general or are specific to TMS.
    # Add models for prediction
    # Add sigmoids function
    # add Fourie trandform to our code and stat function
    # add sin/cos functions to solving of eigenvalue problem  /try to optimize this problem using linear algebra  and eigen vector/ortogonolization(normalization)

    LH = np.zeros((sum(nFs), sum(nFs)))
    RH = np.zeros((sum(nFs), sum(nFs)))

    # Fill the page
    # Fill the left and right sides of the eigenvalue problem
    # Apply kernelization for solving in case with little number of samples
    # apply this solution
    for i in range(nDs):
        RH[sum(nFs[:i]): sum(nFs[:i + 1]),
        sum(nFs[:i]): sum(nFs[:i + 1])] = (crosscovs[i * (nDs + 1)]
                                           + reg * np.eye(nFs[i]))

        for j in range(nDs):
            if i != j:
                LH[sum(nFs[:j]): sum(nFs[:j + 1]),
                sum(nFs[:i]): sum(nFs[:i + 1])] = crosscovs[nDs * j + i]

    LH = (LH + LH.T) / 2.
    RH = (RH + RH.T) / 2.

    maxCC = LH.shape[0]
    r, Vs = eigh(LH, RH, eigvals=(maxCC - numCC, maxCC - 1))
    r[np.isnan(r)] = 0
    rindex = np.argsort(r)[::-1]
    comp = []
    Vs = Vs[:, rindex]
    for i in range(nDs):
        comp.append(Vs[sum(nFs[:i]):sum(nFs[:i + 1]), :numCC])
    return comp


