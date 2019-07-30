from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiSpheresMasker


# make class fo working with fmri ROI
def atlas_masker(k):
    dataset_cort = datasets.fetch_atlas_harvard_oxford("%s-maxprob-thr25-2mm" % k)
    atlas_filename = dataset_cort.maps
    labels = dataset_cort.labels
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
    return masker, labels


def atlas_masker_spheres(coords, radius):
    masker = NiftiSpheresMasker(coords, radius)
    assert isinstance(masker, object)
    return masker


def get_masked_fmri(y, l):
    m = atlas(y, l)
    return m


def atlas(k, atlas_type):
    masker, labels = atlas_masker(atlas_type)
    time_series = masker.fit_transform(k)
    return time_series
