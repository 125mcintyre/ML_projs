# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 09:23:16 2019

@author: HMTV4826
"""

import tarfile
import numpy as np
from sklearn.externals import joblib

from traitlets.traitlets import Bunch

# Grab the module-level docstring to use as a description of the
# dataset
MODULE_DOCS = __doc__


def get_california_housing_data(data_path):
    """

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : ndarray, shape [20640, 8]
        Each row corresponding to the 8 feature values in order.

    dataset.target : numpy array of shape (20640,)
        Each value corresponds to the average house value in units of 100,000.

    dataset.feature_names : array of length 8
        Array of ordered feature names used in the dataset.

    dataset.DESCR : string
        Description of the California housing dataset.

    Notes
    ------

    This dataset consists of 20,640 samples and 9 features.
    """
    fileobj = tarfile.open(data_path,mode="r:gz")
    fileobj = fileobj.extractfile('CaliforniaHousing/cal_housing.data')

    cal_housing = np.loadtxt(fileobj, delimiter=',')
    # Columns are not in the same order compared to the previous
    # URL resource on lib.stat.cmu.edu
    columns_index = [8, 7, 2, 3, 4, 5, 6, 1, 0]
    cal_housing = cal_housing[:, columns_index]
    joblib.dump(cal_housing, data_path, compress=6)

    feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                     "Population", "AveOccup", "Latitude", "Longitude"]

    target, data = cal_housing[:, 0], cal_housing[:, 1:]

    # avg rooms = total rooms / households
    data[:, 2] /= data[:, 5]

    # avg bed rooms = total bed rooms / households
    data[:, 3] /= data[:, 5]

    # avg occupancy = population / households
    data[:, 5] = data[:, 4] / data[:, 5]

    # target in units of 100,000
    target = target / 100000.0

    return Bunch(data=data,
                 target=target,
                 feature_names=feature_names,
                 DESCR=MODULE_DOCS)

get_california_housing_data("Y:\Blk1A\Training\Machine Learning\Hands on ML With Python and Tensorflow\datasets\housing\cal_housing.gz")
