# Imports
import pytest
import numpy as np
from cluster import *

# Test error-checking in initialization
def test_initialization():

    with pytest.raises(AssertionError, match = "k must be greater than zero."):
        km = KMeans(0)

    with pytest.raises(AssertionError, match = "Invalid metric string."):
        km = KMeans(3, metric = "foo")
        
    with pytest.raises(AssertionError, match = "Tolerance must be greater than zero."):
        km = KMeans(3, tol = -1e-6)
        
    with pytest.raises(AssertionError, match = "Maximum iterations must be greater than 0."):
        km = KMeans(3, max_iter = -1)

# Make sure you can't fit without enough observations
def test_not_enough_obs():

    km = KMeans(k = 5)
    c, _ = make_clusters(n = 3, k = 3)

    with pytest.raises(AssertionError, match = "There must be at least 5 observations."):
        km.fit(c)

# Try the simplest possible clustering
def test_one_point_per_cluster():

    c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    l = np.array([0, 1, 2, 3, 4])
    km = KMeans(k = 5)
    km.fit(c)
    pred = km.predict(c)

    assert np.all(np.sort(km.get_centroids(), axis = None) == np.sort(c, axis = None)), "Centroids do not match observations."
    assert np.all(np.sort(pred, axis = None) == np.sort(l, axis = None)), "Not all labels are accounted for."
    assert np.isclose(km.get_error(), 0), "Final error is not close to zero."