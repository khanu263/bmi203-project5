# Imports
import pytest
import numpy as np
from sklearn.metrics import silhouette_score
from cluster import *

# Test error-checking in initialization
def test_initialization():

    with pytest.raises(AssertionError, match = "Invalid metric string."):
        s = Silhouette(metric = "foo")

# Test a simple example worked out by hand
def test_simple_example():

    c = np.array([[0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [3, 0]])
    l = np.array([0, 0, 0, 1, 1, 1])
    pred_score = Silhouette().score(c, l)
    real_score = [0.33945528, 0.65398109, 0.59129809, 0.33945528, 0.65398109, 0.59129809]

    assert np.allclose(pred_score, real_score), "Computed score does not match actual score."

# Compare against sklearn implementation
def test_against_sklearn():

    c, _ = make_clusters()
    km = KMeans(3)
    km.fit(c)
    pred = km.predict(c)
    class_score = Silhouette().score(c, pred)
    sklearn_score = silhouette_score(c, pred)

    assert np.isclose(np.mean(class_score), sklearn_score), "Mean score does not match sklearn implementation."