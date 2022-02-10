import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """

        # Like the KMeans class, verify the metric is acceptable
        acceptable = ("braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "euclidean", "jaccard", "seuclidean", "sqeuclidean")
        assert metric in acceptable, "Invalid metric string."
        self.metric = metric

    def _point_score(self, i: int, d: np.ndarray, y: np.ndarray) -> float:
        """
        calculate the silhouette score for a given point

        inputs:
            i: int
                index of the point under consideration

            d: np.ndarray
                1D array representing distances to all other points 

            y: np.ndarray
                cluster labels for each of the observations in `X`

        outputs:
            float
                silhouette score for the given point
        """

        # Calculate inter-cluster distances, use the lowest one
        clusters = list(range(max(y) + 1))
        clusters.remove(y[i])
        inter = min(np.sum(d[y == c]) / np.sum(y == c) for c in clusters)

        # Calculate intra-cluster distance
        intra = np.sum(d[y == y[i]]) / (np.sum(y == y[i]) - 1)

        # Return silhouette score
        return (inter - intra) / max(inter, intra)

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        # Make sure number of labels is correct
        assert X.shape[0] == len(y), "Incorrect number of labels."

        # Compute pairwise distances between points in X
        distances = cdist(X, X, metric = self.metric)

        # Return scores
        return np.array([self._point_score(i, distances[i], y) for i in range(X.shape[0])])