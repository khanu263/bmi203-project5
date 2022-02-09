import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        # Verify k is acceptable
        assert k > 0, "k must be greater than zero."
        self.k = k

        # Verify metric is acceptable
        # (metrics come from cdist docs, limited to those that work on float vectors with no extra args)
        acceptable = ("braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "euclidean", "jaccard", "seuclidean", "sqeuclidean")
        assert metric in acceptable, "Invalid metric string."
        self.metric = metric

        # Verify tolerance is acceptable
        assert tol > 0, "Tolerance must be greater than zero."
        self.tol = tol

        # Verify iterations are acceptable
        assert max_iter > 0, "Maximum iterations must be greater than 0."
        self.max_iter = max_iter

        # Initialize other variables
        self.error = None
        self.centroids = None
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # Make sure there are enough observations
        assert mat.shape[0] >= self.k, f"There must be at least {self.k} observations."

        # Select k points as the initial centroids and initialize error
        # ("Forgy" initialization as per https://en.wikipedia.org/wiki/K-means_clustering)
        self.centroids = mat[np.random.choice(mat.shape[0], size = self.k, replace = False)]
        self.error = np.inf

        # Since we need to use point-centroid distance twice in a given iteration -- once
        # for cluster assignment and once for error calculation -- we can just calculate
        # it once inside the loop, but we need to initialize the first iteration outside
        distances = cdist(mat, self.centroids, metric = self.metric)

        # Keep going until the max number of iterations has been reached
        for _ in range(self.max_iter):

            # Assign points to clusters (cluster labels are integers, so argmin = label)
            assignments = np.argmin(distances, axis = 1)

            # Update centroids based on assignment
            for i in range(self.k):
                self.centroids[i] = np.mean(mat[assignments == i], axis = 0)

            # Calculate distances to centroids and compute mean-squared error
            distances = cdist(mat, self.centroids, metric = self.metric)
            e = np.mean(np.choose(assignments, distances.T) ** 2)

            # Check if we should stop
            if np.abs(self.error - e) < self.tol:
                self.error = e
                break

            # Set the error before continuing
            self.error = e

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        # Make sure dimensions match before returning assignments
        assert mat.shape[1] == self.centroids.shape[1], "Dimensions do not match."
        return np.argmin(cdist(mat, self.centroids, metric = self.metric), axis = 1)

    def get_error(self) -> float:
        """
        returns the final mean-squared error of the fit model

        outputs:
            float
                the mean-squared error of the fit model
        """

        # Make sure we have an error before returning
        assert self.error is not None, "Please run `fit` before getting the error."
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        # Make sure we have centroids before returning
        assert self.centroids is not None, "Please run `fit` before getting the centroids."
        return self.centroids
