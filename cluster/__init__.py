"""
BMI203: Biocomputing algorithms Winter 2022
Assignment 5: k-means and silhouette scoring
"""

from .kmeans import KMeans
from .silhouette import Silhouette
from .utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

__version__ = "0.1.0"