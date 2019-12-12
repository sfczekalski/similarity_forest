from simforest.cluster import SimilarityForestCluster
import numpy as np
from sklearn.datasets import load_iris
from scipy.special import comb

X, y = load_iris(return_X_y=True)

sf = SimilarityForestCluster()
sf.fit(X)
print(sf.sf_distance(X)[:100])
