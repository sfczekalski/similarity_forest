import numpy as np
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.special import comb
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state


class SimilarityTreeCluster(BaseEstimator):

    def __init__(self,
                 random_state=None,
                 n_directions=1,
                 sim_function=euclidean,
                 max_depth=None,
                 depth=1):
        self.random_state = random_state
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.depth = depth

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba."""

        X = check_array(X)

        return X

    def fit(self, X, y=None):
        pass

    def pair_partition_depth(self, xi, xj):
        pass


class SimilarityForestCluster(BaseEstimator, ClusterMixin):

    def __init__(self,
                 random_state=None,
                 n_estimators=20,
                 n_directions=1,
                 sim_function=euclidean,
                 max_depth=None,
                 depth=1):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.depth = depth

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba."""

        X = check_array(X)

        return X

    def fit(self, X, y=None):
        pass

    def predict(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        self.fit(X)
        self.predict(X)

