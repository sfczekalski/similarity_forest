import numpy as np
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state
from scipy.special import comb
from scipy.spatial.distance import squareform
from ._cluster import CSimilarityForestClusterer
import hdbscan


class SimilarityForestCluster(BaseEstimator, ClusterMixin):
    """Similarity forest clusterer.
            A similarity forest is a meta estimator that fits a number of similarity trees on various sub-samples
            of the dataset. These trees perform data partitioning in order to capture its structure.
            Each pair of data-points traverses down the trees, and average depth on which the pair splits is recorded.
            This values serves as a similarity measure between the pair, and is used for hierarchical clustering.

            The sub-sample size is always the same as the original input sample size but the samples are drawn
            with replacement.

            Parameters
            ----------
            random_state : int or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If None, the random number generator is the RandomState instance used by `np.random`.
            sim_function : string, function used to measure similarity between data-points.
                Possible functions are (for now): 'euclidean' (default) for euclidean distance and 'dot' for dot product
            max_depth : integer or None, optional (default=None)
                The maximum depth of the tree. If None, the trees are fully grown
            n_clusters : int (default=20), number of clusters to find in the data
            technique : string (default='ahc')
                clustering algorithm to use to form the clusters from distance matrix produced by the forest.
                Other possible value is 'hdbscan' for HDBSCAN algorithm
            n_estimators : int (default=20), number of trees to grow

            Attributes
            ----------
            forest_ : underlying Cython implementation of the forest
            estimators_ : list of underlying Cython trees
            distance_matrix_ : array of shape (n_samples, n_samples), array of pairwise distances
            links_ : links produced by AHC algorithm
                refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
            labels_ : cluster labels

    """

    def __init__(self,
                 random_state=None,
                 sim_function='dot',
                 max_depth=5,
                 n_clusters=8,
                 technique='ahc',
                 n_estimators=20,
                 bootstrap=False):
        self.random_state = random_state
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.n_clusters = n_clusters
        self.technique = technique
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba."""
        X = check_array(X)

        return X

    def fit(self, X, y=None, check_input=True):
        """Build a forest of trees from the training set (X, y=None)
                Parameters
                ----------
                X : array-like matrix of shape = [n_samples, n_features]
                    The training data samples.
                y : None
                    y added to follow the API.
                check_input : bool
                    Whenever to check input samples or not. Don't change it unless you know what you're doing.
                Returns
                -------
                self : object.
        """
        # Check input
        if check_input:

            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X)

            X = X.astype(np.float32)

        self.estimators_ = None
        self.forest_ = None
        self.distance_matrix_ = None
        self.links_ = None
        self.labels_ = None

        args = dict()

        if self.random_state is not None:
            args['random_state'] = self.random_state

        if self.max_depth is not None:
            args['max_depth'] = self.max_depth

        if self.n_estimators is not None:
            args['n_estimators'] = self.n_estimators

        args['sim_function'] = self.sim_function

        if self.bootstrap:
            args['bootstrap'] = 1

        self.forest_ = CSimilarityForestClusterer(**args)
        self.forest_.fit(X)
        self.estimators_ = self.forest_.estimators_

        if X.shape[0] > 1:
            self.distance_matrix_ = self.forest_.predict_(X)

            if self.technique == 'ahc':
                self.links_ = linkage(self.distance_matrix_)
                clusters = fcluster(
                    self.links_, self.n_clusters, criterion='maxclust')

            elif self.technique == 'hdbscan':
                hdb = hdbscan.HDBSCAN(metric='precomputed')
                square_distance_matrix = squareform(
                    self.distance_matrix_.astype(float))
                hdb.fit(square_distance_matrix)
                clusters = hdb.labels_

            # cluster labels should start from 0
            clusters = clusters - 1
            assert len(clusters) == X.shape[0]
            self.labels_ = clusters

        else:
            self.labels_ = 0

        return self

    def fit_predict(self, X, y=None):
        """Build a forest of trees from the training set (X, y=None) and return cluster labels
                Parameters
                ----------
                    X : array-like matrix of shape = [n_samples, n_features]
                        The training data samples.
                    y : None
                        y added to follow Sklearn API.
                Returns
                -------
                    p : array of shape = [n_samples,]
                    Predicted labels of the input samples' clusters.
        """
        self.fit(X)
        return self.labels_
