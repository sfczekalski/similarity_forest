import numpy as np
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.special import comb
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state, check_X_y
from scipy.special import comb


class SimilarityTreeCluster(BaseEstimator):
    """A similarity tree clusterer.


            Parameters
            ----------
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.
            sim_function : function used to measure similarity between data-points
            max_depth : integer or None, optional (default=None)
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure.

            Attributes
            ----------
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used by np.random.
            sim_function : a function used to measure similarity between points.
            max_depth : int or None, optional (default=None)
                The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
                leaves contain less than min_samples_split samples.
            depth : int
                Depth of current Node
            _lhs : a pointer to current Node's left child
            _rhs : a pointer to current Node's right child
            _p : first splitting point
            _q : second splitting point
            _similarities : np.array
                projection of all points on the splitting direction
            _split_point = float
                split threshold on the splitting direction
            _is_leaf : bool
                indicates if current Node is a leaf
            is_fitted_ : bool
                indicates if the tree has been already fitted


    """

    def __init__(self,
                 random_state=None,
                 sim_function=euclidean,
                 max_depth=None,
                 depth=1):
        self.random_state = random_state
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.depth = depth
        self._lhs = None
        self._rhs = None
        self._p = None
        self._q = None
        self._similarities = None
        self._split_point = -np.inf
        self._is_leaf = False
        self.is_fitted_ = False

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba."""

        X = check_array(X)

        return X

    def _sample_directions(self, random_state, X):
        """Return a pair of objects to draw split direction on.
            It makes sure the two objects are different - in case of using bootstrap sample it's possible.

        """
        n = X.shape[0]
        first = random_state.choice(range(n), replace=False)
        others = np.unique(np.where(X - X[first] != 0)[0])
        assert len(others) > 0, 'All points are the same'
        second = random_state.choice(others, replace=False)

        return first, second

    def _find_split(self, random_state, X, p, q):
        similarities = sorted([self.sim_function(x, q) - self.sim_function(x, p) for x in X])
        split_point = random_state.uniform(low=np.min(similarities), high=np.max(similarities))

        return split_point, np.array(similarities, dtype=np.float16)

    def fit(self, X, y=None, check_input=True):
        """Build a forest of trees from the training set (X, y=None)
                Parameters
                ----------
                X : array-like matrix of shape = [n_samples, n_features]
                    The training data samples.
                y : None
                    y added to follow the API.
                Returns
                -------
                self : object.
        """
        # Check input
        if check_input:

            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X, check_input)

        if self.random_state is not None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = np.random.RandomState()

        if self._is_pure(X):
            self._is_leaf = True
            return self

        if self.max_depth is not None:
            if self.depth == self.max_depth:
                self._is_leaf = True
                return self

        i, j = self._sample_directions(random_state, X)
        self._p, self._q = X[i], X[j]
        self._split_point, self._similarities = self._find_split(random_state, X, self._p, self._q  )


        # Left- and right-hand side partitioning
        lhs_idxs = np.nonzero(self._similarities <= self._split_point)[0]
        rhs_idxs = np.nonzero(self._similarities > self._split_point)[0]

        if len(lhs_idxs) > 0 and len(rhs_idxs) > 0:
            self._lhs = SimilarityTreeCluster(random_state=self.random_state,
                                              sim_function=self.sim_function,
                                              max_depth=self.max_depth,
                                              depth=self.depth+1). \
                fit(X[lhs_idxs], check_input=False)

            self._rhs = SimilarityTreeCluster(random_state=self.random_state,
                                              sim_function=self.sim_function,
                                              max_depth=self.max_depth,
                                              depth=self.depth+1). \
                fit(X[rhs_idxs], check_input=False)
        else:
            self._is_leaf = True
            return self

        return self

    def _is_pure(self, X):
        """Check whenever current node containts all the same elements."""

        return np.unique(X, axis=0).shape[0] == 1

    def get_depth(self):
        """Returns the depth of the decision tree.
        The depth of a tree is the maximum distance between the root
        and any leaf.
        """

        check_is_fitted(self, ['is_fitted_'])

        if self is None:
            return 0
        elif self._lhs is None and self._rhs is None:
            return 1
        else:
            l_depth = self._lhs.get_depth()
            r_depth = self._rhs.get_depth()

            if l_depth > r_depth:
                return l_depth + 1
            else:
                return r_depth + 1

    def st_distance(self, xi, xj):
        """Measure on what depth of the tree two objects split
            Parameters
            ----------
                xi, xj : a pair of objects
            Returns
            ----------
                depth : depth on which the pair split
        """
        if self._is_leaf:
            return self.depth

        path_i = self.sim_function(xi, self._q) - self.sim_function(xi, self._p) <= self._split_point
        path_j = self.sim_function(xj, self._q) - self.sim_function(xj, self._p) <= self._split_point

        assert path_i is not None
        assert path_j is not None

        if path_i == path_j:
            # the same path, check if go left or right
            if path_i:
                return self._lhs.st_distance(xi, xj)
            else:
                return self._rhs.st_distance(xi, xj)
        else:
            # different path, return current depth
            return self.depth


class SimilarityForestCluster(BaseEstimator, ClusterMixin):
    """A similarity forest clusterer.
            A similarity forest is a meta estimator that fits a number of similarity tree clusterers on various
            sub-samples of the dataset and computes pairwise similarities between objects by measuring the depth
            at which the pairs split. Based on this similarities, distance matrix is calculated, and Agglomerative
            Hierarchical Clustering is performed.

            Distance matrix is produced by calculating 1/sf_distance(xi, xj), where xi, xj is a given pair of objects.
            It is stored as dense array of length comb(N,2), that is all pairwise combinations. To extract N*N distance
            matrix from this representation use scipy.spatial.distance.squareform.

            The sub-sample size is always the same as the original input sample size but the samples are draw
            with replacement.

            Parameters
            ----------
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.
            n_clusters : int, the number of clusters to form
            n_estimators : integer, optional (default=20)
                The number of trees in the forest.
            sim_function : function used to measure similarity between data-points
            max_depth : integer or None, optional (default=None)
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure.

            Attributes
            ----------
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used by np.random.
            n_clusters  : int, optional, default: 8
                The number of clusters to form.
            n_estimators : int, optional (default=20)
                The number of base estimators in the ensemble.
            sim_function : a function used to measure similarity between points.
            max_depth : int or None, optional (default=None)
                The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
                leaves contain less than min_samples_split samples.
            base_estimator_ : SimilarityTreeCluster
                The child estimator template used to create the collection of fitted sub-estimators.
            estimators_ : list of SimilarityTreeCluster
                The collection of fitted sub-estimators.
            distance_matrix : ndarray of shape (comb(n_samples,2),)
                Array of distances between all points.
            links : ndarray
                The hierarchical clustering encoded as a linkage matrix.
            labels_ : ndarray, of shape (n_samples,)
                Labels of each point
            is_fitted_ : bool
                indicates if forest has been already fitted

            Notes
            -----
            The default values for the parameters controlling the size of the trees
            (``max_depth``) lead to fully grown and
            unpruned trees which can potentially be very large on some data sets. To
            reduce memory consumption, the size of the trees should be
            controlled by setting those parameter values.
            To obtain a deterministic behaviour during fitting, ``random_state`` has to be fixed.
    """
    def __init__(self,
                 random_state=None,
                 n_clusters=3,
                 n_estimators=20,
                 sim_function=euclidean,
                 max_depth=None):
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.base_estimator_ = None
        self.estimators_ = None
        self.distance_matrix = None
        self.links = None
        self.labels_ = None
        self.is_fitted_ = False

    def _validate_X_predict(self, X, check_input):
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
                Returns
                -------
                    self : object.
        """
        # Check input
        if check_input:
            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X, check_input)

        self.base_estimator_ = SimilarityTreeCluster
        if self.random_state is not None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = np.random.RandomState()

        self.estimators_ = []
        n = X.shape[0]
        for i in range(self.n_estimators):
            all_idxs = range(n)
            idxs = random_state.choice(all_idxs, n, replace=True)
            tree = SimilarityTreeCluster(random_state=self.random_state,
                                         sim_function=self.sim_function,
                                         max_depth=self.max_depth).\
                fit(X[idxs], check_input=False)

            self.estimators_.append(tree)

        assert len(self.estimators_) == self.n_estimators
        self.is_fitted_ = True

        self.labels_ = self.predict_(X)
        if isinstance(self.labels_, int):
            pass
        else:
            assert self.labels_.shape[0] == X.shape[0]

        return self

    def sf_distance(self, X):
        """Compute distance between each pair of objects.
            Parameters
            ----------
                X : array of shape (n_samples, n_features)
            Returns
            ----------
                dist : flat array of shape (n_samples choose k), that is number of all pairwise combination of input X
        """
        n = X.shape[0]
        distance = np.ones(shape=(comb(n, 2).astype(int),), dtype=np.float32)

        diagonal = 1
        idx = 0
        for c in range(n):
            for r in range(diagonal, n):
                distance[idx] = 1 / np.mean([t.st_distance(X[c], X[r]) for t in self.estimators_])
                idx += 1
            diagonal += 1

        return distance

    def predict_(self, X, y=None, check_input=True):
        """Predict labels of the input samples' clusters.
            It always gets called after fit, but it's not available to be used directly for the user. One should call
            fit_predict, or access label by labels_ parameter.
            Parameters
            ----------
                X : array-like matrix of shape = [n_samples, n_features]
                    The input samples.
            Returns
            -------
                p : array of shape = [n_samples,]
                    Predicted labels of the input samples' clusters.
        """
        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        self.distance_matrix = self.sf_distance(X)
        assert len(self.distance_matrix) == comb(X.shape[0], 2)
        if len(self.distance_matrix) == 0:
            return 0

        self.links = linkage(self.distance_matrix)

        clusters = fcluster(self.links, self.n_clusters, criterion='maxclust')
        assert len(clusters) == X.shape[0]

        # Sklearn API requires cluster labels to start from 0
        return clusters - 1

    def fit_predict(self, X, y=None):
        """Build a forest of trees from the training set (X, y=None) and return cluster labels
                Parameters
                ----------
                    X : array-like matrix of shape = [n_samples, n_features]
                        The training data samples.
                    y : None
                        y added to follow the API.
                Returns
                -------
                    p : array of shape = [n_samples,]
                    Predicted labels of the input samples' clusters.
        """
        self.fit(X)
        return self.labels_

