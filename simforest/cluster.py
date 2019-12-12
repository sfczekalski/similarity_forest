import numpy as np
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.special import comb
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state, check_X_y
from scipy.special import comb


class SimilarityTreeCluster(BaseEstimator):
    # List of all nodes in the tree, that is SimilarityTreeClassifier instances. Shared across all instances
    _nodes_list = []

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

    def _sample_directions(self, random_state, X):
        n = X.shape[0]
        first = random_state.choice(range(n), replace=False)

        # dirty solution. Make sure the second point isn't a copy of the first (when using bootstrap it's possible)
        # when we y information, we could compare that, now we don't
        # TODO find a better one!
        second = np.inf
        for i in range(n):
            second = random_state.choice(range(n), replace=False)
            if not np.array_equal(X[second], X[first]):
                break

        assert second != np.inf, 'Error when choosing unique points to draw split direction. All points are the same.'
        return first, second

    def _find_split(self, random_state, X, p, q):
        n = X.shape[0]
        similarities = np.array([self.sim_function(x, q) - self.sim_function(x, p) for x in X], dtype=np.float16)

        i = random_state.randint(low=0, high=n-1)
        return i, similarities

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
            # Check that X and y have correct shape
            #X, y = check_X_y(X, y)

            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X, check_input)

            if not isinstance(self.n_directions, int):
                raise ValueError('n_directions parameter must be an int')

        if self.random_state is not None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = np.random.RandomState()

        self.X_ = X
        self._lhs = None
        self._rhs = None
        self._p = None
        self._q = None
        self._similarities = None
        self._split_point = -np.inf
        self._value = None
        self._is_leaf = False
        self.is_fitted_ = False

        # Append self to the list of class instances
        self._nodes_list.append(self)

        # Current node id is length of all nodes list. Nodes are numbered from 1, the root node
        self._node_id = len(self._nodes_list)

        if self._is_pure():
            self._is_leaf = True
            return self

        if self.max_depth is not None:
            if self.depth == self.max_depth:
                self._is_leaf = True
                return self

        i, j = self._sample_directions(random_state, X)
        p, q = X[i], X[j]
        self._split_point, self._similarities = self._find_split(random_state, X, p, q)
        self._p = p
        self._q = q


        # Left- and right-hand side partitioning
        lhs_idxs = np.nonzero(self._similarities <= self._split_point)[0]
        rhs_idxs = np.nonzero(self._similarities > self._split_point)[0]

        if len(lhs_idxs) > 0 and len(rhs_idxs) > 0:
            self._lhs = SimilarityTreeCluster(random_state=self.random_state,
                                              n_directions=self.n_directions,
                                              sim_function=self.sim_function,
                                              max_depth=self.max_depth,
                                              depth=self.depth+1). \
                fit(X[lhs_idxs], check_input=False)

            self._rhs = SimilarityTreeCluster(random_state=self.random_state,
                                              n_directions=self.n_directions,
                                              sim_function=self.sim_function,
                                              max_depth=self.max_depth,
                                              depth=self.depth+1). \
                fit(X[rhs_idxs], check_input=False)
        else:
            self._is_leaf = True
            return self

        return self

    def _is_pure(self):
        """Check whenever current node containts all the same elements."""

        return np.unique(self.X_).size == 1

    def st_distance(self, xi, xj):
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
            # Check that X and y have correct shape
            #X, y = check_X_y(X, y)

            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X, check_input)

            if not isinstance(self.n_directions, int):
                raise ValueError('n_directions parameter must be an int')

        self.X_ = X
        self.base_estimator_ = SimilarityTreeCluster
        if self.random_state is not None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = np.random.RandomState()

        self.estimators_ = []
        n = self.X_.shape[0]
        for i in range(self.n_estimators):
            all_idxs = range(n)
            idxs = random_state.choice(all_idxs, n, replace=True)
            tree = SimilarityTreeCluster(random_state=self.random_state, n_directions=self.n_directions,
                                         sim_function=self.sim_function, max_depth=self.max_depth).\
                fit(X[idxs], check_input=False)

            self.estimators_.append(tree)

        assert len(self.estimators_) == self.n_estimators
        self.is_fitted_ = True

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
                distance[idx] = euclidean(X[c], X[r])
                idx += 1
            diagonal += 1

        return distance

    def predict(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        self.fit(X)
        self.predict(X)

