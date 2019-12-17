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
        """Return a pair of objects to draw split direction on.
            It makes sure the two objects are different - in case of using bootstrap sample it's possible.

        """
        n = X.shape[0]
        first = random_state.choice(range(n), replace=False)
        others = np.unique(np.where(X - X[first] != 0)[0])
        assert len(others) > 0, 'All points are the same'
        second = random_state.choice(others, replace=False)
        '''print(f'Number of rows to choose first object: {n}')
        print(X)
        print(f'First: {X[first]}')
        print(f'Number of rows to choose second object: {len(others)}')
        print(X[others])'''

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
            # Check that X and y have correct shape
            X, y = check_X_y(X, y)

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

        self._lhs = None
        self._rhs = None
        self._p = None
        self._q = None
        self._similarities = None
        self._split_point = -np.inf
        self._is_leaf = False
        self.is_fitted_ = False

        # Append self to the list of class instances
        self._nodes_list.append(self)

        # Current node id is length of all nodes list. Nodes are numbered from 1, the root node
        self._node_id = len(self._nodes_list)

        if self._is_pure(X):
            self._is_leaf = True
            return self

        if self.max_depth is not None:
            if self.depth == self.max_depth:
                self._is_leaf = True
                return self

        i, j = self._sample_directions(random_state, X)
        self._p, self._q  = X[i], X[j]
        self._split_point, self._similarities = self._find_split(random_state, X, self._p, self._q  )


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
            print(f'Similarities: {self._similarities}')
            print(f'Split point: {self._split_point}')
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

        check_is_fitted(self, ['X_', 'is_fitted_'])

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
                distance[idx] = 1 / np.mean([t.st_distance(X[c], X[r]) for t in self.estimators_])
                idx += 1
            diagonal += 1

        return distance

    def predict(self, X, y=None):
        pass

    def fit_predict(self, X, y=None):
        self.fit(X)
        self.predict(X)

