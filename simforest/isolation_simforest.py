import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from simforest.splitter import find_split, find_split_classification
from simforest.distance import dot_product
from multiprocessing import Pool


def _h(n):
    """A function estimating average external path length of Similarity Tree.
        Since Similarity Tree, the same as Isolation tree, has an equivalent structure to Binary Search Tree,
        the estimation of average h for external node terminations is the same as the unsuccessful search in BST.
        Parameters
        ----------
        n : int, number of objects used for tree construction
        Returns
        ----------
        average external path length : int
    """
    assert n - 1 > 0
    return 2 * np.log(n - 1) + np.euler_gamma - 2 * (n - 1) / n


class IsolationSimilarityForest(BaseEstimator):

    def __init__(self,
                 random_state=None,
                 n_estimators=20,
                 sim_function=dot_product,
                 max_depth=8,
                 most_different=False,
                 max_samples=256,
                 contamination='auto'):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.most_different = most_different
        self.max_samples = max_samples
        self.contamination = contamination

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba."""

        X = check_array(X)

        return X

    def fit(self, X, y=None, check_input=True):
        if check_input:
            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X)


        # Random state
        if self.random_state is not None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = np.random.RandomState()

        self.n_ = X.shape[0]

        # Compute sub-sample size for each tree
        all_idxs = range(X.shape[0])

        # Calculate sample size for each tree
        if self.max_samples == 'auto':
            sample_size = min(256, X.shape[0])
        elif isinstance(self.max_samples, float):
            n = len(X.shape[0])
            sample_size = int(self.max_samples * n)
            if sample_size > n:
                sample_size = n

        elif isinstance(self.max_samples, int):
            sample_size = self.max_samples
            n = X.shape[0]
            if sample_size > n:
                sample_size = n
        else:
            raise ValueError('max_samples should be \'auto\' or either float or int')

        self.sample_size = sample_size

        self.estimators_ = []
        for i in range(self.n_estimators):

            subsample_idxs = random_state.choice(all_idxs, self.sample_size, replace=False)

            tree = IsolationSimilarityTree(sim_function=self.sim_function,
                                           random_state=self.random_state,
                                           max_depth=self.max_depth,
                                           most_different=self.most_different)
            tree.fit(X[subsample_idxs], check_input=False)

            self.estimators_.append(tree)

        assert len(self.estimators_) == self.n_estimators, f'Build {len(self.estimators_)} trees, ' \
                                                           f'instead of {self.n_estimators}'
        self.is_fitted_ = True

        return self

    def decision_function(self, X, check_input=True):
        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X)

        # Average depth at which a sample lies
        mean_path_lengths = np.mean([t.path_length_(X, check_input=False) for t in self.estimators_], axis=0)
        assert len(mean_path_lengths) == len(X)

        # Depths are normalized in the same fashion as in Isolation Forest
        c = _h(self.sample_size)
        scores = np.array([- 2 ** (-pl / c) for pl in mean_path_lengths])

        if self.contamination == 'auto':
            offset_ = -0.5

        elif isinstance(self.contamination, float):
            assert self.contamination > 0.0
            assert self.contamination < 0.5

            offset_ = np.percentile(scores, 100. * self.contamination)
        else:
            raise ValueError('contamination should be set either to \'auto\' or a float value between 0.0 and 0.5')

        return scores - offset_

    def predict(self, X, check_input=True):
        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X)

        decision_function = self.decision_function(X, check_input=False)
        print(decision_function[:200])

        is_inlier = np.ones(X.shape[0], dtype=int)
        is_inlier[decision_function < 0] = -1
        return is_inlier


class IsolationSimilarityTree:

    def __init__(self,
                 sim_function=dot_product,
                 random_state=None,
                 max_depth=8,
                 most_different=False,
                 depth=1):
        self.sim_function = sim_function
        self.random_state = random_state
        self.max_depth = max_depth
        self.most_different = most_different
        self.depth = depth

    def sample_directions(self, X, random_state):
        X = np.unique(X, axis=0)
        first, second = random_state.choice(range(X.shape[0]), 2, replace=False)

        assert not np.array_equal(first, second)
        return X[first], X[second]

    def find_split(self, X, random_state):
        similarities = self.sim_function(X, self._p, self._q)
        indices = sorted([i for i in range(len(similarities)) if not np.isnan(similarities[i])],
                         key=lambda x: similarities[x])

        if self.most_different:
            # most different consecutive elements:
            i = np.argmax(np.abs(np.ediff1d(similarities[indices])))
            split_point = (similarities[indices[i]] + similarities[indices[i + 1]]) / 2
        else:
            # random split point
            similarities_min = np.min(similarities)
            similarities_max = np.max(similarities)
            split_point = random_state.uniform(similarities_min, similarities_max, 1)

        return split_point, similarities

    def fit(self, X, y=None, check_input=False):
        # Check input
        if check_input:

            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X)

        # Random state
        if self.random_state is not None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = np.random.RandomState()

        self._lhs = None
        self._rhs = None
        self._p = None
        self._q = None
        self._similarities = []
        self._split_point = -np.inf
        self._value = None
        self._is_leaf = False
        self.is_fitted_ = False
        self.n_ = X.shape[0]

        # one data-point, or multiple copies of the same data-point
        if X.shape[0] == 1 or len(np.unique(X, axis=0)) == 1:
            self._is_leaf = True
            return self

        if self.max_depth is not None:
            if self.depth == self.max_depth:
                self._is_leaf = True
                return self

        self._p, self._q = self.sample_directions(X, random_state)
        self._split_point, self._similarities = self.find_split(X, random_state)

        # Left- and right-hand side partitioning
        lhs_idxs = np.nonzero(self._similarities - self._split_point <= 0)[0]
        rhs_idxs = np.setdiff1d(range(len(X)), lhs_idxs)
        assert len(lhs_idxs) + len(rhs_idxs) == self.n_

        if len(lhs_idxs) > 0 and len(rhs_idxs) > 0:
            self._lhs = IsolationSimilarityTree(sim_function=self.sim_function, random_state=self.random_state,
                                                max_depth=self.max_depth, most_different=self.most_different,
                                                depth=self.depth + 1).fit(X[lhs_idxs], check_input=False)

            self._rhs = IsolationSimilarityTree(sim_function=self.sim_function, random_state=self.random_state,
                                                max_depth=self.max_depth, most_different=self.most_different,
                                                depth=self.depth + 1).fit(X[rhs_idxs], check_input=False)

        else:
            print(f'similarities: {self._similarities}')
            print(f'split point: {self._split_point}')
            raise ValueError('Left- and right-hand-side indexes havn\'t been found,'
                             'even though the split had been found')

        self.is_fitted_ = True

        return self

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba."""

        X = check_array(X)

        return X

    def path_length_(self, X, check_input=True):
        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X)

        return np.array([self.row_path_length_(x.reshape(1, -1)) for x in X])

    def depth_estimate(self):
        c = 0
        n = self.n_
        if n > 1:
            c = _h(n)
        return self.depth + c

    def row_path_length_(self, x):
        if self._is_leaf:
            return self.depth_estimate()

        assert self._p is not None
        assert self._q is not None

        t = self._lhs if self.sim_function(x, self._p, self._q)[0] <= self._split_point else self._rhs
        if t is None:
            return self.depth_estimate()

        return t.row_path_length_(x)
