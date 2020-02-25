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
    """An algorithm for outlier detection based on Similarity Forest.
        It borrows the idea of isolating data-points by performing random splits, as in Isolation Forest,
        but they are not performed on features, but in the same way as in Similarity Forest.

            Parameters
            ----------
                random_state : int, RandomState instance or None, optional (default=None)
                    If int, random_state is the seed used by the random number generator;
                    If RandomState instance, random_state is the random number generator;
                    If None, the random number generator is the RandomState instance used
                    by `np.random`.
                n_estimators : integer, optional (default=20)
                    The number of trees in the forest.
                sim_function : function used to measure similarity between data-points
                max_depth : integer or None, optional (default=None)
                    The maximum depth of the tree. If None, then nodes are expanded until
                    all leaves are pure.
                max_samples : int or float
                    size of subsamples used for fitting trees, if int then use number of objects provided, if float then
                    use fraction of whole sample
                contamination : string or float (default='auto'), fraction of expected outliers in the data. If auto then
                    use algorithm criterion described in Isolation Forest paper. Float means fraction of objects that
                     should be considered outliers.
                most_different : bool (default = False)
                    when we don't use strategy of finding split minimizing Gini impurity, we may choose one that finds
                    most different consecutive elements, and splits at this point. Used for outlier detection.

                Attributes
                ----------
                    base_estimator_ : SimilarityIsolationTree
                        The child estimator template used to create the collection of fitted
                        sub-estimators.
                    estimators_ : list of SimilarityTreeClassifiers
                        The collection of fitted sub-estimators.
                    is_fitted_ : bool flag indicating whenever fit has been called

                Notes
                -----
                    To obtain a deterministic behaviour during
                    fitting, ``random_state`` has to be fixed.
            """

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
        """Validate X."""

        X = check_array(X)

        return X

    def fit(self, X, y=None, check_input=True):
        """Build a forest of trees from the training set X.

            Parameters
            ----------
                X : array-like matrix of shape = [n_samples, n_features]
                    The training data samples.
                y : None, added to follow Scikit-Learn convention
                check_input : bool (default=False), allows to skip input validation multiple times.
            Returns
            -------
                self : object.
                """
        if check_input:
            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X)


        # Initialize random number generator instance
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

        # Build trees
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
        """Average anomaly score of X of the base classifiers.
            The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.
            The measure of normality of an observation given a tree is the depth of the leaf containing this observation
            which is equivalent to the number of splittings required to isolate this point.

            Parameters
            ----------
                X : array-like, shape (n_samples, n_features), the input samples.
                check_input : bool indicating if input should be checked or not.

            Returns
            -------
                scores : ndarray, shape (n_samples,)
                    The anomaly score of the input samples. The lower, the more abnormal.
                    Negative scores represent outliers, positive scores represent inliers.
        """
        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X)

        # Average depth at which a sample lies over all trees
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
        """Predict if a particular sample is an outlier or not.
            Paramteres
            ----------
                X : array-like, shape (n_samples, n_features)
                    The input samples.
                check_input : bool indicating if input should be checked or not.
            Returns
            -------
                is_inlier : array, shape (n_samples,) For each observation, tell whether or not (+1 or -1) it should be
                considered as an inlier according to the fitted model.
        """
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
    """Unsupervised Similarity Tree measuring outlyingness score.
        Isolation Similarity Trees are base models used as building blocks for Isolation Similarity Forest ensemble.
            Parameters
            ----------
            random_state : int, random numbers generator seed
            sim_function : function used to measure similarity between data-points
            max_depth : integer or None, optional (default=None)
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure.
            depth : int depth of the tree count
            most_different : bool (default = False)
                when we don't use strategy of finding split minimizing Gini impurity, we may choose one that finds
                most different consecutive elements, and splits at this point. Used for outlier detection.

            Attributes
            ----------
            is_fitted_ : bool flag indicating whenever fit has been called

            _lhs : SimilarityTreeClassifier current node's left child node
            _rhs : SimilarityTreeClassifier current node's right child node
            _p : first data-point used for drawing split direction in the current node
            _q : second data-point used for drawing split direction in the current node
            _similarities :
                ndarray of similarity values between two datapoints used for splitting and rest of training datapoints
            _split_point = float similarity value decision boundary
            _value = class value probabilities for current node, estimated based on training set
            _is_leaf :
                bool indicating if current node is a leaf, because it is pure or stopping createrion
                has been reached (depth == max_depth)
            n_ : int, number of data-points in current node

    """

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
        self.is_fitted_ = False
        self.n_ = X.shape[0]

        # If there is only one data-point, or multiple copies of the same data-point, stop growing a tree.
        if X.shape[0] == 1 or len(np.unique(X, axis=0)) == 1:
            self._is_leaf = True
            return self

        # Max depth reached
        if self.max_depth is not None:
            if self.depth == self.max_depth:
                self._is_leaf = True
                return self

        # Sample data-points used to draw split direction
        self._p, self._q = self.sample_directions(X, random_state)

        # Project all data-points onto the split direction
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
        """Validate X."""

        X = check_array(X)

        return X

    def path_length_(self, X, check_input=True):
        """Get path length for instances of X.
            Parameters
            ----------
                X : array-like, shape (n_samples, n_features)
                    The input samples.
            Returns
            -------
                path_length : ndarray, shape (n_samples,)
                    The path_length for instances of X, according to a single tree.
        """
        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X)

        return np.array([self.row_path_length_(x.reshape(1, -1)) for x in X])

    def depth_estimate(self):
        """Based on depth of not-fully grown tree and number of data-points in current node,
            estimate depth of fully grown tree.
                Returns
                -------
                    depth : int, estimate of fully grown tree.
        """
        c = 0
        n = self.n_
        if n > 1:
            c = _h(n)
        return self.depth + c

    def row_path_length_(self, x):
        """ Get outlyingness path length of a single data-point.
            If current node is a leaf, then return its depth, if not, traverse down the tree.
            If number of objects in external node is more than one, then add an estimate of sub-tree depth,
            if it was fully grown.

            Parameters
            ----------
            x : a data-point
            Returns
            -------
            data-point's path length, according to single a tree.
        """
        if self._is_leaf:
            return self.depth_estimate()

        assert self._p is not None
        assert self._q is not None

        t = self._lhs if self.sim_function(x, self._p, self._q)[0] <= self._split_point else self._rhs
        if t is None:
            return self.depth_estimate()

        return t.row_path_length_(x)
