"""
This is a module implementing Similarity Forest Classifier,
as outlined in 'Similarity Forests', S. Sathe and C. C. Aggarwal, KDD 2017' ,
avaiable here: http://saketsathe.net/downloads/simforest.pdf
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from simforest.rcriterion import theil
from ineqpy import atkinson
from simforest.utils import plot_projection
from simforest.splitter import find_split
from simforest.distance import dot_product, rbf
from multiprocessing import Pool


class SimilarityTreeClassifier(BaseEstimator, ClassifierMixin):
    """Similarity Tree classifier implementation.
        Similarity Trees are base models used as building blocks for Similarity Forest ensemble.
            Parameters
            ----------
            random_state : int, random numbers generator seed
            n_directions : int, number of discriminative directions to check at each split.
                               The best direction is chosen, based on child nodes' purity.
            sim_function : function used to measure similarity between data-points
            classes : list of unique labels found in the data
            max_depth : integer or None, optional (default=None)
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure.
            depth : int depth of the tree count

            Attributes
            ----------
            classes_ : array of shape = [n_classes] or a list of such arrays
                The classes labels (single output problem), or a list of arrays of
                class labels (multi-output problem).
            n_classes_ : int or list
                The number of classes (single output problem), or a list containing the
                number of classes for each output (multi-output problem).
            is_fitted_ : bool flag indicating whenever fit has been called

            _lhs : SimilarityTreeClassifier current node's left child node
            _rhs : SimilarityTreeClassifier current node's right child node
            _p : first data-point used for drawing split direction in the current node
            _q : second data-point used for drawing split direction in the current node
            _similarities :
                ndarray of similarity values between two datapoints used for splitting and rest of training datapoints
            _split_point = float similarity value decision boundary
            _value = class value probabilities for current node, estimated based on training set
            _is_pure : bool indicating if current node contains only samples from one class
            _is_leaf :
                bool indicating if current node is a leaf, because it is pure or stopping createrion
                has been reached (depth == max_depth)
            _node_id : int current node id
            _n : int, number of data-points in current node

    """
    def __init__(self,
                 random_state=None,
                 n_directions=1,
                 sim_function=dot_product,
                 classes=None,
                 max_depth=None,
                 depth=1,
                 gamma=None):
        self.random_state = random_state
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.classes = classes
        self.max_depth = max_depth
        self.depth = depth
        self.gamma = gamma

    def get_depth(self):
        """
        Returns the depth of the decision tree.
        The depth of a tree is the maximum distance between the root and any leaf.
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

    def get_n_leaves(self):
        """Returns the number of leaves of the similarity tree."""
        if self is None:
            return 0
        if self._is_leaf:
            return 1
        else:
            return self._lhs.get_n_leaves() + self._rhs.get_n_leaves()

    def _sample_directions(self, random_state, labels, n_directions=1):
        """
            Parameters
            ----------
            random_state : random state object
            labels : class labels used to determine discrimative directions
            n_directions : number of direction pairs to sample in order to choose the one providing the best split

            Returns
            -------
            generator of object pairs' indexes tuples to draw directions on
        """

        # Choose a data-point from random class, and then, choose a data-point from points of other class
        for _ in range(n_directions):
            first = random_state.choice(range(len(labels)), replace=False)
            first_class = labels[first]
            others = np.where(labels != first_class)[0]
            if len(others) == 0:
                raise ValueError('Could not sample p and q from opposite classes!')
            else:
                second = random_state.choice(others, replace=False)

            yield first, second

    def fit(self, X, y, check_input=True):
        """Build a similarity tree classifier from the training set (X, y).
               Parameters
               ----------
               X : array-like of any type, as long as suitable similarity function is provided
                   The training input samples.
               y : array-like, shape = [n_samples]
                   The labels.

               Returns
               -------
               self : object
        """
        # Check input
        if check_input:
            # Check that X and y have correct shape
            X, y = check_X_y(X, y)

            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X, check_input)

        y = np.atleast_1d(y)
        is_classification = is_classifier(self)

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

        if self.classes is None:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = self.classes
        self.n_classes_ = len(self.classes_)

        # Check parameters
        if self.random_state is not None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = np.random.RandomState()

        if not isinstance(self.n_directions, int):
            raise ValueError('n_directions parameter must be an int')

        self._lhs = None
        self._rhs = None
        self._p = None
        self._q = None
        self._similarities = []
        self._split_point = -np.inf
        self._value = None
        self._is_leaf = False
        self.is_fitted_ = False
        self.n_ = len(y)

        # Append self to the list of class instances

        # Current node id is length of all nodes list. Nodes are numbered from 1, the root node
        self._node_id = id(self)

        # Value of predicion
        probs = np.ones(shape=self.n_classes_)
        for i, c in enumerate(self.classes_):
            count = np.where(y == c)[0].size
            probs[i] = count / len(y) + 0.000000001

        self._value = probs
        self._class_prediction = self.classes_[np.argmax(self._value)]

        if not 1.0 - 0.00001 <= self._value.sum() <= 1.0 + 0.00001:
            raise ValueError('Wrong node class probability values.')

        # Return leaf node value
        if self._is_pure(y):
            self._is_leaf = True
            return self

        if len(y) == 1:
            self._is_leaf = True
            return self

        if self.max_depth is not None:
            if self.depth == self.max_depth:
                self._is_leaf = True
                return self

        # Sample n_direction discriminative split directions and find the best one
        best_impurity = 1.0
        best_split_point = -np.inf
        best_p = None
        best_q = None
        similarities = []
        for i, j in self._sample_directions(random_state, y,  self.n_directions):

            impurity, split_point, curr_similarities = find_split(X, y, X[i], X[j], 'gini',
                                                                  self.sim_function, gamma=self.gamma)
            if impurity < best_impurity:
                best_impurity = impurity
                best_split_point = split_point
                best_p = X[i]
                best_q = X[j]
                similarities = curr_similarities

        if best_impurity < 1.0:
            self._split_point = best_split_point
            self._p = best_p
            self._q = best_q
            self._similarities = np.array(similarities)
            self._impurity = best_impurity

            # Left- and right-hand side partitioning
            lhs_idxs = np.nonzero(self._similarities <= self._split_point)[0]
            rhs_idxs = np.nonzero(self._similarities > self._split_point)[0]

            if len(lhs_idxs) > 0 and len(rhs_idxs) > 0:
                params = self.get_params()
                params['depth'] += 1
                params['classes'] = self.classes_

                self._lhs = SimilarityTreeClassifier(**params).fit(X[lhs_idxs], y[lhs_idxs], check_input=False)
                self._rhs = SimilarityTreeClassifier(**params).fit(X[rhs_idxs], y[rhs_idxs], check_input=False)

            else:
                self._is_leaf = True
                return self

        self.is_fitted_= True
        return self

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba."""

        X = check_array(X)
        return X

    def predict(self, X, check_input=True):
        """A reference implementation of a prediction for a classifier.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.

            Returns
            -------
            y : ndarray, shape (n_samples,)
                The labels.
        """


        # Check if fit had been called
        check_is_fitted(self, ['is_fitted_'])

        # Input validation
        X = check_array(X)

        X = self._validate_X_predict(X, check_input)

        return np.array([self.apply_x(x.reshape(1, -1))._class_prediction for x in X])

    def predict_proba(self, X, check_input=True):
        """ Predict class probabilities of the input samples X.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input data-points. Can be of any type, provided that proper function has been given.

            Returns
            -------
            y : ndarray, shape (n_samples,)
                The labels.
            """

        # Check is fit had been called
        check_is_fitted(self, ['is_fitted_'])

        # Input validation
        X = check_array(X)
        X = self._validate_X_predict(X, check_input)

        return np.array([self.apply_x(x.reshape(1, -1))._value for x in X])

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.
            Parameters
            ----------
            X : array-like matrix of shape = [n_samples, n_features]
                The input samples.
            Returns
            -------
            p : array of shape = [n_samples, n_classes_].
                The class log-probabilities of the input samples.
        """
        probas = self.predict_proba(X)
        probas += 1e-10

        vect_log = np.vectorize(np.log)
        return vect_log(probas)

    def apply(self, X, check_input=False):
        """Returns the index of the leaf that each sample is predicted as."""

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([self.apply_x(x.reshape(1, -1))._node_id for x in X])

    def apply_x(self, x, check_input=False):
        """Returns the index of the leaf that each sample is predicted as."""

        if self._is_leaf:
            return self

        t = self._lhs if self.sim_function(x, self._p, self._q, self.gamma)[0] <= self._split_point else self._rhs
        if t is None:
            return self
        return t.apply_x(x)

    def _is_pure(self, y):
        """Check whenever current node containts only elements from one class."""

        return np.unique(y).size == 1


class SimilarityForestClassifier(BaseEstimator, ClassifierMixin):
    """A similarity forest classifier.
            A similarity forest is a meta estimator that fits a number of similarity tree
            classifiers on various sub-samples of the dataset and uses averaging to
            improve the predictive accuracy and control over-fitting.
            The sub-sample size is always the same as the original
            input sample size but the samples are drawn with replacement.
            Parameters
            ----------
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.
            n_estimators : integer, optional (default=20)
                The number of trees in the forest.
            n_directions : int, number of discriminative directions to check at each split.
                               The best direction is chosen, based on child nodes' purity.
            sim_function : function used to measure similarity between data-points
            max_depth : integer or None, optional (default=None)
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure.
            oob_score : bool (default=False)
                Whether to use out-of-bag samples to estimate
                the generalization accuracy.
            bootstrap : bool (default=True)
                whenever to use bootstrap sampling when fitting trees, or a subsample of size described by max_samples

            Attributes
            ----------
            base_estimator_ : SimilarityTreeClassifier
                The child estimator template used to create the collection of fitted
                sub-estimators.
            estimators_ : list of SimilarityTreeClassifiers
                The collection of fitted sub-estimators.
            classes_ : array of shape = [n_classes] or a list of such arrays
                The classes labels (single output problem), or a list of arrays of
                class labels (multi-output problem).
            n_classes_ : int or list
                The number of classes (single output problem), or a list containing the
                number of classes for each output (multi-output problem).
            oob_score_ : float
                Score of the training dataset obtained using an out-of-bag estimate.
            is_fitted_ : bool flag indicating whenever fit has been called

            Notes
            -----
            The default values for the parameters controlling the size of the trees
            (``max_depth``) lead to fully grown and
            unpruned trees which can potentially be very large on some data sets. To
            reduce memory consumption, the size of the trees should be
            controlled by setting those parameter values.
            To obtain a deterministic behaviour during
            fitting, ``random_state`` has to be fixed.
        """
    def __init__(self,
                 random_state=None,
                 n_estimators=20,
                 n_directions=1,
                 sim_function='dot',
                 max_depth=None,
                 oob_score=False,
                 bootstrap=True,
                 gamma=None):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.oob_score = oob_score
        self.bootstrap = bootstrap
        self.gamma = gamma

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba."""

        X = check_array(X)
        return X

    def fit(self, X, y, check_input=True):
        """Build a forest of trees from the training set (X, y)
            Parameters
            ----------
            X : array-like matrix of shape = [n_samples, n_features]
                The training data samples.
            y : array-like matrix of shape = [n_samples,]
                The training data labels.
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

        y = np.atleast_1d(y)
        is_classification = is_classifier(self)

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

        y = np.array(y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = self.classes_.shape[0]
        self.base_estimator_ = SimilarityTreeClassifier

        # Check input
        if self.random_state is not None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = np.random.RandomState()

        if not isinstance(self.n_directions, int):
            raise ValueError('n_directions parameter must be an int')

        # Default similarity functions: dot product or rbf kernel
        if self.sim_function == 'dot':
            self.sim_function = dot_product
        elif self.sim_function == 'rbf':
            self.sim_function = rbf

        self.oob_score_ = 0.0

        self.estimators_ = []
        for i in range(self.n_estimators):

            if self.bootstrap:
                all_idxs = range(len(y))
                idxs = random_state.choice(all_idxs, len(y), replace=True)
                tree = SimilarityTreeClassifier(classes=self.classes_, n_directions=self.n_directions,
                                                sim_function=self.sim_function, random_state=self.random_state,
                                                gamma=self.gamma)
                tree.fit(X[idxs], y[idxs], check_input=False)

                self.estimators_.append(tree)

                if self.oob_score:
                    idxs_oob = np.setdiff1d(np.array(range(y.size)), idxs)
                    self.oob_score_ += tree.score(X[idxs_oob], y[idxs_oob])

            else:
                tree = SimilarityTreeClassifier(classes=self.classes_, n_directions=self.n_directions,
                                                sim_function=self.sim_function, random_state=self.random_state,
                                                max_depth=self.max_depth, gamma=self.gamma)
                tree.fit(X, y, check_input=False)

                self.estimators_.append(tree)

        if self.oob_score:
            self.oob_score_ /= self.n_estimators

        assert len(self.estimators_) == self.n_estimators
        self.is_fitted_ = True
        return self

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.
            Parameters
            ----------
            X : array-like matrix of shape = [n_samples, n_features]
                The input samples.
            Returns
            -------
            p : array of shape = [n_samples, n_classes_].
                The class probabilities of the input samples.
        """
        # Check is fit had been called
        check_is_fitted(self, ['is_fitted_'])

        # Input validation
        X = check_array(X)

        # Check if provided similarity function applies to input
        X = self._validate_X_predict(X, check_input)

        return np.mean([t.predict_proba(X) for t in self.estimators_], axis=0)

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.
            Parameters
            ----------
            X : array-like matrix of shape = [n_samples, n_features]
                The input samples.
            Returns
            -------
            p : array of shape = [n_samples, n_classes_].
                The class log-probabilities of the input samples.
        """
        probas = self.predict_proba(X)
        probas += 1e-10

        vect_log = np.vectorize(np.log)
        return vect_log(probas)

    def predict(self, X, check_input=True):
        """Predict class of the input samples X
            Parameters
            ----------
            X : array-like matrix of shape = [n_samples, n_features]
                The input samples.
            Returns
            -------
            p : array of shape = [n_samples,].
                Predicted classes of the input samples.
        """

        # Check is fit had been called
        check_is_fitted(self, ['is_fitted_'])

        # Input validation
        X = check_array(X)

        # Check if provided similarity function applies to input
        X = self._validate_X_predict(X, check_input)

        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def apply(self, X, check_input=True):
        """Apply trees in the forest to X, return leaf indices."""

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([t.apply(X, check_input=False) for t in self.estimators_]).transpose()


class SimilarityTreeRegressor(BaseEstimator, RegressorMixin):
    """Similarity Tree regressor implementation.
            Similarity Trees are base models used as building blocks for Similarity Forest ensemble.

            Parameters
            ----------
            random_state : int, random numbers generator seed
            n_directions : int, number of discriminative directions to check at each split.
                The best direction is chosen, based on child nodes' purity.
            sim_function : function used to measure similarity between data-points
            max_depth : integer or None, optional (default=None)
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure.
            depth : int depth of the tree count
            discriminative_sampling : bool (default=True)
                Whenever to use discriminative_sampling, that is a strategy used for choosing split points at each node.
                If True, points are chosen in a way that difference between their regression value is greater that one
                standard deviation of y distribution (in the current node).
            criterion : str (default='variance')
                Criterion used to determine split point at similarity line at each node.
                The default 'variance' means that weighted variance of splitted y distributions is minimized.
                Alternatively, we can choose:
                    'step', in this case the split is chosen halfway between consecutive points
                        with most different y value.
                    'theil', at each split the Theil index will be minimized.
                    'atkinson', at each split the Atkinson index will be minimized.
            plot_splits : bool (default=False)
                If set to True, data points projected into similarity line are plotted. Might be helpful when determining
                proper split criterion.

            Attributes
            ----------
            is_fitted_ : bool flag indicating whenever fit has been called
            _lhs : SimilarityTreeClassifier current node's left child node
            _rhs : SimilarityTreeClassifier current node's right child node
            _p : first data-point used for drawing split direction in the current node
            _q : second data-point used for drawing split direction in the current node
            _similarities :
                ndarray of similarity values between two data-points used for splitting and rest of training datapoints
            _split_point = float similarity value decision boundary
            _value = output value for current node, estimated based on training set
            _is_leaf :
                bool indicating if current node is a leaf, because stopping createrion
                has been reached (depth == max_depth)
            _node_id : int current node id
        """
    def __init__(self,
                 random_state=None,
                 n_directions=1,
                 sim_function=dot_product,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 depth=1,
                 discriminative_sampling=True,
                 criterion='variance',
                 plot_splits=False,
                 gamma=None):
        self.random_state = random_state
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.depth = depth
        self.discriminative_sampling = discriminative_sampling
        self.criterion = criterion
        self.plot_splits = plot_splits
        self.gamma = gamma

    def apply(self, X, check_input=False):
        """Returns the index of the leaf that each sample is predicted as."""

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([self.apply_x(x.reshape(1, -1))._node_id for x in X])

    def apply_x(self, x, check_input=False):
        """Returns the index of the leaf that sample is predicted as."""

        if self._is_leaf:
            return self

        t = self._lhs if self.sim_function(x, self._p, self._q)[0] <= self._split_point else self._rhs
        if t is None:
            return self
        return t.apply_x(x)

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

    def get_n_leaves(self):
        """Returns the number of leaves of the similarity tree."""

        if self is None:
            return 0
        if self._lhs is None and self._rhs is None:
            return 1
        else:
            return self._lhs.get_n_leaves() + self._rhs.get_n_leaves()

    def _sample_discriminative_direction(self, random_state, y):
        """Sample a pair of data-points to draw splitting direction on them,
            such that difference of their regression values is at least one std of all regression values distribution

            Parameters
            ----------
                random_state : random state object
                y : output vector

            Returns
            -------
                a pair of data-points indexes to draw split direction on
        """
        first = random_state.choice(range(len(y)), replace=False)
        first_value = y[first]
        min_diff = np.std(y)
        different = np.where(np.abs(y - first_value) > min_diff)[0]
        # if current node is already too pure, sample data-points randomly
        if len(different) == 0:
            first, second = self._sample_random_direction(random_state, y)
        else:
            second = random_state.choice(different, replace=False)

        return first, second

    def _sample_random_direction(self, random_state, y):
        """Randomly sample a pair of data-points to draw splitting direction on them

                Parameters
                ----------
                    random_state : random state object
                    y : output vector

                Returns
                -------
                    a pair of data-points indexes to draw split direction on
        """
        first = random_state.choice(range(len(y)), replace=False)
        first_value = y[first]
        different = np.where(np.abs(y - first_value) > 0.0)[0]
        second = random_state.choice(different, replace=False)

        return first, second

    def _sample_directions(self, random_state, y, n_directions=1):
        """Sample a pair of data-points to draw splitting direction on them.
            By default sampling is performed according to achieve splits that seperate
            data-points with different regression values.

            Parameters
            ----------
            random_state : random state object
            y : output vector
            n_directions : number of direction pairs to sample in order to choose the one providing the best split

            Returns
            -------
            generator of object pairs' indexes tuples to draw directions on
        """
        # Choose two data-points to draw directions on
        for _ in range(n_directions):
            if self.discriminative_sampling:
                first, second = self._sample_discriminative_direction(random_state, y)

            else:
                first, second = self._sample_random_direction(random_state, y)

            assert first is not None
            assert second is not None

            yield first, second

    def fit(self, X, y, check_input=True):
        """Build a similarity tree regressor from the training set (X, y).
               Parameters
               ----------
               X : array-like of any type, as long as suitable similarity function is provided
                   The training input samples.
               y : array-like, shape = [n_samples]
                   The training outputs.

               Returns
               -------
               self : object
        """

        # Check input
        if check_input:
            # Check that X and y have correct shape
            X, y = check_X_y(X, y)

            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X, check_input)

            if self.criterion == 'theil' or self.criterion == 'atkinson':
                if not np.where(y >= 0)[0].size == y.size:
                    raise ValueError('When using Theil or Atkinson indexes, one need to make sure y has all positive values')

        # Check parameters
        random_state = check_random_state(self.random_state)

        if not isinstance(self.n_directions, int):
            raise ValueError('n_directions parameter must be an int')

        self._lhs = None
        self._rhs = None
        self._p = None
        self._q = None
        self._similarities = []
        self._split_point = -np.inf
        self._value = None
        self._is_leaf = False
        self.is_fitted_ = False
        self._impurity = None

        # Append self to the list of class instances

        # Current node id is length of all nodes list. Nodes are numbered from 1, the root node
        self._node_id = id(self)

        # Value of predicion
        self._value = np.mean(y)

        # Current node's impurity
        if self.criterion == 'variance':
            self._impurity = np.var(y)
        elif self.criterion == 'theil':
            self._impurity = theil(y)
        elif self.criterion == 'atkinson':
            self._impurity = atkinson(y)
        else:
            raise ValueError('Unknown split criterion')

        if y.size == 1:
            self._is_leaf = True
            self.is_fitted_ = True
            return self

        if self._is_pure(y):
            self._is_leaf = True
            self.is_fitted_ = True
            return self

        if self.max_depth is not None:
            if self.depth == self.max_depth:
                self._is_leaf = True
                self.is_fitted_ = True
                return self

        if len(y) <= self.min_samples_split:
            self._is_leaf = True
            self.is_fitted_ = True
            return self

        # Sample n_direction discriminative directions and find the best one
        best_impurity = np.inf
        best_split_point = None
        best_p = None
        best_q = None
        similarities = []
        for i, j in self._sample_directions(random_state, y, self.n_directions):

            impurity, split_point, curr_similarities = find_split(X, y, X[i], X[j], self.criterion, self.sim_function, gamma=self.gamma)

            if impurity < best_impurity:
                best_impurity = impurity
                best_p = X[i]
                best_q = X[j]
                best_split_point = split_point
                similarities = curr_similarities

        if best_split_point is None:
            self.is_fitted_ = True
            self._is_leaf = True
            return self

        # if split improves impurity
        if self._impurity - best_impurity > 0.0:
            self._split_point = best_split_point
            self._p = best_p
            self._q = best_q
            self._similarities = np.array(similarities, dtype=np.float32)

            e = 0.000000001
            # Left- and right-hand side partitioning
            lhs_idxs = np.nonzero(self._similarities - self._split_point < e)[0]
            rhs_idxs = np.nonzero(self._similarities - self._split_point > -e)[0]

            if len(lhs_idxs) > 0 and len(rhs_idxs) > 0:
                params = self.get_params()
                params['depth'] += 1
                self._lhs = SimilarityTreeRegressor(**params).fit(X[lhs_idxs], y[lhs_idxs], check_input=False)

                self._rhs = SimilarityTreeRegressor(**params).fit(X[rhs_idxs], y[rhs_idxs], check_input=False)
            else:
                raise ValueError('Left- and right-hand-side indexes havn\'t been found,'
                                 'even though the split had been found')

        # Split doesn't improve impurity, stop growing a tree
        else:
            self.is_fitted_ = True
            self._is_leaf = True
            return self

        return self

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply."""

        X = check_array(X)
        return X

    def predict(self, X, check_input=True):
        """Predict regression value for X.
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.
            Returns
            -------
            y : ndarray, shape (n_samples,)
                Predicted regression values.
        """

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([self.apply_x(x.reshape(1, -1))._value for x in X], dtype=np.float32)

    def _is_pure(self, y):
        """Check whenever current node containts only elements from one class."""

        return np.unique(y).size == 1


class SimilarityForestRegressor(BaseEstimator, RegressorMixin):
    """A similarity forest regressor.
            A similarity forest is a meta estimator that fits a number of similarity tree
            regressors on various sub-samples of the dataset and uses averaging to
            improve the predictive accuracy and control over-fitting.
            The sub-sample size is always the same as the original
            input sample size but the samples are drawn with replacement.
            Parameters
            ----------
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.
            n_estimators : integer, optional (default=20)
                The number of trees in the forest.
            n_directions : int, number of discriminative directions to check at each split.
                            The best direction is chosen, based on child nodes' purity.
            sim_function : function used to measure similarity between data-points
            max_depth : integer or None, optional (default=None)
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure.
            oob_score : bool (default=False)
                Whether to use out-of-bag samples to estimate the R^2 on unseen data.
            discriminative_sampling : bool (default=True)
                Whenever to use discriminative_sampling, that is a strategy used for choosing split points at each node.
                If True, points are chosen in a way that difference between their regression value is greater that one
                standard deviation of y distribution (in the current node).
            bootstrap : bool (default=True)
                Whether bootstrap samples are used when building trees. If False, sub_sample_fraction of dataset
                is used to build each tree.
            sub_sample_fraction : float (default=1.0)
                When bootstrap is set to False, sub_sample_fraction controls fraction of dataset used to build each tree
            criterion : str (default='variance')
                Criterion used to determine split point at similarity line at each node.
                The default 'variance' means that weighted variance of splitted y distributions is minimized.
                Alternatively, we can choose:
                    'step', in this case the split is chosen halfway between consecutive points
                        with most different y value.
                    'theil', at each split the Theil index will be minimized.
                    'atkinson', at each split the Atkinson index will be minimized.

            Attributes
            ----------
            base_estimator_ : SimilarityTreeRegressor
                The child estimator template used to create the collection of fitted
                sub-estimators.
            estimators_ : list of SimilarityTreeRegressors
                The collection of fitted sub-estimators.
            oob_score_ : float
                Score of the training dataset obtained using an out-of-bag estimate.
            is_fitted_ : bool flag indicating whenever fit has been called
            X_ : data used for fitting the forest
            y_ : data labels

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
                 n_estimators=20,
                 n_directions=1,
                 sim_function='dot',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 oob_score=False,
                 discriminative_sampling=True,
                 bootstrap=True,
                 sub_sample_fraction=1.0,
                 criterion='variance',
                 gamma=None):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.oob_score = oob_score
        self.discriminative_sampling = discriminative_sampling
        self.bootstrap = bootstrap
        self.sub_sample_fraction = sub_sample_fraction
        self.criterion = criterion
        self.gamma = gamma

    def apply(self, X, check_input=False):
        """Returns the index of the leaf that each sample is predicted as."""

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([t.apply(X, check_input=False) for t in self.estimators_]).transpose()

    def fit_tree_(self, tree, bootstrap=True):
        """Fit single tree. Function to be passed to parallel map.
            Parameters
            ----------
                tree : SimilarityTreeRegressor, tree to be fitted
                bootstrap : whenever to use bootstrap sample for fitting trees
            Returns
            ----------
                tree : SimilarityTreeRegressor, fitted tree
        """
        n = len(self.y_)
        all_idxs = range(n)

        if bootstrap:
            idxs = self.random_state_.choice(all_idxs, n, replace=True)
        else:
            sample_size = int(self.sub_sample_fraction * n)
            idxs = self.random_state_.choice(all_idxs, sample_size, replace=False)

        tree.fit(self.X_[idxs], self.y_[idxs], check_input=False)

        if self.oob_score:
            idxs_oob = np.setdiff1d(np.array(range(n)), idxs)
            self.oob_score_ += tree.score(self.X_[idxs_oob], self.y_[idxs_oob])

        return tree

    def fit(self, X, y, check_input=True):
        """Build a similarity forest regressor from the training set (X, y).
                Parameters
                ----------
                X : array-like of any type, as long as suitable similarity function is provided
                    The training input samples.
                y : array-like, shape = [n_samples]
                    The training outputs.

                Returns
                -------
                self : object
        """

        # Check input
        if check_input:
            # Check that X and y have correct shape
            X, y = check_X_y(X, y)

            # Input validation, check it to be a non-empty 2D array containing only finite values
            X = check_array(X)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X, check_input)

            if self.criterion == 'theil' or self.criterion == 'atkinson':
                if not np.where(y >= 0)[0].size == y.size:
                    raise ValueError('When using Theil or Atkinson indexes, one need to make sure y has all positive values')

        y = np.atleast_1d(y)

        self.base_estimator_ = SimilarityTreeRegressor

        # Check input
        random_state = check_random_state(self.random_state)

        if not isinstance(self.n_directions, int):
            raise ValueError('n_directions parameter must be an int')

        # Default similarity functions: dot product or rbf kernel
        if self.sim_function == 'dot':
            self.sim_function = dot_product
        elif self.sim_function == 'rbf':
            self.sim_function = rbf

        self.oob_score_ = 0.0

        self.X_ = X
        self.y_ = y
        self.random_state_ = random_state

        self.estimators_ = []
        for i in range(self.n_estimators):
            tree = SimilarityTreeRegressor(n_directions=self.n_directions, sim_function=self.sim_function,
                                           random_state=self.random_state, max_depth=self.max_depth,
                                           min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           discriminative_sampling=self.discriminative_sampling,
                                           criterion=self.criterion, gamma=self.gamma)

            self.estimators_.append(tree)

        pool = Pool(processes=1)
        self.estimators_ = pool.map(self.fit_tree_, self.estimators_)
        pool.close()
        pool.join()

        if self.oob_score:
            self.oob_score_ /= self.n_estimators

        assert len(self.estimators_) == self.n_estimators
        self.is_fitted_ = True

        return self

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply."""

        X = check_array(X)

        return X

    def predict(self, X, check_input=True):
        """Predict regression target for X.
                Parameters
                ----------
                X : array-like, shape (n_samples, n_features)
                    The input samples.
                Returns
                -------
                y : array-like, shape = [n_samples]
                    Array of predicted regression outputs.
        """

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.mean([t.predict(X) for t in self.estimators_], axis=0)
