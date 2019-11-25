"""
This is a module implementing Similarity Forest Classifier,
as outlined in 'Similarity Forests', S. Sathe and C. C. Aggarwal, KDD 2017' ,
avaiable here: http://saketsathe.net/downloads/simforest.pdf
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier, is_regressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from sklearn.preprocessing import LabelEncoder
from simforest.criterion import find_split_index


def gini_index(split_index, y):
    left_partition, right_partition = y[:split_index], y[split_index:]

    left_gini = 1.0 - np.sum([(np.where(left_partition == cl)[0].size / len(left_partition)) ** 2 for cl in np.unique(y)])
    right_gini = 1.0 - np.sum([(np.where(right_partition == cl)[0].size / len(right_partition)) ** 2 for cl in np.unique(y)])

    left_prop = len(left_partition) / len(y)
    return left_prop * left_gini + (1.0 - left_prop) * right_gini


def _h(n):
    """A function estimating average external path length of Similarity Tree
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
            X_ : data used for fitting the forest
            y_ : data labels
            _nodes_list : list of SimilarityTreeClassifier instances, that is tree nodes

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

    """

    # List of all nodes in the tree, that is SimilarityTreeClassifier instances. Shared across all instances
    _nodes_list = []

    def __init__(self,
                 random_state=None,
                 n_directions=1,
                 sim_function=np.dot,
                 classes=None,
                 max_depth=None,
                 depth=1,
                 discriminative_sampling=True,
                 most_different=False,
                 estimator_samples=None):
        self.random_state = random_state
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.classes = classes
        self.max_depth = max_depth
        self.depth = depth
        self.discriminative_sampling = discriminative_sampling
        self.most_different = most_different
        self.estimator_samples = estimator_samples

    def get_depth(self):
        """Returns the depth of the decision tree.
        The depth of a tree is the maximum distance between the root
        and any leaf.
        """

        check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

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
        """Returns the number of leaves of the similarity tree.
        """
        if self is None:
            return 0
        if self._lhs is None and self._rhs is None:
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
            if self.discriminative_sampling:
                first = random_state.choice(range(len(labels)), replace=False)
                first_class = labels[first]
                others = np.where(labels != first_class)[0]
                if len(others) == 0:
                    first, second = random_state.choice(a=range(len(labels)), size=2, replace=False)
                else:
                    second = random_state.choice(others, replace=False)
            else:
                first, second = random_state.choice(range(len(labels)), 2, replace=False)

            yield first, second

    def draw_sim_line(self, s, p, q, split_point, y):
        import matplotlib.pyplot as plt
        from matplotlib import collections
        import matplotlib as mpl

        fig, ax = plt.subplots()
        mpl.rc('image', cmap='bwr')

        right = [True if s[i] > split_point else False for i in range(len(y))]
        left = [True if s[i] <= split_point else False for i in range(len(y))]

        # right-side lines
        right_lines = []
        for i in range(len(s)):
            if right[i]:
                pair = [(s[i], 0), (s[i], y[i])]
                right_lines.append(pair)
        linecoll_right = collections.LineCollection(right_lines)
        r = ax.add_collection(linecoll_right)
        r.set_alpha(0.9)
        r.set_color('red')
        r = ax.fill_between(s, 0, y, where=right)
        r.set_alpha(0.3)
        r.set_color('red')

        # left-side lines
        left_lines = []
        for i in range(len(s)):
            if not right[i]:
                pair = [(s[i], 0), (s[i], y[i])]
                left_lines.append(pair)
        linecoll_left = collections.LineCollection(left_lines)
        l = ax.add_collection(linecoll_left)
        l.set_alpha(0.9)
        l.set_color('blue')
        l = ax.fill_between(s, 0, y, where=left)
        l.set_alpha(0.3)
        l.set_color('blue')

        # dots at the top
        plt.scatter(s, y, c=right, alpha=0.7)

        # horizontal line
        ax.axhline(c='grey')

        # p and q
        p_similarity = self.sim_function(p, q) - self.sim_function(p, p)
        ax.axvline(p_similarity, c='green')
        plt.text(p_similarity, np.max(y), 'p', c='green')

        q_similarity = self.sim_function(q, q) - self.sim_function(q, p)
        ax.axvline(q_similarity, c='green')
        plt.text(q_similarity, np.max(y), 'q', c='green')

        # split point
        ax.axvline(split_point, c='green')
        plt.text(split_point, np.min(y), 'split point', c='green', rotation=90)

        # titles
        plt.title(f'Split at depth {self.depth}')
        plt.xlabel('Similarity')
        plt.ylabel('y')
        plt.show()

    def _find_split(self, random_state, X, y, p, q):
        """ Find split among direction drew on pair of data-points from different classes
            Parameters
            ----------
            X : all data-points
            y : labels
            p : first data-point used for drawing direction of the split
            q : second data-point used for drawing direction of the split

            Returns
            -------
            best_impurity_decrease : decrease of Gini index after the split
            best_p : first data point from pair providing the best impurity decrease
            best_p : second data point from pair providing the best impurity decrease
            best_criterion : classification threshold
        """
        similarities = np.array([self.sim_function(x, q) - self.sim_function(x, p) for x in X])
        indices = sorted([i for i in range(len(y)) if not np.isnan(similarities[i])],
                         key=lambda x: similarities[x])

        y = np.array(y[indices])
        if y.dtype != int:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)

        y = y.astype(np.int32)
        classes = np.unique(y).astype(np.int32)

        best_impurity = None
        best_p = None
        best_q = None
        best_split_point = None

        n = len(y)
        if self.discriminative_sampling:
            i, best_impurity = find_split_index(y[indices], np.int32(n-1), classes)

        else:
            if self.most_different:
                # most different consecutive elements:
                i = np.argmax(np.abs(np.ediff1d(similarities[indices])))
            else:
                # random split point
                i = random_state.randint(low=0, high=n-1)
            best_impurity = gini_index(i + 1, y[indices])

        best_p = p
        best_q = q
        best_split_point = (similarities[indices[i]] + similarities[indices[i + 1]]) / 2

        #self.draw_sim_line(similarities[indices], best_p, best_q, best_split_point, y[indices])

        return best_impurity, best_p, best_q, best_split_point, similarities

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

        self.X_ = X
        self.y_ = y
        self._lhs = None
        self._rhs = None
        self._p = None
        self._q = None
        self._similarities = []
        self._split_point = -np.inf
        self._value = None
        self._is_leaf = False
        self.is_fitted_ = False

        # Append self to the list of class instances
        self._nodes_list.append(self)

        # Current node id is length of all nodes list. Nodes are numbered from 1, the root node
        self._node_id = len(self._nodes_list)

        # Value of predicion
        probs = np.ones(shape=self.n_classes_)
        for i, c in enumerate(self.classes_):
            count = np.where(y == c)[0].size
            probs[i] = count / len(y) + 0.000000001

        self._value = probs

        if not 1.0 - 0.00001 <= self._value.sum() <= 1.0 + 0.00001:
            raise ValueError('Wrong node class probability values.')

        # Return leaf node value
        if self._is_pure():
            self._is_leaf = True
            return self

        if len(self.y_) == 1:
            self._is_leaf = True
            return self

        if self.max_depth is not None:
            if self.depth == self.max_depth:
                self._is_leaf = True
                return self

        # Sample n_direction discriminative directions and find the best one
        best_impurity = 1.0
        best_split_point = -np.inf
        best_p = None
        best_q = None
        similarities = []
        for i, j in self._sample_directions(random_state, self.y_,  self.n_directions):

            impurity, p, q, split_point, curr_similarities = self._find_split(random_state, X, y, X[i], X[j])
            if impurity < best_impurity:
                best_impurity = impurity
                best_split_point = split_point
                best_p = p
                best_q = q
                similarities = curr_similarities

        if best_impurity < 1.0:
            self._split_point = best_split_point
            self._p = best_p
            self._q = best_q
            self._similarities = np.array(similarities)

            # Left- and right-hand side partitioning
            lhs_idxs = np.nonzero(self._similarities <= self._split_point)[0]
            rhs_idxs = np.nonzero(self._similarities > self._split_point)[0]

            if len(lhs_idxs) > 0 and len(rhs_idxs) > 0:
                self._lhs = SimilarityTreeClassifier(random_state=self.random_state,
                                                     n_directions=self.n_directions,
                                                     sim_function=self.sim_function,
                                                     classes=self.classes_,
                                                     max_depth=self.max_depth,
                                                     depth=self.depth+1,
                                                     discriminative_sampling=self.discriminative_sampling).\
                                                    fit(self.X_[lhs_idxs], self.y_[lhs_idxs], check_input=False)

                self._rhs = SimilarityTreeClassifier(random_state=self.random_state,
                                                     n_directions=self.n_directions,
                                                     sim_function=self.sim_function,
                                                     classes=self.classes_,
                                                     max_depth=self.max_depth,
                                                     depth=self.depth+1,
                                                     discriminative_sampling=self.discriminative_sampling).\
                                                    fit(self.X_[rhs_idxs], self.y_[rhs_idxs], check_input=False)
            else:
                self._is_leaf = True
                return self

        self.is_fitted_= True

        return self

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba."""

        X = check_array(X)

        return X

    def predict_row_class(self, x):
        """ Predict class of a single data-point.
            If current node is a leaf, return its prediction value, if not, traverse down the tree to find point's class

            Parameters
            ----------
            x : a data-point

            Returns
            -------
            data-point's class
        """

        if self._is_leaf:
            return self.classes_[np.argmax(self._value)]

        t = self._lhs if self.sim_function(x, self._q) - self.sim_function(x, self._p) <= self._split_point else self._rhs
        if t is None:
            return self.classes_[np.argmax(self._value)]

        return t.predict_row_class(x)

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
        check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

        # Input validation
        X = check_array(X)

        X = self._validate_X_predict(X, check_input)

        return np.array([self.predict_row_class(x) for x in X])

    def predict_row_prob(self, x):
        """ Predict class of a single data-point.
            If current node is a leaf, return its prediction value, if not, traverse down the tree to find point's class

            Parameters
            ----------
            x : a data-point

            Returns
            -------
            data-point's class
        """

        if self._is_leaf:
            return self._value

        t = self._lhs if self.sim_function(x, self._q) - self.sim_function(x, self._p) <= self._split_point else self._rhs
        if t is None:
            return self._value
        return t.predict_row_prob(x)

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
        check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

        # Input validation
        X = check_array(X)
        X = self._validate_X_predict(X, check_input)

        return np.array([self.predict_row_prob(x) for x in X])

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
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([self.row_path_length_(x) for x in X])

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
            c = 0
            n = len(self.y_)
            if n > 1:
                c = _h(n)
            return self.depth + c

        assert self._p is not None
        assert self._q is not None

        t = self._lhs if self.sim_function(x, self._q) - self.sim_function(x,
                                                                           self._p) <= self._split_point else self._rhs
        if t is None:
            return self.depth

        return t.row_path_length_(x)

    def apply(self, X, check_input=False):
        """Returns the index of the leaf that each sample is predicted as."""

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([self.apply_x(x) for x in X])

    def apply_x(self, x, check_input=False):
        """Returns the index of the leaf that each sample is predicted as."""

        if self._is_leaf:
            return self._node_id

        t = self._lhs if self.sim_function(x, self._q) - self.sim_function(x, self._p) <= self._split_point else self._rhs
        if t is None:
            return self._node_id
        return t.apply_x(x)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree."""

        if check_input:
            # Check is fit had been called
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)
            X = self._validate_X_predict(X, check_input)

        if self._is_leaf:
            return f'In the leaf node, containing samples: \n {list(zip(self.X_, self.y_))}'

        similarity = self.sim_function(X, self._q) - self.sim_function(X, self._p)
        left = similarity <= self._split_point
        t = self._lhs if left else self._rhs
        if t is None:
            return f'In the leaf node, containing samples: \n {list(zip(self.X_, self.y_))}'

        if left:
            print(f'Going left P: {self._p}, \t Q: {self._q}, \t split point: {self._split_point}, \t similarity: {similarity}')
        else:
            print(f'Going right P: {self._p}, \t Q: {self._q}, \t split point: {self._split_point}, \t similarity: {similarity}')

        return t.decision_path(X, check_input=False)

    def _prune_tree(self):
        """Prune tree using Minimal Cost-Complexity Pruning."""
        pass

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        """Compute the pruning path during Minimal Cost-Complexity Pruning."""
        pass

    def _is_pure(self):
        """Check whenever current node containts only elements from one class."""

        return np.unique(self.y_).size == 1

    '''def _more_tags(self):
        return {'binary_only': True}'''

    @property
    def feature_importances_(self):
        """Return the feature importances."""
        pass


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
            oob_decision_function_ : array of shape = [n_samples, n_classes]
                Decision function computed with out-of-bag estimate on the training
                set. If n_estimators is small it might be possible that a data point
                was never left out during the bootstrap. In this case,
                `oob_decision_function_` might contain NaN.
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
            To obtain a deterministic behaviour during
            fitting, ``random_state`` has to be fixed.
        """
    def __init__(self,
                 random_state=None,
                 n_estimators=20,
                 n_directions=1,
                 sim_function=np.dot,
                 max_depth=None,
                 oob_score=False,
                 bootstrap=True,
                 max_samples=None,
                 contamination='auto',
                 discriminative_sampling=True,
                 most_different=False):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.oob_score = oob_score
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.contamination = contamination
        self.discriminative_sampling = discriminative_sampling
        self.most_different = most_different

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

        self.X_ = X
        self.y_ = np.array(y)
        self.classes_ = unique_labels(self.y_)
        self.n_classes_ = self.classes_.shape[0]
        self.base_estimator_ = SimilarityTreeClassifier

        '''
        if self.n_classes_ > 2:
            raise Exception('Similarity Tree is a binary classifier!')
        '''

        # Check input
        if self.random_state is not None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = np.random.RandomState()

        if not isinstance(self.n_directions, int):
            raise ValueError('n_directions parameter must be an int')

        self.oob_score_ = 0.0
        self.oob_decision_function_ = np.ndarray(shape=(X.size, self.n_classes_))
        self.oob_decision_function_.fill(np.nan)
        nans = [np.nan] * X.size

        self.estimators_ = []

        for i in range(self.n_estimators):

            if self.bootstrap:
                all_idxs = range(self.y_.size)
                idxs = random_state.choice(all_idxs, self.y_.size, replace=True)

                tree = SimilarityTreeClassifier(classes=self.classes_, n_directions=self.n_directions,
                                                sim_function=self.sim_function, random_state=self.random_state,
                                                max_depth=self.max_depth,
                                                discriminative_sampling=self.discriminative_sampling,
                                                estimator_samples=idxs)
                tree.fit(X[idxs], y[idxs], check_input=False)

                self.estimators_.append(tree)

                if self.oob_score:
                    idxs_oob = np.setdiff1d(np.array(range(y.size)), idxs)
                    self.oob_score_ += tree.score(X[idxs_oob], y[idxs_oob])

                    tree_probs = tree.predict_proba(X[np.unique(idxs_oob)])

                    self.oob_decision_function_[idxs_oob] = np.add(np.nan_to_num(self.oob_decision_function_[idxs_oob]),
                                                                   tree_probs)
            else:
                all_idxs = range(y.size)

                # Calculate sample size for each tree
                if self.max_samples == 'auto':
                    sample_size = min(256, y.size)
                elif isinstance(self.max_samples, float):
                    sample_size = int(self.max_samples * y.size)
                    assert sample_size <= len(y), f'max_samples cannot be bigger than whole sample size \n' \
                                                        f'max_samples is {sample_size}, sample is {len(self.y_)}'
                elif isinstance(self.max_samples, int):
                    sample_size = self.max_samples
                    assert sample_size <= len(y), f'max_samples cannot be bigger than whole sample size \n' \
                                                        f'max_samples is {sample_size}, sample is {len(self.y_)}'
                else:
                    raise ValueError('max_samples should be \'auto\' or either float or int')

                idxs = random_state.choice(all_idxs, sample_size, replace=False)

                tree = SimilarityTreeClassifier(classes=self.classes_, n_directions=self.n_directions,
                                                sim_function=self.sim_function, random_state=self.random_state,
                                                max_depth=self.max_depth,
                                                discriminative_sampling=self.discriminative_sampling,
                                                estimator_samples=idxs)
                tree.fit(X[idxs], self.y_[idxs], check_input=False)

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
        check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

        # Input validation
        X = check_array(X)

        # Check if provided similarity function applies to input
        X = self._validate_X_predict(X, check_input)

        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def decision_function_outliers(self, X, check_input=True):
        """Average anomaly score of X of the base classifiers.
            The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.
            The measure of normality of an observation given a tree is the depth of the leaf containing this observation
            which is equivalent to the number of splittings required to isolate this point.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.
            check_input : bool indicating if input should be checked or not.
            Returns
            -------
            scores : ndarray, shape (n_samples,)
                The anomaly score of the input samples. The lower, the more abnormal.
                Negative scores represent outliers, positive scores represent inliers.
        """
        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        # Average depth of leaf containing each sample
        path_lengths = np.mean([t.path_length_(X, check_input=False) for t in self.estimators_], axis=0)

        # Depths are normalized in the same fashion as in Isolation Forest
        if self.max_samples is not None:
            n = min(self.y_.size, self.max_samples)
        else:
            n = self.y_.size
        c = _h(n)

        scores = np.array([1 - 2 ** (-pl/c) for pl in path_lengths])

        if self.contamination == 'auto':
            offset_ = 0.5

        elif isinstance(self.contamination, float):
            assert self.contamination > 0.0
            assert self.contamination < 0.5
            # pass for now
            pass
        else:
            raise ValueError('contamination should be set either to \'auto\' or a float value between 0.0 and 0.5')

        return scores - offset_

    def predict_outliers(self, X, check_input=True):
        """Predict if a particular sample is an outlier or not.
            Paramteres
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.
            check_input : bool indicating if input should be checked or not.
            Returns
            -------
            is_inlier : array, shape (n_samples,) For each observation, tells whether or not (+1 or -1) it should be
            considered as an inlier according to the fitted model.

        """

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        decision_function_outliers = self.decision_function_outliers(X, check_input=False)

        return np.array([1 if x > 0.0 else -1 for x in decision_function_outliers])

    def apply(self, X, check_input=True):
        """Apply trees in the forest to X, return leaf indices."""

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([t.apply(X, check_input=False) for t in self.estimators_]).transpose()


def weighted_variance(split_index, y):
    """Calculate sum of left and right partition variances, weighted by their length."""

    assert len(y) > 1
    assert split_index >= 1
    assert split_index <= len(y) - 1

    left_partition, right_partition = y[:split_index], y[split_index:]
    left_proportion = len(left_partition) / len(y)

    return left_proportion * np.var(left_partition) + (1 - left_proportion) * np.var(right_partition)


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
                sampling_strategy : string, strategy used to sample data-points to draw splitting direction.
                    Possible values are 'discriminative' for sampling points with regression values far away from
                    each other, and 'random' for sampling randomly.

                Attributes
                ----------
                is_fitted_ : bool flag indicating whenever fit has been called
                X_ : data used for fitting the forest
                y_ : training data outputs
                _nodes_list : list of SimilarityTreeClassifier instances, that is tree nodes
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

    # List of all nodes in the tree, that is SimilarityTreeClassifier instances. Shared across all instances
    _nodes_list = []

    def __init__(self,
                 random_state=1,
                 n_directions=1,
                 sim_function=np.dot,
                 max_depth=None,
                 min_samples_split=2,
                 depth=1,
                 discriminative_sampling=True):
        self.random_state = random_state
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.depth = depth
        self.discriminative_sampling = discriminative_sampling

    def apply(self, X, check_input=False):
        """Returns the index of the leaf that each sample is predicted as."""

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([self.apply_x(x) for x in X])

    def apply_x(self, x, check_input=False):
        """Returns the index of the leaf that sample is predicted as."""

        if self._is_leaf:
            return self._node_id

        t = self._lhs if self.sim_function(x, self._q) - self.sim_function(x,
                                                                           self._p) <= self._split_point else self._rhs
        if t is None:
            return self._node_id
        return t.apply_x(x)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree."""

        if check_input:
            # Check is fit had been called
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)
            X = self._validate_X_predict(X, check_input)

        if self._is_leaf:
            return f'In the leaf node, containing samples: \n {list(zip(self.X_, self.y_))}'

        similarity = self.sim_function(X, self._q) - self.sim_function(X, self._p)
        left = similarity <= self._split_point
        t = self._lhs if left else self._rhs
        if t is None:
            return f'In the leaf node, containing samples: \n {list(zip(self.X_, self.y_))}'

        if left:
            print(
                f'Going left P: {self._p}, \t Q: {self._q}, \t split point: {self._split_point},'
                f'\t similarity: {similarity}')
        else:
            print(
                f'Going right P: {self._p}, \t Q: {self._q}, \t split point: {self._split_point},'
                f'\t similarity: {similarity}')

        return t.decision_path(X, check_input=False)

    def get_depth(self):
        """Returns the depth of the decision tree.
        The depth of a tree is the maximum distance between the root
        and any leaf.
        """

        check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

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
                first = random_state.choice(range(len(y)), replace=False)
                first_value = y[first]
                min_diff = np.std(y)
                different = np.where(np.abs(y - first_value) > min_diff)[0]
                if len(different) == 0:
                    first, second = random_state.choice(a=range(len(y)), size=2, replace=False)
                else:
                    second = random_state.choice(different, replace=False)

            else:
                first, second = random_state.choice(a=range(len(y)), size=2, replace=False)

            assert first is not None
            assert second is not None

            yield first, second

    def _find_split(self, X, y, p, q):
        """ Find split among direction drew on pair of data-points
            Parameters
            ----------
            X : all data-points
            y : output vector
            p : first data-point used for drawing direction of the split
            q : second data-point used for drawing direction of the split

            Returns
            -------
            best_impurity_decrease : decrease of variance after the split
            best_p : first data point from pair providing the best impurity decrease
            best_p : second data point from pair providing the best impurity decrease
            best_split_point : split threshold
        """
        similarities = [self.sim_function(x, q) - self.sim_function(x, p) for x in X]
        indices = sorted([i for i in range(len(y)) if not np.isnan(similarities[i])],
                         key=lambda x: similarities[x])

        y = y[indices]

        best_impurity = np.inf
        best_p = None
        best_q = None
        best_split_point = 0

        n = len(y)
        for i in range(n - 1):

            impurity = weighted_variance(i+1, y[indices])

            if impurity < best_impurity:
                best_impurity = impurity
                best_p = p
                best_q = q

                best_split_point = (similarities[indices[i]] + similarities[indices[i + 1]]) / 2

        return best_impurity, best_p, best_q, best_split_point, similarities

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

        # Check parameters
        random_state = check_random_state(self.random_state)

        if not isinstance(self.n_directions, int):
            raise ValueError('n_directions parameter must be an int')

        self.X_ = X
        self.y_ = y
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
        self._nodes_list.append(self)

        # Current node id is length of all nodes list. Nodes are numbered from 1, the root node
        self._node_id = len(self._nodes_list)

        # Value of predicion
        self._value = np.mean(self.y_)

        # Current node's impurity
        self._impurity = np.var(self.y_)

        if self.y_.size == 1:
            self._is_leaf = True
            self.is_fitted_ = True
            return self

        if self.max_depth is not None:
            if self.depth == self.max_depth:
                self._is_leaf = True
                self.is_fitted_ = True
                return self

        if len(self.y_) <= self.min_samples_split:
            self._is_leaf = True
            self.is_fitted_ = True
            return self

        # Sample n_direction discriminative directions and find the best one
        best_impurity = np.inf
        best_split_point = -np.inf
        best_p = None
        best_q = None
        similarities = []

        for i, j in self._sample_directions(random_state, self.y_, self.n_directions):

            impurity, p, q, split_point, curr_similarities = self._find_split(X, y, X[i], X[j])
            if impurity < best_impurity:
                best_impurity = impurity
                best_split_point = split_point
                best_p = p
                best_q = q
                similarities = curr_similarities

        if self._impurity - best_impurity > 0.0:
            self._split_point = best_split_point
            self._p = best_p
            self._q = best_q
            self._similarities = np.array(similarities)

            # Left- and right-hand side partitioning
            lhs_idxs = np.nonzero(self._similarities <= self._split_point)[0]
            rhs_idxs = np.nonzero(self._similarities > self._split_point)[0]

            if len(lhs_idxs) > 0 and len(rhs_idxs) > 0:
                self._lhs = SimilarityTreeRegressor(random_state=self.random_state,
                                                     n_directions=self.n_directions,
                                                     sim_function=self.sim_function,
                                                     max_depth=self.max_depth,
                                                     depth=self.depth + 1,
                                                     discriminative_sampling=self.discriminative_sampling).\
                    fit(self.X_[lhs_idxs], self.y_[lhs_idxs], check_input=False)

                self._rhs = SimilarityTreeRegressor(random_state=self.random_state,
                                                     n_directions=self.n_directions,
                                                     sim_function=self.sim_function,
                                                     max_depth=self.max_depth,
                                                     depth=self.depth + 1,
                                                     discriminative_sampling=self.discriminative_sampling).\
                    fit(self.X_[rhs_idxs], self.y_[rhs_idxs], check_input=False)
            else:
                self.is_fitted_ = True
                self._is_leaf = True
                return self

        # Split has not been found
        else:
            self.is_fitted_ = True
            self._is_leaf = True
            return self

        self.is_fitted_= True

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
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([self.predict_row_output(x) for x in X])

    def predict_row_output(self, x):
        """ Predict regression output of a single data-point.
            If current node is a leaf, return its prediction value,
            if not, traverse down the tree to find point's output

            Parameters
            ----------
            x : a data-point
            Returns
            -------
            data-point's output
        """

        if self._is_leaf:
            return self._value

        assert self._p is not None
        assert self._q is not None

        t = self._lhs if self.sim_function(x, self._q) - self.sim_function(x, self._p) <= self._split_point else self._rhs
        if t is None:
            return self._value
        return t.predict_row_output(x)

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
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([self.row_path_length_(x) for x in X])

    def row_path_length_(self, x):
        """ Get outlyingness path length of a single data-point.
            If current node is a leaf, return its depth,
            if not, traverse down the tree

            Parameters
            ----------
            x : a data-point
            Returns
            -------
            data-point's path length, according to single a tree.
        """

        if self._is_leaf:

            return self.depth

        assert self._p is not None
        assert self._q is not None

        t = self._lhs if self.sim_function(x, self._q) - self.sim_function(x,
                                                                           self._p) <= self._split_point else self._rhs
        if t is None:

            return self.depth

        return t.row_path_length_(x)

    @property
    def feature_importances_(self):
        """Return the feature importances."""
        pass


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
                random_state=1,
                n_estimators=20,
                n_directions=1,
                sim_function=np.dot,
                max_depth=None,
                oob_score=False,
                discriminative_sampling=True,
                bootstrap=True,
                max_samples=0.5):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.oob_score = oob_score
        self.discriminative_sampling = discriminative_sampling
        self.bootstrap = bootstrap
        self.max_samples = max_samples

    def apply(self, X, check_input=False):
        """Returns the index of the leaf that each sample is predicted as."""

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([t.apply(X, check_input=False) for t in self.estimators_]).transpose()

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

        y = np.atleast_1d(y)

        self.X_ = X
        self.y_ = y
        self.base_estimator_ = SimilarityTreeRegressor

        # Check input
        random_state = check_random_state(self.random_state)

        if not isinstance(self.n_directions, int):
            raise ValueError('n_directions parameter must be an int')

        self.oob_score_ = 0.0

        self.estimators_ = []
        for i in range(self.n_estimators):
            if self.bootstrap:
                all_idxs = range(y.size)
                idxs = random_state.choice(all_idxs, y.size, replace=True)

                tree = SimilarityTreeRegressor(n_directions=self.n_directions, sim_function=self.sim_function,
                                               random_state=self.random_state, max_depth=self.max_depth,
                                               discriminative_sampling=self.discriminative_sampling)
                tree.fit(X[idxs], y[idxs], check_input=False)

                self.estimators_.append(tree)

                if self.oob_score:
                    idxs_oob = np.setdiff1d(np.array(range(y.size)), idxs)
                    self.oob_score_ += tree.score(X[idxs_oob], y[idxs_oob])
            else:
                all_idxs = range(y.size)
                sample_size = int(self.max_samples * y.size)
                idxs = random_state.choice(all_idxs, sample_size, replace=False)

                tree = SimilarityTreeRegressor(n_directions=self.n_directions, sim_function=self.sim_function,
                                               random_state=self.random_state, max_depth=self.max_depth,
                                               discriminative_sampling=self.discriminative_sampling)
                tree.fit(X[idxs], y[idxs], check_input=False)

                self.estimators_.append(tree)

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
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.mean([t.predict(X) for t in self.estimators_], axis=0)

    def outlyingness(self, X, check_input=True):
        """Get outlyingness measure for X.
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input samples.
            check_input : bool indicating if input values should be checked or not.
            Returns
            -------
            outlyingness : ndarray, shape (n_samples,)
                The outlyingness measure, values are scaled to fit within range between 0 and 1.
        """

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        path_lengths = np.mean([t.path_length_(X, check_input=False) for t in self.estimators_], axis=0)
        n = X.size
        # Scaling factor is chosen as an average tree length in BST, in the same fashion as in Isolation Forest
        scaling_factor = 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)
        score = np.array([2 ** (-pl/scaling_factor) for pl in path_lengths])
        return score

    @property
    def feature_importances_(self):
        """Return the feature importances."""
        pass
