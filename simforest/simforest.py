"""
This is a module implementing Similarity Forest Classifier,
as outlined in 'Similarity Forests', S. Sathe and C. C. Aggarwal, KDD 2017' ,
avaiable here: http://saketsathe.net/downloads/simforest.pdf
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.ensemble.forest import ForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from scipy.spatial import distance


def calc_gini(total_left, total_right, true_left, true_right):
    left_pred = true_left / total_left
    right_pred = true_right / total_right

    left_gini = 1 - left_pred ** 2 - (1 - left_pred) ** 2
    right_gini = 1 - right_pred ** 2 - (1 - right_pred) ** 2

    left_prop = total_left / (total_left + total_right)
    return left_prop * left_gini + (1 - left_prop) * right_gini


def gini_index(split_index, y):
    left_partition, right_partition = y[:split_index], y[split_index:]

    left_gini = 1.0 - np.sum([(np.where(left_partition == cl)[0].size / len(left_partition)) ** 2 for cl in np.unique(y)])
    right_gini = 1.0 - np.sum([(np.where(right_partition == cl)[0].size / len(right_partition)) ** 2 for cl in np.unique(y)])

    left_prop = len(left_partition) / len(y)
    return left_prop * left_gini + (1.0 - left_prop) * right_gini


class SimilarityTreeClassifier(BaseEstimator, ClassifierMixin):
    """Similarity Tree classifier implementation.
        Similarity Trees are base models used as building blocks for Similarity Forest ensemble.
            Parameters
            ----------
            random_state : int, random numbers generator seed
            n_directions: int, number of discriminative directions to check at each split.
                               The best direction is chosen, based on child nodes' purity.
            sim_function: function used to measure similarity between data-points

            Attributes
            ----------

    """

    # List of all nodes in the tree, that is SimilarityTreeClassifier instances. Shared across all instances
    _nodes_list = []

    def __init__(self,
                 random_state=1,
                 n_directions=1,
                 sim_function=distance.euclidean,
                 classes=None,
                 max_depth=None,
                 depth=1):
        self.random_state = random_state
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.classes = classes
        self.max_depth = max_depth
        self.depth = depth

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
        """Returns the number of leaves of the decision tree.
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
            first = random_state.choice(range(len(labels)), replace=False)
            first_class = labels[first]
            others = np.where(labels != first_class)[0]
            second = random_state.choice(others, replace=False)

            yield first, second

        '''first = np.where(labels == 1)[0]
        other = np.where(labels == 0)[0]

        for _ in range(n_directions):
            yield random_state.choice(first, replace=False), random_state.choice(other, replace=False)'''

    def _find_split(self, X, y, p, q):
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
        similarities = [self.sim_function(x, q) - self.sim_function(x, p) for x in X]
        indices = sorted([i for i in range(len(y)) if not np.isnan(similarities[i])],
                         key=lambda x: similarities[x])

        y = y[indices]

        best_impurity = 1.0
        best_p = None
        best_q = None
        best_split_point = 0

        n = len(y)
        #total_true = np.sum(y)
        #total_true = sum([y[j] for j in indices])
        #left_true = 0
        for i in range(n - 1):
            '''
            left_true += y[indices[i]]
            right_true = total_true - left_true
            impurity2 = calc_gini(i + 1, n - i - 1, left_true, right_true)
            '''

            impurity = gini_index(i+1, y[indices])

            if impurity < best_impurity:
                best_impurity = impurity
                best_p = p
                best_q = q

                best_split_point = (similarities[indices[i]] + similarities[indices[i + 1]]) / 2

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

        '''
        if self.n_classes_ > 2:
            raise Exception('Similarity Tree is a binary classifier!')
        '''

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

            impurity, p, q, split_point, curr_similarities = self._find_split(X, y, X[i], X[j])
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
                                                     depth=self.depth+1).fit(self.X_[lhs_idxs], self.y_[lhs_idxs],
                                                                             check_input=False)

                self._rhs = SimilarityTreeClassifier(random_state=self.random_state,
                                                     n_directions=self.n_directions,
                                                     sim_function=self.sim_function,
                                                     classes=self.classes_,
                                                     max_depth=self.max_depth,
                                                     depth=self.depth+1).fit(self.X_[rhs_idxs], self.y_[rhs_idxs],
                                                                             check_input=False)
            else:
                return

        self.is_fitted_= True

        return self

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba.
            In case of Similarity Tree, check if similarity function provided applies to input.
            Check result of applying similarity function to two first data-points. """

        if X.shape[0] > 1:
            res = self.sim_function(X[0], X[1])
            if not isinstance(res, (int, float)):
                raise ValueError('Provided similarity function does not apply to input.')

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
        """ Predict class probability of input array.

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
            X : array-like or sparse matrix of shape = [n_samples, n_features]
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
    def __init__(self,
                 random_state=1,
                 n_trees=20,
                 n_directions=1,
                 sim_function=np.dot,
                 max_depth=None,
                 oob_score=False):
        self.random_state = random_state
        self.n_trees = n_trees
        self.n_directions = n_directions
        self.sim_function = sim_function
        self.max_depth = max_depth
        self.oob_score = oob_score

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba.
            In case of Similarity Forest, check if similarity function provided applies to input.
            Check result of applying similarity function to two first data-points. """

        res = self.sim_function(X[0, :], X[1, :])
        if not isinstance(res, (int, float)):
            raise ValueError('Provided similarity function does not apply to input.')

        return X

    def fit(self, X, y, check_input=True):
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

        self.classes = unique_labels(y)
        self.n_classes_ = self.classes.shape[0]

        '''
        if self.n_classes_ > 2:
            raise Exception('Similarity Tree is a binary classifier!')
        '''

        # Check input
        random_state = check_random_state(self.random_state)

        if not isinstance(self.n_directions, int):
            raise ValueError('n_directions parameter must be an int')

        self.X_ = X
        self.y_ = y

        self.oob_score_ = 0.0

        self.classes = np.unique(y)
        self.trees = []
        for i in range(self.n_trees):
            idxs = random_state.choice(range(y.size), y.size, replace=True)

            tree = SimilarityTreeClassifier(classes=self.classes, n_directions=self.n_directions,
                                            random_state=self.random_state, max_depth=self.max_depth)
            tree.fit(X[idxs], y[idxs], check_input=False)

            self.trees.append(tree)

            if self.oob_score:
                idxs_oob = np.setdiff1d(np.array(range(y.size)), idxs)

                self.oob_score_ += tree.score(X[idxs_oob], y[idxs_oob])

        if self.oob_score:
            self.oob_score_ /= self.n_trees

        assert len(self.trees) == self.n_trees
        self.is_fitted_ = True

        return self

    def predict_proba(self, X, check_input=True):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

        # Input validation
        X = check_array(X)

        # Check if provided similarity function applies to input
        X = self._validate_X_predict(X, check_input)

        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.
            Parameters
            ----------
            X : array-like or sparse matrix of shape = [n_samples, n_features]
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

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

        # Input validation
        X = check_array(X)

        # Check if provided similarity function applies to input
        X = self._validate_X_predict(X, check_input)

        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
        #return np.array(self.predict_proba(X) >= 0.5).astype(np.int)

    def apply(self, X, check_input=True):
        """Apply trees in the forest to X, return leaf indices."""

        if check_input:
            # Check if fit had been called
            check_is_fitted(self, ['X_', 'y_', 'is_fitted_'])

            # Input validation
            X = check_array(X)

            X = self._validate_X_predict(X, check_input)

        return np.array([t.apply(X, check_input=False) for t in self.trees]).transpose()

'''
class SimilarityForestClassifier(ForestClassifier):
    """A similarity forest classifier.
        A similarity forest is a meta estimator that fits a number of similarity tree
        classifiers on various sub-samples of the dataset and uses averaging to
        improve the predictive accuracy and control over-fitting.
        The sub-sample size is always the same as the original
        input sample size but the samples are drawn with replacement if
        `bootstrap=True` (default).
        Read more in the :ref:`User Guide <forest>`.
        Parameters
        ----------
        n_estimators : integer, optional (default=100)
            The number of trees in the forest.
            .. versionchanged:: 0.22
               The default value of ``n_estimators`` changed from 10 to 100
               in 0.22.
        criterion : string, optional (default="gini")
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity and "entropy" for the information gain.
            Note: this parameter is tree-specific.
        max_depth : integer or None, optional (default=None)
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.
        min_samples_split : int, float, optional (default=2)
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.
            .. versionchanged:: 0.18
               Added float values for fractions.
        min_samples_leaf : int, float, optional (default=1)
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
              `ceil(min_samples_leaf * n_samples)` are the minimum
              number of samples for each node.
            .. versionchanged:: 0.18
               Added float values for fractions.
        min_weight_fraction_leaf : float, optional (default=0.)
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.
        max_features :
        max_leaf_nodes : int or None, optional (default=None)
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.
        min_impurity_decrease : float, optional (default=0.)
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.
            The weighted impurity decrease equation is the following::
                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)
            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.
            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.
            .. versionadded:: 0.19
        min_impurity_split : float, (default=1e-7)
            Threshold for early stopping in tree growth. A node will split
            if its impurity is above the threshold, otherwise it is a leaf.
            .. deprecated:: 0.19
               ``min_impurity_split`` has been deprecated in favor of
               ``min_impurity_decrease`` in 0.19. The default value of
               ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
               will be removed in 0.25. Use ``min_impurity_decrease`` instead.
        bootstrap : boolean, optional (default=True)
            Whether bootstrap samples are used when building trees. If False, the
            whole datset is used to build each tree.
        oob_score : bool (default=False)
            Whether to use out-of-bag samples to estimate
            the generalization accuracy.
        n_jobs : int or None, optional (default=None)
            The number of jobs to run in parallel.
            `fit`, `predict`, `decision_path` and `apply` are all
            parallelized over the trees.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        verbose : int, optional (default=0)
            Controls the verbosity when fitting and predicting.
        warm_start : bool, optional (default=False)
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit a whole
            new forest. See :term:`the Glossary <warm_start>`.
        class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
        None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.
            Note that for multioutput (including multilabel) weights should be
            defined for each class of every column in its own dict. For example,
            for four-class multilabel classification weights should be
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
            [{1:1}, {2:5}, {3:1}, {4:1}].
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``
            The "balanced_subsample" mode is the same as "balanced" except that
            weights are computed based on the bootstrap sample for every tree
            grown.
            For multi-output, the weights of each column of y will be multiplied.
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.
        ccp_alpha : non-negative float, optional (default=0.0)
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.
            .. versionadded:: 0.22

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
        n_features_ : int
            The number of features when ``fit`` is performed.
        n_outputs_ : int
            The number of outputs when ``fit`` is performed.
        feature_importances_ :
        oob_score_ : float
            Score of the training dataset obtained using an out-of-bag estimate.
        oob_decision_function_ : array of shape = [n_samples, n_classes]
            Decision function computed with out-of-bag estimate on the training
            set. If n_estimators is small it might be possible that a data point
            was never left out during the bootstrap. In this case,
            `oob_decision_function_` might contain NaN.

        Notes
        -----
        The default values for the parameters controlling the size of the trees
        (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
        unpruned trees which can potentially be very large on some data sets. To
        reduce memory consumption, the complexity and size of the trees should be
        controlled by setting those parameter values.
        To obtain a deterministic behaviour during
        fitting, ``random_state`` has to be fixed.
    """
    def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0):
        super().__init__(
            base_estimator=SimilarityTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=[],
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
'''
