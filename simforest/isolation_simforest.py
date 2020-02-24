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
                 max_depth=None,
                 most_different=False,
                 max_samples=None,
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

            if y is not None:
                # Check that X and y have correct shape
                X, y = check_X_y(X, y)

            # Check if provided similarity function applies to input
            X = self._validate_X_predict(X)

        y = np.array(y)

        # Random state
        if self.random_state is not None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = np.random.RandomState()

        self.estimators_ = []
        for i in range(self.n_estimators):
            all_idxs = range(y.size)

            # Calculate sample size for each tree
            if self.max_samples == 'auto':
                sample_size = min(256, y.size)
            elif isinstance(self.max_samples, float):
                n = len(y)
                sample_size = int(self.max_samples * n)
                if sample_size > n:
                    sample_size = n

            elif isinstance(self.max_samples, int):
                sample_size = self.max_samples
                n = len(y)
                if sample_size > n:
                    sample_size = n
            else:
                raise ValueError('max_samples should be \'auto\' or either float or int')

            idxs = random_state.choice(all_idxs, sample_size, replace=False)

            tree = SimilarityIsolationTree(sim_function=self.sim_function,
                                           random_state=self.random_state,
                                           max_depth=self.max_depth,
                                           most_different=self.most_different)
            tree.fit(X[idxs], y[idxs], check_input=False)

            self.estimators_.append(tree)
            assert len(self.estimators_) == self.n_estimators
            self.is_fitted_ = True

            return self

    def decision_function(self, X):
        pass

    def predict(self, X):
        pass


class SimilarityIsolationTree:

    def __init__(self,
                 sim_function=dot_product,
                 random_state=None,
                 max_depth=8,
                 most_different=False):
        self.sim_function = sim_function
        self.random_state = random_state
        self.max_depth = max_depth
        self.most_different = most_different

    def fit(self, X, y=None, check_input=False):
        pass

    def path_length_(self, X, check_input=True):
        pass
