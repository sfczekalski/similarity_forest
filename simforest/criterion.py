from typing import List
from types import FunctionType
import numpy as np


def gini_index(split_index, y):
    left_partition, right_partition = y[:split_index], y[split_index:]

    left_gini = 1.0 - np.sum([(np.where(left_partition == cl)[0].size / len(left_partition)) ** 2 for cl in np.unique(y)])
    right_gini = 1.0 - np.sum([(np.where(right_partition == cl)[0].size / len(right_partition)) ** 2 for cl in np.unique(y)])

    left_prop = len(left_partition) / len(y)
    return left_prop * left_gini + (1.0 - left_prop) * right_gini


def weighted_variance(split_index, y):
    """Calculate sum of left and right partition variances, weighted by their length."""

    assert len(y) > 1
    assert split_index >= 1
    assert split_index <= len(y) - 1

    left_partition, right_partition = y[:split_index], y[split_index:]
    left_proportion = len(left_partition) / len(y)

    return left_proportion * np.var(left_partition) + (1 - left_proportion) * np.var(right_partition)


def evaluate_split(X: List[float], eval_function: FunctionType, eval_objective: str = 'min',
                   min_group_size: int = 1) -> (int, float):
    """Finds optimal splitting point of X into two disjoint segments
       given the evaluation function and objective direction

        Parameters
        ----------
            X: list of values
            eval_function: function used to evaluate the purity of each segment
            eval_objective: either 'min' or 'max'
            min_group_size: minimum size of the segment
        Returns
        ----------
        tuple (index of the optimal splitting point, impurity)
    """
    assert eval_objective in ['min', 'max'], "eval_objective must be either 'min' or 'max' "
    assert min_group_size >= 1, "min_group_size must be equal or greater than 1"
    assert len(X) >= 2 * min_group_size, "len(X) must be at least twice as big as min_group_size"

    _fun = lambda r, s: (len(r) / len(X)) * eval_function(r) + (len(s) / len(X)) * eval_function(s)

    evals = np.array([_fun(X[:i], X[i:]) for i in range(min_group_size, len(X) - min_group_size + 1)])

    if eval_objective == 'min':
        i = np.argmin(evals)
        return min_group_size + i, evals[i]
    elif eval_objective == 'max':
        i = np.argmax(evals)
        return min_group_size + i, evals[i]
