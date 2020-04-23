from matplotlib import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import pearsonr


def outliers_rank_stability(model, X, plot=True):
    """Check stability of outliers ranking with different number of subestimators.
        We check 1, 2, 3, 5, 10, 20, 30, 50, 70 and 100 trees respectively
        Paramaters
        ----------
            X : array, data to perform outlier detection
            plot : bool, flag indicating, if a chart with rank stability would be plotted
        Returns
        ---------
            rcorrelations : array of shape(number of estimators used for checing ranking stability, 2)
                First column represented Spearman correlation of ranking predicted with current number of trees,
                second column gives p-values
    """
    model = model(n_estimators=100).fit(X)
    initial_decision_function = model.decision_function(X, check_input=True, n_estimators=1)
    n_outliers = np.where(initial_decision_function <= 0)[0].size
    order = initial_decision_function[::-1].argsort()

    trees = [2, 3, 5, 10, 20, 30, 50, 70, 100]
    rcorrelations = np.zeros(shape=(9, 2), dtype=np.float)

    for idx, v in enumerate(trees):
        model = model(n_estimators=v).fit(X)
        decision_function = model.decision_function(X, check_input=False, n_estimators=v)
        rcorr, p = spearmanr(initial_decision_function[::-1][order][:n_outliers],
                             decision_function[::-1][order][:n_outliers])

        rcorrelations[idx] = rcorr, p
        initial_decision_function = decision_function

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        line = Line2D(trees, rcorrelations[:, 0])
        ax.add_line(line)
        ax.set_xlim(trees[0], trees[-1])
        ax.set_ylim(-0.2, 1.0)
        ax.set_xlabel('Number of estimators')
        ax.set_ylabel('Rcorrelation with previous value')

        plt.show()

    return rcorrelations


def plot_projection(s, p, q, split_point, y, sim_function, depth, criterion):
    """Plot projection of data points on the split direction
        Parameters
        ----------
            s : array of shape (n_samples,), values of similarity-values based projection
            p : first data-point used to draw splitting direction
            q : second data-point used to draw splitting direction
            split_point : split threshold
            y : array of shape (n_samples,), values for each data-point
            sim_function : a function used to perform the split
            depth : depth of current node
            criterion : criterion used to optimize split

    """

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
    p_similarity = sim_function(p, q) - sim_function(p, p)
    ax.axvline(p_similarity, c='green')
    plt.text(p_similarity, np.max(y), 'p', c='green')

    q_similarity = sim_function(q, q) - sim_function(q, p)
    ax.axvline(q_similarity, c='green')
    plt.text(q_similarity, np.max(y), 'q', c='green')

    # split point
    ax.axvline(split_point, c='green')
    plt.text(split_point, np.min(y), 'split point', c='green', rotation=90)

    # titles
    plt.title(f'Split at depth {depth}, criterion: {criterion}')
    plt.xlabel('Similarity')
    plt.ylabel('y')
    plt.show()


def plot_model_selection(model, parameter, parameter_range, X, y):
    # grid search
    param = {parameter: parameter_range}
    search = GridSearchCV(model, param_grid=param, cv=5, return_train_score=True)
    search = search.fit(X, y)

    # extract scores
    train_scores = search.cv_results_['mean_train_score']
    val_scores = search.cv_results_['mean_test_score']

    # plot
    train_curve, = plt.plot(parameter_range, train_scores, marker='o', label='train')
    validation_curve, = plt.plot(parameter_range, val_scores, marker='o', label='cross-validation')
    plt.legend(handles=[train_curve, validation_curve])
    plt.title('Validation curve')
    plt.xlabel(parameter)
    plt.ylabel('score')
    plt.show()

    # return grid-search results
    return pd.DataFrame(search.cv_results_)


def plot_confusion_matrix(model, X_test, y_test, classes, cmap='Purples'):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xticks(rotation=90)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()


def create_correlated_feature(y, a=10, b=5, fraction=0.2, seed=None, verbose=False):
    """
    Create synthetic column, strongly correlated with target.
    Each value is calculated according to the formula:
        v = y * a + random(-b, b)
        So its scaled target value with some noise.
    Then a fraction of values is permuted, to reduce the correlation.

    Parameters
    ---------
        y : np.ndarray, target vector
        a : int or float (default=10), scaling factor in a formula above
        b : int or float (default=5), value that determines the range of noise to be added
        fraction : float (default=0.2), fraction of values to be permuted to reduce the correlation
        seed : int (default=None), random seed that can be specified to obtain deterministic behaviour
        verbose : bool (default=False), when True, print correlation before and after the shuffling

    Returns
    ----------
        new_column : np.ndarray, new feature vector
        corr : float, correlation of new feature vector with target vector
        p : float, p value of correlation
    """
    if seed is not None:
        np.seed(seed)

    new_column = y * a + np.random.uniform(low=-b, high=b, size=len(y))
    if verbose:
        print(f'Initial new feature correlation, without shuffling: {pearsonr(new_column, y)}')

    # Choose which samples to permute
    indices = np.random.choice(range(len(y)), int(fraction * len(y)))

    # Find new order of this samples
    shuffled_indices = np.random.permutation(len(indices))
    new_column[indices] = new_column[indices][shuffled_indices]
    corr, p = pearsonr(new_column, y)
    if verbose:
        print(f'New feature correlation, after shuffling {fraction} of samples: {pearsonr(new_column, y)}')

    return new_column, corr, p

