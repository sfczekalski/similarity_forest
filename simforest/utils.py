import matplotlib.pyplot as plt
from matplotlib import collections
import matplotlib as mpl
import numpy as np


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
