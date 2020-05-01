from simforest import SimilarityForestClassifier, SimilarityForestRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from scipy.stats import pointbiserialr, spearmanr
import tqdm


def create_numerical_feature_classification(y, a=10, b=5, fraction=0.2, seed=None, verbose=False):
    """
    Create synthetic numerical column, strongly correlated with binary classification target.
    Each value is calculated according to the formula:
        v = y * a + random(-b, b)
        So its scaled target value with some noise.
    Then a fraction of values is permuted, to reduce the correlation.

    Point biserial correlation is used to measure association.
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
    """
    if seed is not None:
        np.random.seed(seed)

    new_column = y * a + np.random.uniform(low=-b, high=b, size=len(y))
    if verbose:
        corr, v = pointbiserialr(new_column, y)
        print(f'Initial new feature - target point biserial correlation, without shuffling: {round(corr, 3)}, p: {round(v, 3)}')

    # Choose which samples to permute
    indices = np.random.choice(range(len(y)), int(fraction * len(y)), replace=False)

    # Find new order of this samples
    shuffled_indices = np.random.permutation(len(indices))
    new_column[indices] = new_column[indices][shuffled_indices]
    corr, p = pointbiserialr(new_column, y)
    if verbose:
        print(f'New feature - target point biserial correlation, after shuffling: {round(corr, 3)}, p: {round(v, 3)}')

    return new_column, corr


def create_categorical_feature_classification(y, fraction=0.2, seed=None, verbose=False):
    """
    Create synthetic categorical binary column, strongly correlated with binary classification target.
    New column is a copy of target, with a `fraction` of samples shuffled to reduce the correlation.

    Phi coefficient is used to measure association.
    Parameters
    ---------
        y : np.ndarray, target vector
        fraction : float (default=0.2), fraction of values to be permuted to reduce the correlation
        seed : int (default=None), random seed that can be specified to obtain deterministic behaviour
        verbose : bool (default=False), when True, print correlation before and after the shuffling

    Returns
    ----------
        new_column : np.ndarray, new feature vector
        corr : float, correlation of new feature vector with target vector
    """
    if seed is not None:
        np.random.seed(seed)

    new_column = y.copy()
    if verbose:
        corr = matthews_corrcoef(new_column, y)
        print(f'Initial new feature - target point Phi coefficient, without shuffling: {round(corr, 3)}')

    # Choose which samples to permute
    indices = np.random.choice(range(len(y)), int(fraction * len(y)), replace=False)

    # Find new order of this samples
    shuffled_indices = np.random.permutation(len(indices))
    new_column[indices] = new_column[indices][shuffled_indices]

    corr = matthews_corrcoef(new_column, y)
    if verbose:
        print(f'New feature - target point Phi coefficient, after shuffling: {round(corr, 3)}')

    return new_column, corr


def create_numerical_feature_regression(y, fraction=0.2, seed=None, verbose=False):
    """
    Create synthetic numerical column, strongly correlated with regression target.
    Each value is calculated according to the formula:
        v = y * a + random(-b, b)
        Where:
            a: 10
            b: one standard deviation of target vector
        So its scaled target value with some noise.
    Then a fraction of values is permuted, to reduce the correlation.

    Spearman rank correlation is used to measure association.
    Parameters
    ---------
        y : np.ndarray, target vector
        fraction : float (default=0.2), fraction of values to be permuted to reduce the correlation
        seed : int (default=None), random seed that can be specified to obtain deterministic behaviour
        verbose : bool (default=False), when True, print correlation before and after the shuffling

    Returns
    ----------
        new_column : np.ndarray, new feature vector
        corr : float, correlation of new feature vector with target vector
    """
    if seed is not None:
        np.random.seed(seed)

    a = 10
    b = np.std(y)
    new_column = y * a + np.random.uniform(low=-b, high=b, size=len(y))

    if verbose:
        corr, v = spearmanr(new_column, y)
        print(f'Initial new feature - target Spearman correlation, without shuffling: {round(corr, 3)}, p: {round(v, 3)}')

    # Choose which samples to permute
    indices = np.random.choice(range(len(y)), int(fraction * len(y)), replace=False)

    # Find new order of this samples
    shuffled_indices = np.random.permutation(len(indices))
    new_column[indices] = new_column[indices][shuffled_indices]
    corr, p = spearmanr(new_column, y)
    if verbose:
        print(f'New feature - target Spearman correlation, after shuffling: {round(corr, 3)}, p: {round(v, 3)}')

    return new_column, corr


def importance(model, X, y):
    """
    Measure permutation importance of features in a dataset, according to a given model.
    Returns
    -------
    dictionary with permutation importances
    index of features, from most to least important
    """
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=4)
    sorted_idx = result.importances_mean.argsort()

    return result, sorted_idx


def get_permutation_importances(rf, sf, X_train, y_train, X_test, y_test, corr=None, labels=None, plot=False):
    """
    Measure permutation features importances according to two models, on both train and test set
    :param rf: first model, already fitted
    :param sf: second model, already fitted
    :param X_train: training dataset
    :param y_train: training labels
    :param X_test: test dataset
    :param y_test: test labels
    :param corr: correlation of new feature with target, used only for plot's legend
    :param labels: name of features in the datasets, used only for plot's legend
    :param plot: bool, whenever to plot the feature importances boxplots or not

    :return:
    dictionary with four values, each with  new feature importances according to a model, on certain dataset
    """

    # Get feature importances for both training and test set
    rf_train_result, rf_train_sorted_idx = importance(rf, X_train, y_train)
    rf_test_result, rf_test_sorted_idx = importance(rf, X_test, y_test)
    sf_train_result, sf_train_sorted_idx = importance(sf, X_train, y_train)
    sf_test_result, sf_test_sorted_idx = importance(sf, X_test, y_test)

    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(14, 8))
        ax[0, 0].set_xlim(-0.05, 0.5)
        ax[0, 0].boxplot(rf_train_result.importances[rf_train_sorted_idx].T,
                         vert=False, labels=labels[rf_train_sorted_idx])
        ax[0, 0].set_title('Random Forest, train set')
        ax[0, 1].set_xlim(-0.05, 0.5)
        ax[0, 1].boxplot(rf_test_result.importances[rf_test_sorted_idx].T,
                         vert=False, labels=labels[rf_test_sorted_idx])
        ax[0, 1].set_title('Random Forest, test set')

        ax[1, 0].set_xlim(-0.05, 0.5)
        ax[1, 0].boxplot(sf_train_result.importances[sf_train_sorted_idx].T,
                         vert=False, labels=labels[sf_train_sorted_idx])
        ax[1, 0].set_title('Similarity Forest, train set')
        ax[1, 1].set_xlim(-0.05, 0.5)
        ax[1, 1].boxplot(sf_test_result.importances[sf_test_sorted_idx].T,
                         vert=False, labels=labels[sf_test_sorted_idx])
        ax[1, 1].set_title('Similarity Forest, test set')
        plt.suptitle(f'Feature importances, correlation: {round(corr, 3)}', fontsize=16)
        plt.show()

    # Return importance of new feature (it's first in the list)
    results = {'rf_train': rf_train_result['importances_mean'][0],
               'rf_test': rf_test_result['importances_mean'][0],
               'sf_train': sf_train_result['importances_mean'][0],
               'sf_test': sf_test_result['importances_mean'][0]}

    return results


def score_model(model, X_train, y_train, X_test, y_test):
    """
    Fit the model on train set and score it on test set.
    For classification, use f1 score, for regression use r2 score.
    Handy function to avoid some duplicated code.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classifier(model):
        score = f1_score(y_test, y_pred)
    else:
        score = r2_score(y_test, y_pred)
    return model, score


def bias_experiment(df, y, task, column_type, fraction_range, SEED=None):
    """
    Conduct a experiment, measuring how Random Forest and Similarity Forest compare,
    if a biased column is added to a dataset.

    At each step of this simulation, a new feature is generated using create_correlated_feature function.
    A fraction of this feature values gets shuffled to reduce the correlation.
    During whole experiment, a new features varies from very correlated (biased) feature to completely random.
    Random Forest and Similarity Forest scores and permutation feature importances are measured,
    to asses, how both models are robust to bias present in the dataset.


    :param df: pandas DataFrame with the dataset
    :param y: vector with labels
    :param task: string, `classification` or `regression`
    :param column_type: string, `numerical` or `categorical`
    :param fraction_range:
    :param SEED: random number generator seed
    :return:
    """
    # Function used to create synthetic feature
    create_feature = None

    if task == 'classification':
        RandomForest = RandomForestClassifier
        SimilarityForest = SimilarityForestClassifier
        if column_type == 'numerical':
            create_feature = create_numerical_feature_classification
        elif column_type == 'categorical':
            create_feature = create_categorical_feature_classification
        else:
            raise ValueError(f'column_type should be either `numerical` or `categorical`, found: {column_type}')

    elif task == 'regression':
        RandomForest = RandomForestRegressor
        SimilarityForest = SimilarityForestRegressor
        if column_type == 'numerical':
            create_feature = create_numerical_feature_regression
        elif column_type == 'categorical':
            raise NotImplementedError
        else:
            raise ValueError(f'column_type should be either `numerical` or `categorical`, found: {column_type}')
    else:
        raise ValueError(f'task should be either `classification` or `regression`, found: {column_type}')

    correlations = np.zeros(len(fraction_range), dtype=np.float32)
    rf_scores = np.zeros(len(fraction_range), dtype=np.float32)
    sf_scores = np.zeros(len(fraction_range), dtype=np.float32)
    permutation_importances = []

    for i, f in tqdm.tqdm(enumerate(fraction_range)):
        # Pop old values
        if 'new_feature' in df.columns:
            df.pop('new_feature')

        # Add new
        new_feature, correlations[i] = create_feature(y, fraction=f, seed=SEED)
        df = pd.concat([pd.Series(new_feature, name='new_feature'), df], axis=1)

        # Split the data with random seed
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.3, random_state=SEED)

        # Preprocess
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Score
        rf, rf_scores[i] = score_model(RandomForest(random_state=SEED),
                                       X_train, y_train, X_test, y_test)

        sf, sf_scores[i] = score_model(SimilarityForest(n_estimators=100, random_state=SEED),
                                       X_train, y_train, X_test, y_test)

        # Measure features importances
        permutation_importances.append(get_permutation_importances(rf, sf, X_train, y_train, X_test, y_test))

    return correlations, rf_scores, sf_scores, permutation_importances


def tick_function(correlations):
    return [round(c, 2) for c in correlations]


def plot_bias(fraction_range, correlations, rf_scores, sf_scores, permutation_importances, dataset_name):
    # Axis for scores
    # Set figure and first axis
    fig = plt.figure(figsize=(14, 16))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.xticks(rotation=90)
    ax1.set_xticks(fraction_range)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xlabel('Fraction of shuffled instances')

    # Set second axis
    ax2 = ax1.twiny()
    plt.xticks(rotation=90)
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xticklabels(tick_function(correlations))
    ax2.set_xlabel('New feature correlation')

    # Plot scores
    plt.plot(fraction_range, rf_scores, label='Random Forest')
    plt.plot(fraction_range, sf_scores, label='Similarity Forest')

    # Set legend and titles
    plt.legend()
    ax1.set_ylabel('Score')
    plt.title(f'Scores, {dataset_name} dataset', fontsize=16)

    # Axis for importances
    df_permutation_importances = pd.DataFrame(permutation_importances)

    # Set figure and first axis
    ax3 = fig.add_subplot(2, 1, 2)
    plt.xticks(rotation=90)
    ax3.set_xticks(fraction_range)
    ax3.set_xlim(0.0, 1.0)
    ax3.set_xlabel('Fraction of shuffled instances')

    # Set second axis
    ax4 = ax3.twiny()
    plt.xticks(rotation=90)
    ax4.set_xticks(ax3.get_xticks())
    ax4.set_xlim(0.0, 1.0)
    ax4.set_xticklabels(tick_function(correlations))
    ax4.set_xlabel('New feature correlation')

    # Plot importances
    plt.plot(fraction_range, df_permutation_importances['rf_train'].values, label='Random Forest, train')
    plt.plot(fraction_range, df_permutation_importances['rf_test'].values, label='Random Forest, test')
    plt.plot(fraction_range, df_permutation_importances['sf_train'].values, label='Similarity Forest, train')
    plt.plot(fraction_range, df_permutation_importances['sf_test'].values, label='Similarity Forest, test')

    # Set legend and titles
    plt.legend()
    ax3.set_ylabel('New feature importance')
    plt.title(f'Permutation importance, {dataset_name} dataset', fontsize=16)
    plt.tight_layout()
    plt.show()

