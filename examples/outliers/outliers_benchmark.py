import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import ttest_ind
import neptune
from sklearn.ensemble import IsolationForest
from simforest.outliers.isolation_simforest import IsolationSimilarityForest
from examples.outliers.outliers_datasets import get_datasets


neptune.set_project('sfczekalski/SimilarityForest')

neptune.init('sfczekalski/SimilarityForest')

# set parameters
params = dict()
params['most_different'] = True
params['n_estimators'] = 100


# set experiment properties
n_iterations = 20

# create experiment
neptune.create_experiment(name='Outlier detection',
                          params=params,
                          properties={'n_iterations': n_iterations})

# init log
df = pd.DataFrame(columns=['dataset',
                           'IF roc-auc', 'SF roc-auc',
                           'IF precision', 'SF precision',
                           'IF recall', 'SF recall',
                           'IF f1', 'SF f1',
                           'IF roc-auc std', 'SF roc-auc std',
                           'IF precision std', 'SF precision std',
                           'IF recall std', 'SF recall std',
                           'IF f1 std', 'SF f1 std',
                           'p-val roc-auc', 'p-val precision', 'p-val recall', 'p-val f1'])

log_name = 'logs/outlier_detectionmost_different_more_metrics_log.csv'


# load and prepare data
for d_idx, d in enumerate(get_datasets()):
    X_train, X_test, y_train, y_test, dataset = d
    y_train = y_train.ravel().astype(np.int32)
    y_test = y_test.ravel().astype(np.int32)

    # store auc-roc for t-test
    if_auc = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_auc = np.zeros(shape=(n_iterations,), dtype=np.float32)
    if_precison = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_precision = np.zeros(shape=(n_iterations,), dtype=np.float32)
    if_recall = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_recall = np.zeros(shape=(n_iterations,), dtype=np.float32)
    if_f1 = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_f1 = np.zeros(shape=(n_iterations,), dtype=np.float32)

    if_auc_std = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_auc_std = np.zeros(shape=(n_iterations,), dtype=np.float32)
    if_precison_std = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_precision_std = np.zeros(shape=(n_iterations,), dtype=np.float32)
    if_recall_std = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_recall_std = np.zeros(shape=(n_iterations,), dtype=np.float32)
    if_f1_std = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_f1_std = np.zeros(shape=(n_iterations,), dtype=np.float32)

    # run
    for i in range(n_iterations):
        print(f'{dataset}, {i+1} / {n_iterations}')
        IF = IsolationForest()
        IF.fit(X_train, y_train)
        if_pred = IF.decision_function(X_test)
        if_auc[i] = roc_auc_score(y_test, if_pred)

        if_class_pred = np.ones_like(if_pred)
        if_class_pred[if_pred <= 0.0] = -1
        if_precison[i] = precision_score(y_test, if_class_pred)
        if_recall[i] = recall_score(y_test, if_class_pred)
        if_f1[i] = f1_score(y_test, if_class_pred)

        sf = IsolationSimilarityForest(**params)
        sf.fit(X_train, y_train)
        sf_pred = sf.decision_function(X_test)
        sf_auc[i] = roc_auc_score(y_test, sf_pred)

        sf_class_pred = np.ones_like(sf_pred)
        sf_class_pred[sf_pred <= 0.0] = -1
        sf_precision[i] = precision_score(y_test, sf_class_pred)
        sf_recall[i] = recall_score(y_test, sf_class_pred)
        sf_f1[i] = f1_score(y_test, sf_class_pred)

    # log results
    neptune.log_metric(f'{dataset} IF ROC-AUC', np.mean(if_auc))
    neptune.log_metric(f'{dataset} SF ROC-AUC', np.mean(sf_auc))
    neptune.log_metric(f'{dataset} IF precision', np.mean(if_precison))
    neptune.log_metric(f'{dataset} SF precision', np.mean(sf_precision))
    neptune.log_metric(f'{dataset} IF recall', np.mean(if_recall))
    neptune.log_metric(f'{dataset} SF recall', np.mean(sf_recall))
    neptune.log_metric(f'{dataset} IF f1', np.mean(if_f1))
    neptune.log_metric(f'{dataset} SF f1', np.mean(sf_f1))

    neptune.log_metric(f'{dataset} IF ROC-AUC std', np.std(if_auc))
    neptune.log_metric(f'{dataset} SF ROC-AUC std', np.std(sf_auc))
    neptune.log_metric(f'{dataset} IF precision std', np.std(if_precison))
    neptune.log_metric(f'{dataset} SF precision std', np.std(sf_precision))
    neptune.log_metric(f'{dataset} IF recall std', np.std(if_recall))
    neptune.log_metric(f'{dataset} SF recall std', np.std(sf_recall))
    neptune.log_metric(f'{dataset} IF f1 std', np.std(if_f1))
    neptune.log_metric(f'{dataset} SF f1 std', np.std(sf_f1))

    t_rocauc, p_rocauc = ttest_ind(if_auc, sf_auc)
    neptune.log_metric(f'{dataset} t-stat roc-auc', t_rocauc)
    neptune.log_metric(f'{dataset} p-val roc-auc', p_rocauc)

    t_precision, p_precision = ttest_ind(if_precison, sf_precision)
    neptune.log_metric(f'{dataset} t-stat precision', t_precision)
    neptune.log_metric(f'{dataset} p-val precision', p_precision)

    t_recall, p_recall = ttest_ind(if_recall, sf_recall)
    neptune.log_metric(f'{dataset} t-stat precision', t_recall)
    neptune.log_metric(f'{dataset} p-val precision', p_recall)

    t_f1, p_f1 = ttest_ind(if_f1, sf_f1)
    neptune.log_metric(f'{dataset} t-stat f1', t_f1)
    neptune.log_metric(f'{dataset} p-val f1', p_f1)

    df.loc[d_idx] = [dataset,
                     np.mean(if_auc), np.mean(sf_auc),
                     np.mean(if_precison), np.mean(sf_precision),
                     np.mean(if_recall), np.mean(sf_recall),
                     np.mean(if_f1), np.mean(sf_f1),
                     np.std(if_auc), np.std(sf_auc),
                     np.std(if_precison), np.std(sf_precision),
                     np.std(if_recall), np.std(sf_recall),
                     np.std(if_f1), np.std(sf_f1),
                     p_rocauc, p_precision, p_recall, p_f1]
    df.to_csv(log_name, index=False)


df.to_csv(log_name, index=False)
neptune.log_artifact(log_name)
neptune.stop()
