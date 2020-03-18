import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.stats import ttest_ind
import neptune
from sklearn.ensemble import RandomForestClassifier
from simforest import SimilarityForestClassifier
from simforest.distance import rbf, dot_product
from examples.classification.classification_datasets import get_datasets
from sklearn.multiclass import OneVsRestClassifier


use_neptune = False
binary = False

if use_neptune:
    neptune.set_project('sfczekalski/SimilarityForest')
    neptune.init('sfczekalski/SimilarityForest')


# set experiment properties
n_iterations = 20

# create experiment
if use_neptune:
    neptune.create_experiment(name='Multiclass classification',
                              properties={'n_iterations': n_iterations})

# init log
df = pd.DataFrame(columns=['dataset',
                           'SF f1', 'SF roc-auc', 'SF acc',
                           'RF f1', 'RF roc-auc', 'RF acc',
                           'p-val f1', 'p-val roc-auc', 'p-val acc'])
log_name = 'logs/multiclass_classification_log.csv'


# load and prepare data
for d_idx, d in enumerate(get_datasets()):
    X_train, X_test, y_train, y_test, dataset, sf_params, rf_params = d

    # store results for t-test
    rf_f1 = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_f1 = np.zeros(shape=(n_iterations,), dtype=np.float32)

    rf_roc = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_roc = np.zeros(shape=(n_iterations,), dtype=np.float32)

    rf_acc = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_acc = np.zeros(shape=(n_iterations,), dtype=np.float32)

    # run
    for i in range(n_iterations):
        print(f'{dataset}, {i+1} / {n_iterations}')

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        rf_dec_f = rf.predict_proba(X_test)
        rf_pred = rf.classes_[np.argmax(rf_dec_f, axis=1)]

        rf_f1[i] = f1_score(y_test, rf_pred, average='weighted')
        if binary:
            rf_roc[i] = roc_auc_score(y_test, rf_dec_f[:, 1])
        else:
            rf_roc[i] = roc_auc_score(y_test, rf_dec_f, average='weighted', multi_class='ovr')
        rf_acc[i] = accuracy_score(y_test, rf_pred)

        sf = SimilarityForestClassifier(**sf_params)
        sf.fit(X_train, y_train)
        sf_dec_f = sf.predict_proba(X_test)
        sf_pred = sf.classes_[np.argmax(sf_dec_f, axis=1)]

        sf_f1[i] = f1_score(y_test, sf_pred, average='weighted')
        if binary:
            sf_roc[i] = roc_auc_score(y_test, sf_dec_f[:, 1])
        else:
            sf_roc[i] = roc_auc_score(y_test, sf_dec_f, average='weighted', multi_class='ovr')
        sf_acc[i] = accuracy_score(y_test, sf_pred)

    mean_rf_f1 = np.mean(rf_f1)
    mean_sf_f1 = np.mean(sf_f1)

    mean_rf_roc = np.mean(rf_roc)
    mean_sf_roc = np.mean(sf_roc)

    mean_rf_acc = np.mean(rf_acc)
    mean_sf_acc = np.mean(sf_acc)

    # log results
    if use_neptune:
        neptune.log_metric(f'{dataset} RF f1', mean_rf_f1)
        neptune.log_metric(f'{dataset} SF f1', mean_sf_f1)

        neptune.log_metric(f'{dataset} RF roc-auc', mean_rf_roc)
        neptune.log_metric(f'{dataset} SF roc-auc', mean_sf_roc)

        neptune.log_metric(f'{dataset} RF accuracy', mean_rf_acc)
        neptune.log_metric(f'{dataset} SF accuracy', mean_sf_acc)

    t_f1, p_f1 = ttest_ind(rf_f1, sf_f1)
    t_roc, p_roc = ttest_ind(rf_roc, sf_roc)
    t_acc, p_acc = ttest_ind(rf_acc, sf_acc)

    if use_neptune:
        neptune.log_metric(f'{dataset} t-stat', t_f1)
        neptune.log_metric(f'{dataset} p-val', p_f1)

        neptune.log_metric(f'{dataset} t-stat', t_roc)
        neptune.log_metric(f'{dataset} p-val', p_roc)

        neptune.log_metric(f'{dataset} t-stat', t_acc)
        neptune.log_metric(f'{dataset} p-val', p_acc)

    df.loc[d_idx] = [dataset,
                     mean_sf_f1, mean_sf_roc, mean_sf_acc,
                     mean_rf_f1, mean_rf_roc, mean_rf_acc,
                     p_f1, p_roc, p_acc]
    df.to_csv(log_name, index=False)

if use_neptune:
    neptune.log_artifact(log_name)
    neptune.stop()
