import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind
import neptune
from sklearn.ensemble import IsolationForest
from simforest.isolation_simforest import IsolationSimilarityForest
from examples.outliers.outliers_datasets import get_datasets


neptune.set_project('sfczekalski/SimilarityForest')

neptune.init('sfczekalski/SimilarityForest')

# set parameters
params = dict()
params['most_different'] = True
params['max_samples'] = 256
params['max_depth'] = int(np.ceil(np.log2(256)))
params['n_estimators'] = 100


# set experiment properties
n_iterations = 20

# create experiment
neptune.create_experiment(name='Outlier detection - most different split strategy',
                          params=params,
                          properties={'n_iterations': n_iterations})

# init log
df = pd.DataFrame(columns=['dataset', 'IF roc-auc', ' SF roc-auc', 'p-val'])
log_name = 'logs/outlier_detection_most_different_log.csv'


# load and prepare data
for d_idx, d in enumerate(get_datasets()):
    X_train, X_test, y_train, y_test, dataset = d

    # store auc-roc for t-test
    if_auc = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_auc = np.zeros(shape=(n_iterations,), dtype=np.float32)

    # run
    for i in range(n_iterations):
        print(f'{dataset}, {i+1} / {n_iterations}')
        IF = IsolationForest()
        IF.fit(X_train, y_train)
        if_pred = IF.decision_function(X_test)
        if_auc[i] = roc_auc_score(y_test, if_pred)

        sf = IsolationSimilarityForest(**params)
        sf.fit(X_train, y_train)
        sf_pred = sf.decision_function(X_test)
        sf_auc[i] = roc_auc_score(y_test, sf_pred)

    # log results
    neptune.log_metric(f'{dataset} IF ROC-AUC', np.mean(if_auc))
    neptune.log_metric(f'{dataset} SF ROC-AUC', np.mean(sf_auc))

    t, p = ttest_ind(if_auc, sf_auc)
    neptune.log_metric(f'{dataset} t-stat', t)
    neptune.log_metric(f'{dataset} p-val', p)

    df.loc[d_idx] = [dataset, np.mean(if_auc), np.mean(sf_auc), p]
    df.to_csv(log_name, index=False)


df.to_csv(log_name, index=False)
neptune.log_artifact(log_name)
neptune.stop()
