import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind
import neptune
from sklearn.ensemble import IsolationForest
from simforest import SimilarityForestClassifier
from examples.outliers.outliers_datasets import get_datasets


neptune.init('sfczekalski/similarity-forest')

# set parameters
params = dict()
params['discriminative_sampling'] = False
params['most_different'] = False
params['bootstrap'] = False
params['max_samples'] = 256
params['max_depth'] = int(np.ceil(np.log2(256)))
params['n_estimators'] = 20


# set experiment properties
n_iterations = 20

# create experiment
neptune.create_experiment(name='Outlier detection',
                          params=params,
                          properties={'n_iterations': n_iterations})


# load and prepare data
for d in get_datasets():
    X_train, X_test, y_train, y_test, dataset = d


    # store auc-roc for t-test
    if_auc = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_auc = np.zeros(shape=(n_iterations,), dtype=np.float32)

    # run
    for i in range(n_iterations):
        print(f'{dataset}, {i+1} / {n_iterations}')
        IF = IsolationForest()
        IF.fit(X_train, y_train)
        if_pred = IF.predict(X_test)
        if_auc[i] = roc_auc_score(y_test, if_pred)

        sf = SimilarityForestClassifier(**params)
        sf.fit(X_train, y_train)
        sf_pred = sf.predict_outliers(X_test)
        sf_auc[i] = roc_auc_score(y_test, sf_pred)

    # log results
    neptune.log_metric(f'{dataset} IF ROC-AUC', np.mean(if_auc))
    neptune.log_metric(f'{dataset} SF ROC-AUC', np.mean(sf_auc))

    t, p = ttest_ind(if_auc, sf_auc)
    neptune.log_metric(f'{dataset} t-stat', t)
    neptune.log_metric(f'{dataset} p-val', p)

neptune.stop()
