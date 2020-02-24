import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import ttest_ind
import neptune
from sklearn.ensemble import RandomForestClassifier
from simforest import SimilarityForestClassifier
from simforest.distance import rbf, dot_product
from examples.classification.classification_datasets import get_datasets


neptune.init('sfczekalski/similarity-forest')

# set parameters
params = dict()
params['max_depth'] = None
params['n_estimators'] = 100
params['sim_function'] = dot_product
params['n_directions'] = 1


# set experiment properties
n_iterations = 20

# create experiment
neptune.create_experiment(name='Classification',
                          params=params,
                          properties={'n_iterations': n_iterations})


# load and prepare data
for d in get_datasets():
    X_train, X_test, y_train, y_test, dataset = d


    # store results for t-test
    rf_acc = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_acc = np.zeros(shape=(n_iterations,), dtype=np.float32)

    # run
    for i in range(n_iterations):
        print(f'{dataset}, {i+1} / {n_iterations}')
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc[i] = accuracy_score(y_test, rf_pred)

        sf = SimilarityForestClassifier(**params)
        sf.fit(X_train, y_train)
        sf_pred = sf.predict(X_test)
        sf_acc[i] = accuracy_score(y_test, sf_pred)

    # log results
    neptune.log_metric(f'{dataset} RF accuracy', np.mean(rf_acc))
    neptune.log_metric(f'{dataset} SF accuracy', np.mean(sf_acc))

    t, p = ttest_ind(rf_acc, sf_acc)
    neptune.log_metric(f'{dataset} t-stat', t)
    neptune.log_metric(f'{dataset} p-val', p)

neptune.stop()
