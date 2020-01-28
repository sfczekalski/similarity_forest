import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import ttest_ind
import neptune
from sklearn.ensemble import RandomForestRegressor
from simforest import SimilarityForestRegressor
from simforest.distance import rbf, dot_product
from examples.regression.datasets import get_datasets
import time


neptune.init('sfczekalski/similarity-forest')

# set parameters
params = dict()
params['criterion'] = 'atkinson'
params['discriminative_sampling'] = True
params['max_depth'] = None
params['n_estimators'] = 100
params['sim_function'] = rbf
params['n_directions'] = 1


# set experiment properties
n_iterations = 20

# create experiment
neptune.create_experiment(name='Regression',
                          params=params,
                          properties={'n_iterations': n_iterations})


# load and prepare data
for d in get_datasets():
    X_train, X_test, y_train, y_test, dataset = d


    # store mse for t-test
    rf_mse = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_mse = np.zeros(shape=(n_iterations,), dtype=np.float32)

    # init log
    df = pd.DataFrame(columns=[f'{dataset} RF RMSE',
                               f'{dataset} SF RMSE',
                               f'{dataset} p-val'])

    # run
    for i in range(n_iterations):
        print(f'{dataset}, {i+1} / {n_iterations}')
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_mse[i] = mean_squared_error(y_test, rf_pred)

        sf = SimilarityForestRegressor(**params)
        sf.fit(X_train, y_train)
        sf_pred = sf.predict(X_test)
        sf_mse[i] = mean_squared_error(y_test, sf_pred)

    # log results
    neptune.log_metric(f'{dataset} RF RMSE', np.mean(np.sqrt(rf_mse)))
    neptune.log_metric(f'{dataset} SF RMSE', np.mean(np.sqrt(sf_mse)))
    df[f'{dataset} RF RMSE'] = np.mean(np.sqrt(rf_mse))
    df[f'{dataset} SF RMSE'] = np.mean(np.sqrt(sf_mse))

    t, p = ttest_ind(np.sqrt(rf_mse), np.sqrt(sf_mse))
    neptune.log_metric(f'{dataset} t-stat', t)
    neptune.log_metric(f'{dataset} p-val', p)
    df[f'{dataset} p-val'] = p


df.to_csv('regression_log.csv')
neptune.log_artifact('regression_log.csv')
neptune.stop()
