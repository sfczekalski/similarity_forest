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
from examples.regression.regression_datasets import get_datasets
import time

neptune.set_project('sfczekalski/SimilarityForest')

neptune.init('sfczekalski/SimilarityForest')


# set experiment properties
n_iterations = 20

# create experiment
neptune.create_experiment(name='Regression - tuned',
                          properties={'n_iterations': n_iterations})

# init log
df = pd.DataFrame(columns=['dataset', 'RF RMSE', ' SF RMSE', 'p-val'])


# load and prepare data
for d_idx, d in enumerate(get_datasets()):
    X_train, X_test, y_train, y_test, dataset, sf_params, rf_params = d

    # store mse for t-test
    rf_mse = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_mse = np.zeros(shape=(n_iterations,), dtype=np.float32)

    # run
    for i in range(n_iterations):
        print(f'{dataset}, {i+1} / {n_iterations}')
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_mse[i] = mean_squared_error(y_test, rf_pred)

        sf = SimilarityForestRegressor(**sf_params)
        sf.fit(X_train, y_train)
        sf_pred = sf.predict(X_test)
        sf_mse[i] = mean_squared_error(y_test, sf_pred)

    mean_rf_rmse = np.sqrt(np.mean(rf_mse))
    mean_sf_rmse = np.sqrt(np.mean(sf_mse))

    # log results
    neptune.log_metric(f'{dataset} RF RMSE', mean_rf_rmse)
    neptune.log_metric(f'{dataset} SF RMSE', mean_sf_rmse)

    t, p = ttest_ind(np.sqrt(rf_mse), np.sqrt(sf_mse))
    neptune.log_metric(f'{dataset} t-stat', t)
    neptune.log_metric(f'{dataset} p-val', p)

    df.loc[d_idx] = [dataset, mean_rf_rmse, mean_sf_rmse, p]


log_name = 'logs/regression_tuned_log.csv'
df.to_csv(log_name, index=False)
neptune.log_artifact(log_name)
neptune.stop()
