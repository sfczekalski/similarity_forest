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
from sklearn.model_selection import GridSearchCV

use_neptune = True

# set experiment properties
n_iterations = 20
n_folds = 10

if use_neptune:
    neptune.set_project('sfczekalski/SimilarityForest')
    neptune.init('sfczekalski/SimilarityForest')
    # create experiment
    neptune.create_experiment(name='Regression',
                              properties={'n_iterations': n_iterations, 'n_folds': n_folds})

# init log
df = pd.DataFrame(columns=['dataset', 'RF RMSE', ' SF RMSE', 'p-val', 'sf params', 'rf params'])
df_std_cv_score = pd.DataFrame(columns=['dataset', 'SF cv score std', 'RF cv score std'])
log_name = 'logs/regression_log_rbf5'

sf_params = {
    'max_depth': [8, 10, 12, 14, None],
    'n_directions': [1, 2, 3],
    'sim_function': [rbf],
    'criterion': ['variance'],
    'gamma': [1.0, None],
    'n_estimators': [25, 50, 100]
}

rf_params = {
    'max_depth': [10, 12, 14, None],
    'n_estimators': [25, 50, 100]
}

# load and prepare data
for d_idx, d in enumerate(get_datasets()):
    X_train, X_test, y_train, y_test, dataset = d

    # store mse for t-test
    rf_mse = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_mse = np.zeros(shape=(n_iterations,), dtype=np.float32)

    # Find parameters
    # RF
    rf = GridSearchCV(RandomForestRegressor(), param_grid=rf_params, cv=n_folds, n_jobs=4)
    rf.fit(X_train, y_train)

    # SF
    sf = GridSearchCV(SimilarityForestRegressor(), param_grid=sf_params, cv=n_folds, verbose=5, n_jobs=4)
    sf.fit(X_train, y_train)

    # Log std of CV scores for different parameters
    rf_std_score = np.std(rf.cv_results_['mean_test_score'])
    if use_neptune:
        neptune.log_metric(f'{dataset} RF std cv score', rf_std_score)

    sf_std_score = np.std(sf.cv_results_['mean_test_score'])
    if use_neptune:
        neptune.log_metric(f'{dataset} SF std cv score', sf_std_score)

    df_std_cv_score.loc[d_idx] = [dataset, sf_std_score, rf_std_score]
    print(df_std_cv_score.loc[d_idx])
    df_std_cv_score.to_csv(log_name + '_cv_std_score.csv', index=False)

    sf_best_params = sf.best_params_
    rf_best_params = rf.best_params_

    # run
    for i in range(n_iterations):
        print(f'{dataset}, {i+1} / {n_iterations}')
        rf = RandomForestRegressor(**rf_best_params)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_mse[i] = mean_squared_error(y_test, rf_pred)

        sf = SimilarityForestRegressor(**sf_best_params)
        sf.fit(X_train, y_train)
        sf_pred = sf.predict(X_test)
        sf_mse[i] = mean_squared_error(y_test, sf_pred)

    mean_rf_rmse = np.sqrt(np.mean(rf_mse))
    mean_sf_rmse = np.sqrt(np.mean(sf_mse))

    # Calculate std across the runs
    std_rf_rmse = np.std(np.sqrt(rf_mse))
    std_sf_rmse = np.std(np.sqrt(sf_mse))
    t, p = ttest_ind(np.sqrt(rf_mse), np.sqrt(sf_mse))

    if use_neptune:
        # log results
        neptune.log_metric(f'{dataset} RF RMSE', mean_rf_rmse)
        neptune.log_metric(f'{dataset} SF RMSE', mean_sf_rmse)

        # Std
        neptune.log_metric(f'std {dataset} RF f1', std_rf_rmse)
        neptune.log_metric(f'std {dataset} SF f1', std_sf_rmse)

        neptune.log_metric(f'{dataset} t-stat', t)
        neptune.log_metric(f'{dataset} p-val', p)

    df.loc[d_idx] = [dataset, mean_rf_rmse, mean_sf_rmse, p, sf_best_params, rf_best_params]
    print(df.loc[d_idx])
    df.to_csv(log_name + '.csv', index=False)

if use_neptune:
    neptune.log_artifact(log_name + '.csv')
    neptune.log_artifact(log_name + 'cv_std_score.csv')
    neptune.stop()
