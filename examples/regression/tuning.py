import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_ind
import neptune
from sklearn.ensemble import RandomForestRegressor
from simforest import SimilarityForestRegressor
from simforest.distance import rbf, dot_product
from examples.regression.regression_datasets import get_datasets
from sklearn.model_selection import GridSearchCV


sf_params = {
    'max_depth': [8, 10, 12, 14, None],
    'n_directions': [1, 2, 3],
    'criterion': ['variance', 'theil', 'atkinson'],
    'sim_function': [rbf, dot_product]
}
rf_params = {
    'max_depth': [8, 10, 12, 14, None]
}

# init log
df = pd.DataFrame(columns=['dataset',
                           'SF R2', 'SF MSE', 'SF criterion', 'SF n_directions', 'SF max_depth', 'SF sim_f',
                           'RF R2', 'RF MSE', 'RF max_depth'])

log_name = 'logs/regression_tuning.csv'

binary = False

# load and prepare data
for d_idx, d in enumerate(get_datasets()):
    X_train, X_test, y_train, y_test, dataset, _, _ = d
    print(dataset)

    # RF
    rf = GridSearchCV(RandomForestRegressor(), param_grid=rf_params, cv=5)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    # SF
    sf = GridSearchCV(SimilarityForestRegressor(), param_grid=sf_params, cv=5)
    sf.fit(X_train, y_train)
    sf_pred = sf.predict(X_test)
    sf_mse = mean_squared_error(y_test, sf_pred)
    sf_r2 = r2_score(y_test, sf_pred)

    df.loc[d_idx] = [dataset, sf_r2, sf_mse, sf.best_params_['criterion'], sf.best_params_['n_directions'],
                     sf.best_params_['max_depth'], sf.best_params_['sim_function'],
                     rf_r2, rf_mse, rf.best_params_['max_depth']]
    print(df.loc[d_idx])
    df.to_csv(log_name, index=False)
