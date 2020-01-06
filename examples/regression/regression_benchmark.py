import neptune
from simforest import SimilarityForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression, load_svmlight_file, load_wine, make_friedman1, load_boston
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd
import gc


def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


def get_hardware_dataset():
    df = pd.read_csv('../data/machine.data', header=None)
    df.loc[:, 0] = LabelEncoder().fit_transform(df.loc[:, 0])
    df.loc[:, 1] = LabelEncoder().fit_transform(df.loc[:, 1])
    df.pop(0)
    y, X = df.pop(9), df
    y = np.log1p(y)

    return y, X


def get_forest_fires_dataset():
    df = pd.read_csv('../data/forestfires.csv')
    df['month'] = LabelEncoder().fit_transform(df['month'])
    df['day'] = LabelEncoder().fit_transform(df['day'])
    y, X = df.pop('area'), df

    return y, X


def get_concrete_slump_dataset():
    """
    Attribute Information:
        Input variables (7)(component kg in one M^3 concrete):
            Cement
            Slag
            Fly ash
            Water
            SP
            Coarse Aggr.
            Fine Aggr.

        Output variables (3):
            SLUMP (cm)
            FLOW (cm)
            28-day Compressive Strength (Mpa)
    """
    df = pd.read_csv('../data/slump_test.data')
    df.drop(columns=['FLOW(cm)', 'Compressive Strength (28-day)(Mpa)'], inplace=True)
    y, X = df.pop('SLUMP(cm)'), df

    return y, X


def get_concrete_flow_dataset():
    """
    Attribute Information:
        Input variables (7)(component kg in one M^3 concrete):
            Cement
            Slag
            Fly ash
            Water
            SP
            Coarse Aggr.
            Fine Aggr.

        Output variables (3):
            SLUMP (cm)
            FLOW (cm)
            28-day Compressive Strength (Mpa)
    """
    df = pd.read_csv('../data/slump_test.data')
    df.drop(columns=['SLUMP(cm)', 'Compressive Strength (28-day)(Mpa)'], inplace=True)
    y, X = df.pop('FLOW(cm)'), df

    return y, X


def get_energy_efficiency_heating():
    df = pd.read_excel('../data/ENB2012_data.xlsx')
    df.pop('Y2')
    y, X = df.pop('Y1'), df

    return y, X


def get_energy_efficiency_cooling():
    df = pd.read_excel('../data/ENB2012_data.xlsx')
    df.pop('Y1')
    y, X = df.pop('Y2'), df

    return y, X


def get_who_dataset():
    df = pd.read_csv('../data/Life Expectancy Data.csv')
    '''df['Country'] = LabelEncoder().fit_transform(df['Country'])
    df['Status'] = LabelEncoder().fit_transform(df['Status'])'''
    df = pd.concat([df, pd.get_dummies(df['Country'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Status'])], axis=1)
    df.drop(columns=['Country', 'Status'], inplace=True)
    df.dropna(inplace=True)
    df = downcast_dtypes(df)
    y, X = df.pop('Life expectancy '), df

    return y, X

neptune.init('sfczekalski/similarity-forest')

# set parameters
params = dict()
params['criterion'] = 'variance'
params['discriminative_sampling'] = True
params['max_depth'] = None
params['n_estimators'] = 100


# set experiment properties
n_iterations = 10

# store mse for t-test
rf_mse = np.zeros(shape=(n_iterations,), dtype=np.float32)
sf_mse = np.zeros(shape=(n_iterations,), dtype=np.float32)

# create experiment
neptune.create_experiment(name='Regression WHO life expectancy variance different from mean',
                          params=params,
                          properties={'n_iterations': n_iterations,
                                      'dataset': 'WHO life expectancy'})


# load and prepare data
y, X = get_who_dataset()
y = y + np.abs(np.min(y))

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# run
for i in range(n_iterations):
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    neptune.log_metric('RF MSE', mean_squared_error(y_test, rf_pred))
    neptune.log_metric('RF R2', r2_score(y_test, rf_pred))
    rf_mse[i] = mean_squared_error(y_test, rf_pred)

    sf = SimilarityForestRegressor(**params)
    sf.fit(X_train, y_train)
    sf_pred = sf.predict(X_test)
    neptune.log_metric('SF MSE', mean_squared_error(y_test, sf_pred))
    neptune.log_metric('SF R2', r2_score(y_test, sf_pred))
    sf_mse[i] = mean_squared_error(y_test, sf_pred)

    # clean up
    rf = None
    sf = None
    gc.collect()

# log results
neptune.log_metric('RF mean MSE', np.mean(rf_mse))
neptune.log_metric('SF mean MSE', np.mean(sf_mse))

t, p = ttest_ind(rf_mse, sf_mse)
neptune.log_metric('t-stat', t)
neptune.log_metric('p-val', p)
neptune.stop()
