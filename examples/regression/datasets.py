import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file, load_boston, make_friedman1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def fix_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    #obj_cols = [c for c in df if df[c].dtype == 'object']
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    '''for c in obj_cols:
        df[c] = LabelEncoder().fit_transform(df[c])'''

    return df


def get_hardware_dataset():
    df = pd.read_csv('../data/machine.data', header=None)
    df.loc[:, 0] = LabelEncoder().fit_transform(df.loc[:, 0])
    df.loc[:, 1] = LabelEncoder().fit_transform(df.loc[:, 1])
    df.pop(0)
    y, X = df.pop(9), df
    y = np.log1p(y)

    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'computer_hardware'


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
    df = fix_dtypes(df)
    y, X = df.pop('SLUMP(cm)'), df

    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'concrete_slump'


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
    df = fix_dtypes(df)
    y, X = df.pop('FLOW(cm)'), df

    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'concrete_flow'


def get_energy_efficiency_heating():
    df = pd.read_excel('../data/ENB2012_data.xlsx')
    df.pop('Y2')
    df = fix_dtypes(df)
    y, X = df.pop('Y1'), df

    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'energy_efficiency_heating'


def get_energy_efficiency_cooling():
    df = pd.read_excel('../data/ENB2012_data.xlsx')
    df.pop('Y1')
    df = fix_dtypes(df)
    y, X = df.pop('Y2'), df

    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'energy_efficiency_cooling'


def get_who_dataset():
    df = pd.read_csv('../data/Life Expectancy Data.csv')
    '''df['Country'] = LabelEncoder().fit_transform(df['Country'])
    df['Status'] = LabelEncoder().fit_transform(df['Status'])'''
    df = pd.concat([df, pd.get_dummies(df['Country'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Status'])], axis=1)
    df.drop(columns=['Country', 'Status'], inplace=True)
    df.dropna(inplace=True)
    df = fix_dtypes(df)
    y, X = df.pop('Life expectancy '), df

    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'who'


def get_mpg_dataset():
    X, y = load_svmlight_file('../data/mpg')
    X = X.toarray()
    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'mpg'


def get_eunite2001_dataset():
    X, y = load_svmlight_file('../data/eunite2001')
    X = X.toarray()
    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'eunite2001'


def get_abalone_dataset():
    X, y = load_svmlight_file('../data/abalone')
    X = X.toarray()
    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'abalone'


def get_spacega_dataset():
    X, y = load_svmlight_file('../data/space_ga')
    X = X.toarray()
    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'space_ga'


def get_boston_dataset():
    X, y = load_boston(return_X_y=True)
    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'boston'


def get_auto_dataset():
    df = pd.read_csv('auto.data', header=None)
    df.dropna(inplace=True)
    df = fix_dtypes(df)


def get_servo_dataset(onehot=False, scale=False):
    df = pd.read_csv('../data/servo.data', header=None, names=['a', 'b', 'c', 'd', 'class'])
    if onehot:
        df = pd.concat([df, pd.get_dummies(df['a'])], axis=1)
        df = pd.concat([df, pd.get_dummies(df['b'])], axis=1)
        df.drop(columns=['a', 'b'], inplace=True)
    else:
        df['a'] = LabelEncoder().fit_transform(df['a'])
        df['b'] = LabelEncoder().fit_transform(df['b'])

    y, X = df.pop('class'), df.values
    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

    if scale:
        scaler = StandardScaler()
        X_train[:, 0:2] = scaler.fit_transform(X_train[:, 0:2])
        X_test[:, 0:2] = scaler.transform(X_test[:, 0:2])

    return X_train, X_test, y_train, y_test, 'servo'


def get_wine_quality():
    df = pd.read_csv('../data/winequality-white.csv', sep=';')
    df.dropna(inplace=True)

    y, X = df.pop('quality'), df
    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'wine_quality'


def get_friedman_dataset():
    X, y = make_friedman1(n_samples=1000, random_state=42)

    y = y + np.abs(np.min(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'friedman'

'''
            get_friedman_dataset(),
            get_boston_dataset(),
            get_energy_efficiency_heating(),
            get_spacega_dataset(),
            get_eunite2001_dataset(),
            get_wine_quality(),
            get_abalone_dataset(),
            get_boston_dataset()
'''

datasets = [

            get_mpg_dataset(),
            get_concrete_flow_dataset(),
            get_hardware_dataset()
]


def get_datasets():
    for d in datasets:
        yield d

