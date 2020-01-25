import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file, load_boston
from sklearn.preprocessing import LabelEncoder


def fix_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    obj_cols = [c for c in df if df[c].dtype == 'object']
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    for c in obj_cols:
        df[c] = LabelEncoder().fit_transform(df[c])

    return df


def get_hardware_dataset():
    df = pd.read_csv('../data/machine.data', header=None)
    df.loc[:, 0] = LabelEncoder().fit_transform(df.loc[:, 0])
    df.loc[:, 1] = LabelEncoder().fit_transform(df.loc[:, 1])
    df.pop(0)
    y, X = df.pop(9), df
    y = np.log1p(y)

    return X, y, 'computer_hardware'


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

    return X, y, 'concrete_slump'


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

    return X, y, 'concrete_flow'


def get_energy_efficiency_heating():
    df = pd.read_excel('../data/ENB2012_data.xlsx')
    df.pop('Y2')
    y, X = df.pop('Y1'), df

    return X, y, 'energy_efficiency_heating'


def get_energy_efficiency_cooling():
    df = pd.read_excel('../data/ENB2012_data.xlsx')
    df.pop('Y1')
    y, X = df.pop('Y2'), df

    return X, y, 'energy_efficiency_cooling'


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

    return X, y, 'who'


def get_mpg_dataset():
    X, y = load_svmlight_file('../data/mpg')
    X = X.toarray()
    return X, y, 'mpg'


def get_eunite2001_dataset():
    X, y = load_svmlight_file('../data/eunite2001')
    X = X.toarray()
    return X, y, 'eunite2001'


def get_abalone_dataset():
    X, y = load_svmlight_file('../data/abalone')
    X = X.toarray()
    return X, y, 'abalone'


def get_spacega_dataset():
    X, y = load_svmlight_file('../data/space_ga')
    X = X.toarray()
    return X, y, 'space_ga'


def get_boston_dataset():
    X, y = load_boston(return_X_y=True)
    return X, y, 'boston'


def get_auto_dataset():
    df = pd.read_csv('auto.data', header=None)
    df.dropna(inplace=True)


datasets = [
            get_mpg_dataset()
]

'''
            get_concrete_flow_dataset(),
            get_hardware_dataset(),
            get_boston_dataset(),
            get_energy_efficiency_heating()
'''

def get_datasets():
    for d in datasets:
        yield d
