from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd


def fix_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    #obj_cols = [c for c in df if df[c].dtype == 'object']
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    '''for c in obj_cols:
        df[c] = LabelEncoder().fit_transform(df[c])'''

    return df


# Binary datasets
def get_a1a():
    X, y = load_svmlight_file('../data/a1a')
    X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'a1a'


def get_svmguide3():
    X, y = load_svmlight_file('../data/svmguide3')
    X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'svmguide3'


def get_heart():
    X, y = load_svmlight_file('../data/heart')
    X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'heart'


def get_ionosphere():
    X, y = load_svmlight_file('../data/ionosphere_scale')
    X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, 'ionosphere_scale'


def get_breast_cancer():
    X, y = load_svmlight_file('../data/breast-cancer')
    X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'breast_cancer'


def get_german_numer():
    X, y = load_svmlight_file('../data/german_numer')
    X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'german_numer'


def get_mushrooms():
    X, y = load_svmlight_file('../data/mushrooms')
    X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'mushrooms'


# Multi-class datasets
def get_iris():
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'iris'


def get_glass():
    df = pd.read_csv('../data/dataset_glass.csv')
    y, X = df.pop('Type'), df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'glass'


datasets = [
    get_a1a(),
    get_svmguide3(),
    get_heart(),
    get_ionosphere(),
    get_breast_cancer(),
    get_german_numer(),
    get_mushrooms(),
    get_iris(),
    get_glass()
]


def get_datasets():
    for d in datasets:
        yield d
