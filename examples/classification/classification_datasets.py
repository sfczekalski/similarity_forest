from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
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


def get_a1a():
    X, y = load_svmlight_file('../data/a1a')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 10
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2

    rf_params = dict()
    rf_params['max_depth'] = 10

    return X_train, X_test, y_train, y_test, 'a1a', sf_params, rf_params


def get_svmguide3():
    X, y = load_svmlight_file('../data/svmguide3')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 12
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2

    rf_params = dict()
    rf_params['max_depth'] = 10

    return X_train, X_test, y_train, y_test, 'svmguide3', sf_params, rf_params


def get_heart():
    X, y = load_svmlight_file('../data/heart')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 14
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 1

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'heart', sf_params, rf_params


def get_ionosphere():
    X, y = load_svmlight_file('../data/ionosphere_scale')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    sf_params = dict()
    sf_params['max_depth'] = 14
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'ionosphere_scale', sf_params, rf_params


def get_breast_cancer():
    X, y = load_svmlight_file('../data/breast-cancer')
    X = X.toarray().astype(np.float32)
    y = LabelBinarizer().fit_transform(y).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 8
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2

    rf_params = dict()
    rf_params['max_depth'] = 10

    return X_train, X_test, y_train, y_test, 'breast_cancer', sf_params, rf_params


def get_german_numer():
    X, y = load_svmlight_file('../data/german_numer')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 14
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 3

    rf_params = dict()
    rf_params['max_depth'] = 12

    return X_train, X_test, y_train, y_test, 'german_numer', sf_params, rf_params


def get_mushrooms():
    X, y = load_svmlight_file('../data/mushrooms')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 14
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 3

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'mushrooms', sf_params, rf_params


def get_iris():
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = None
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 3

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'iris', sf_params, rf_params


def get_glass():
    df = pd.read_csv('../data/dataset_glass.csv')
    df = fix_dtypes(df)
    y, X = df.pop('Type'), df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train.values[shuffled_indices], y_train.values[shuffled_indices]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 8
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 1

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'glass', sf_params, rf_params


def get_seed():
    df = pd.read_csv('../data/seeds_dataset.csv')
    df = fix_dtypes(df)
    y, X = df.pop('class'), df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train.values[shuffled_indices], y_train.values[shuffled_indices]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 10
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 1

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'seed', sf_params, rf_params


def get_wine():
    df = pd.read_csv('../data/wine.data')
    df = fix_dtypes(df)
    y, X = df.iloc[:, 0], df.iloc[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train.values[shuffled_indices], y_train.values[shuffled_indices]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 14
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 1

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'wine', sf_params, rf_params


def get_splice():
    X, y = load_svmlight_file('../data/splice')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 12
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2

    rf_params = dict()
    rf_params['max_depth'] = 14

    return X_train, X_test, y_train, y_test, 'splice', sf_params, rf_params


def get_madelon():
    X, y = load_svmlight_file('../data/madelon')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 8
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 1

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'madelon', sf_params, rf_params


def get_diabetes():
    X, y = load_svmlight_file('../data/diabetes')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 12
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 1

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'diabetes', sf_params, rf_params


def get_australian():
    X, y = load_svmlight_file('../data/australian')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = 12
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2

    rf_params = dict()
    rf_params['max_depth'] = 10

    return X_train, X_test, y_train, y_test, 'australian', sf_params, rf_params


def get_dna():
    X, y = load_svmlight_file('../data/dna.scale')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    sf_params = dict()
    sf_params['max_depth'] = 10
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2

    rf_params = dict()
    rf_params['max_depth'] = 14

    return X_train, X_test, y_train, y_test, 'dna', sf_params, rf_params


def get_letter():
    X, y = load_svmlight_file('../data/letter.scale')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, 'letter'


def get_pendigits():
    X, y = load_svmlight_file('../data/pendigits')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, 'pendigits'


def get_segment():
    X, y = load_svmlight_file('../data/segment.scale')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['max_depth'] = None
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 3

    rf_params = dict()
    rf_params['max_depth'] = 10

    return X_train, X_test, y_train, y_test, 'segment', sf_params, rf_params


'''
    # binary
    get_breast_cancer(),
    get_german_numer(),
    get_mushrooms(),
    get_madelon(),
    get_diabetes(),
    get_australian(),
    get_splice(),
    get_a1a(),
    get_svmguide3(),
    get_heart(),
    get_ionosphere()
'''

datasets = [
    # multiclass
    get_iris(),
    get_glass(),
    get_seed(),
    get_wine(),
    get_dna(),
    get_segment()
]


def get_datasets():
    for d in datasets:
        yield d

