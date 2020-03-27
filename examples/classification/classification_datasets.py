from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, load_iris, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
import numpy as np
import pandas as pd
from simforest.distance import rbf, dot_product
from sklearn.utils import shuffle
from scipy.io.arff import loadarff


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
    sf_params['max_depth'] = 14
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2
    sf_params['sim_function'] = dot_product

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
    sf_params['max_depth'] = 14
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 1
    sf_params['sim_function'] = rbf

    rf_params = dict()
    rf_params['max_depth'] = None

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
    sf_params['n_directions'] = 2
    sf_params['sim_function'] = rbf

    rf_params = dict()
    rf_params['max_depth'] = None

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
    sf_params['sim_function'] = rbf

    rf_params = dict()
    rf_params['max_depth'] = 12

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
    sf_params['max_depth'] = None
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = None

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
    sf_params['max_depth'] = 10
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = None

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
    sf_params['max_depth'] = 10
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2
    sf_params['sim_function'] = rbf

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
    sf_params['max_depth'] = 8
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 2
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = None

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
    sf_params['max_depth'] = 10
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 1
    sf_params['sim_function'] = rbf

    rf_params = dict()
    rf_params['max_depth'] = 12

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
    sf_params['max_depth'] = 8
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 3
    sf_params['sim_function'] = rbf

    rf_params = dict()
    rf_params['max_depth'] = 12

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
    sf_params['max_depth'] = 8
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 1
    sf_params['sim_function'] = rbf

    rf_params = dict()
    rf_params['max_depth'] = 12

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
    sf_params['sim_function'] = rbf

    rf_params = dict()
    rf_params['max_depth'] = 12

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
    sf_params['n_directions'] = 2
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = None

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
    sf_params['max_depth'] = None
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 3
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = 12

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
    sf_params['max_depth'] = None
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 3
    sf_params['sim_function'] = rbf

    rf_params = dict()
    rf_params['max_depth'] = 8

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
    sf_params['n_directions'] = 3
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = None

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
    sf_params['max_depth'] = 14
    sf_params['n_estimators'] = 100
    sf_params['n_directions'] = 3
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = 14

    return X_train, X_test, y_train, y_test, 'segment', sf_params, rf_params


def get_arcene():
    X_train = pd.read_csv('../data/arcene_train.data', sep=' ', header=None)
    X_train = X_train.drop(columns=[10000])
    X_train = X_train.astype(np.int32)
    y_train = pd.read_csv('../data/arcene_train.labels', sep=' ', header=None)
    y_train = LabelBinarizer().fit_transform(y_train).ravel()

    X_test = pd.read_csv('../data/arcene_valid.data', sep=' ', header=None)
    X_test = X_test.drop(columns=[10000])
    X_test = X_test.astype(np.int32)
    y_test = pd.read_csv('../data/arcene_valid.labels', sep=' ', header=None)
    y_test = LabelBinarizer().fit_transform(y_test).ravel()

    # shuffle training data - greed search does not do it automatically
    '''random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]'''

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    sf_params = dict()
    sf_params['n_estimators'] = 100
    sf_params['max_depth'] = None
    sf_params['n_directions'] = 1
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'arcene', sf_params, rf_params


def get_dexter():
    dataset = fetch_openml('Dexter')
    X = dataset.data
    X = np.array(X.todense(), dtype=np.int32)
    y = dataset.target
    X, y = shuffle(X, y, random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    sf_params = dict()
    sf_params['n_estimators'] = 100

    rf_params = dict()

    return X_train, X_test, y_train, y_test, 'dexter', sf_params, rf_params


def get_asian_religions():
    df = pd.read_csv('../data/AsianReligionsData/AllBooks_baseline_DTM_Labelled.csv')
    df['class'] = df['class'].apply(lambda x: x.partition('+')[0])
    y, X = df.pop('class'), df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    sf_params = dict()
    #sf_params['n_estimators'] = 100

    rf_params = dict()

    return X_train, X_test, y_train, y_test, 'asian_religions', sf_params, rf_params


def get_liver_disorders():
    X_train, y_train = load_svmlight_file('../data/liver-disorders')
    X_train = X_train.toarray().astype(np.float32)

    X_test, y_test = load_svmlight_file('../data/liver-disorders.t')
    X_test = X_test.toarray().astype(np.float32)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sf_params = dict()
    sf_params['n_estimators'] = 100

    rf_params = dict()

    return X_train, X_test, y_train, y_test, 'liver-disorders', sf_params, rf_params


def get_leukemia():
    X_train, y_train = load_svmlight_file('../data/leu')
    X_train = X_train.toarray().astype(np.float32)

    X_test, y_test = load_svmlight_file('../data/leu.t')
    X_test = X_test.toarray().astype(np.float32)

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

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
    sf_params['n_estimators'] = 100
    sf_params['max_depth'] = 10
    sf_params['n_directions'] = 2
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = 12

    return X_train, X_test, y_train, y_test, 'leukemia', sf_params, rf_params


def get_fourclass():
    X, y = load_svmlight_file('../data/fourclass')
    X = X.todense()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    '''scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)'''

    sf_params = dict()
    sf_params['n_estimators'] = 100
    sf_params['max_depth'] = 8
    sf_params['n_directions'] = 1
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'fourclass', sf_params, rf_params


def get_duke():
    X, y = load_svmlight_file('../data/duke')
    X = X.todense()

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
    sf_params['n_estimators'] = 100
    sf_params['max_depth'] = 8
    sf_params['n_directions'] = 3
    sf_params['sim_function'] = rbf

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'duke', sf_params, rf_params


def get_colon_cancer():
    X, y = load_svmlight_file('../data/colon-cancer')
    X = X.todense()

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
    sf_params['n_estimators'] = 100
    sf_params['max_depth'] = 8
    sf_params['n_directions'] = 2
    sf_params['sim_function'] = rbf

    rf_params = dict()
    rf_params['max_depth'] = 8

    rf_params = dict()

    return X_train, X_test, y_train, y_test, 'colon-cancer', sf_params, rf_params


def get_gisette():
    X_train, y_train = load_svmlight_file('../data/gisette_scale')
    X_train = X_train.toarray().astype(np.float32)

    X_test, y_test = load_svmlight_file('../data/gisette_scale.t')
    X_test = X_test.toarray().astype(np.float32)

    X = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    sf_params = dict()
    sf_params['n_estimators'] = 100
    sf_params['max_depth'] = None
    sf_params['n_directions'] = 1
    sf_params['sim_function'] = dot_product

    rf_params = dict()
    rf_params['max_depth'] = 8

    return X_train, X_test, y_train, y_test, 'gisette', sf_params, rf_params


'''
    # binary
    get_madelon(),
    get_diabetes(),
    get_australian(),
    get_splice(),
    get_a1a(),
    get_svmguide3(),
    get_heart(),
    get_ionosphere()
    get_breast_cancer(),
    get_german_numer(),
    get_mushrooms(),
    get_liver_disorders()
    
    # multiclass
    get_iris(),
    get_glass(),
    get_seed(),
    get_wine(),
    get_dna(),
    get_segment()

'''

datasets = [
    # very high dimensionality
    get_arcene(),
    #get_asian_religions(),
    get_leukemia(),
    get_duke(),
    get_colon_cancer(),

    # very low dimensionality
    get_fourclass()
]


def get_datasets():
    for d in datasets:
        yield d


#get_datasets()
