from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, load_iris, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


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

    return X_train, X_test, y_train, y_test, 'a1a'


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

    return X_train, X_test, y_train, y_test, 'svmguide3'


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

    return X_train, X_test, y_train, y_test, 'heart'


def get_ionosphere():
    X, y = load_svmlight_file('../data/ionosphere_scale')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    return X_train, X_test, y_train, y_test, 'ionosphere_scale'


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

    return X_train, X_test, y_train, y_test, 'breast_cancer'


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

    return X_train, X_test, y_train, y_test, 'german_numer'


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

    return X_train, X_test, y_train, y_test, 'mushrooms'


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

    return X_train, X_test, y_train, y_test, 'iris'


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

    return X_train, X_test, y_train, y_test, 'glass'


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

    return X_train, X_test, y_train, y_test, 'seed'


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

    return X_train, X_test, y_train, y_test, 'wine'


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

    return X_train, X_test, y_train, y_test, 'splice'


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

    return X_train, X_test, y_train, y_test, 'madelon'


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

    return X_train, X_test, y_train, y_test, 'diabetes'


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

    return X_train, X_test, y_train, y_test, 'australian'


def get_dna():
    X, y = load_svmlight_file('../data/dna.scale')
    X = X.toarray().astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    return X_train, X_test, y_train, y_test, 'dna'


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

    return X_train, X_test, y_train, y_test, 'segment'

def get_arcene():

    label_bin = LabelBinarizer()
    X_train = pd.read_csv('../data/arcene_train.data', sep=' ', header=None)
    X_train = X_train.drop(columns=[10000])
    X_train = X_train.astype(np.int32)
    y_train = pd.read_csv('../data/arcene_train.labels', sep=' ', header=None)
    y_train = label_bin.fit_transform(y_train).ravel()

    X_test = pd.read_csv('../data/arcene_valid.data', sep=' ', header=None)
    X_test = X_test.drop(columns=[10000])
    X_test = X_test.astype(np.int32)
    y_test = pd.read_csv('../data/arcene_valid.labels', sep=' ', header=None)
    y_test = label_bin.transform(y_test).ravel()

    X = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))

    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, 'arcene'


def get_dexter():
    dataset = fetch_openml('Dexter')
    X = dataset.data
    X = np.array(X.todense(), dtype=np.int32)
    y = dataset.target
    X, y = shuffle(X, y, random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, 'dexter'


def get_asian_religions():
    df = pd.read_csv('../data/AsianReligionsData/AllBooks_baseline_DTM_Labelled.csv')
    df['class'] = df['class'].apply(lambda x: x.partition('+')[0])
    y, X = df.pop('class'), df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, 'asian_religions'


def get_liver_disorders():
    X_train, y_train = load_svmlight_file('../data/liver-disorders')
    X_train = X_train.toarray().astype(np.float32)

    X_test, y_test = load_svmlight_file('../data/liver-disorders.t')
    X_test = X_test.toarray().astype(np.float32)

    X = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
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

    return X_train, X_test, y_train, y_test, 'liver-disorders'


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

    return X_train, X_test, y_train, y_test, 'leukemia'


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

    return X_train, X_test, y_train, y_test, 'fourclass'


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

    return X_train, X_test, y_train, y_test, 'duke'


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

    return X_train, X_test, y_train, y_test, 'colon-cancer'


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

    return X_train, X_test, y_train, y_test, 'gisette'


datasets = [
    # binary
    #get_heart(),
    get_ionosphere(),
    get_breast_cancer(),
    #get_german_numer(),
    #get_madelon(),
    #get_diabetes(),
    #get_australian(),
    #get_splice(),
    get_a1a(),
    get_svmguide3()
    #get_liver_disorders()

    # multiclass
    #get_iris(),
    #get_glass(),
    #get_seed(),
    #get_wine(),
    #get_dna(),
    #get_segment(),

    # very high dimensionality
    #get_arcene(),
    #get_asian_religions(),
    #get_leukemia(),
    #get_duke(),
    #get_colon_cancer()

    # very low dimensionality
    #get_fourclass()
]


def get_datasets():
    for d in datasets:
        yield d


#get_datasets()
