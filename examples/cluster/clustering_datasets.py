import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from os.path import join
from os import listdir
from sklearn.preprocessing import LabelEncoder, StandardScaler


def fix_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]
    obj_cols = [c for c in df if df[c].dtype == 'object']
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    try:
        df[obj_cols] = df[obj_cols].astype(np.float32)
    except ValueError:
        for c in obj_cols:
            df[c] = LabelEncoder().fit_transform(df[c])
    return df


def get_artificial_datasets():
    path = '../data/clustering_benchmark/artificial/'
    interesing = ['2d-3c-no123.arff',
                  'dpb.arff',
                  'cure-t2-4k.arff',
                  'sizes3.arff',
                  '2d-10c.arff',
                  'DS-850.arff',
                  'shapes.arff',
                  'shapes.arff',
                  'jain.arff',
                  '2d-4c-no9.arff',
                  'impossible.arff',
                  'hypercube.arff',
                  'hypercube.arff',
                  'zelnik2.arff',
                  'sizes4.arff',
                  'compound.arff',
                  'triangle1.arff',
                  'cure-t0-2000n-2D.arff',
                  'donut3.arff',
                  'rings.arff',
                  'longsquare.arff',
                  'pmf.arff',
                  'zelnik6.arff',
                  'blobs.arff',
                  'ds3c3sc6.arff',
                  'cluto-t7-10k.arff',
                  'chainlink.arff',
                  'square2.arff',
                  'spherical_6_2.arff',
                  'cluto-t4-8k.arff',
                  '2sp2glob.arff',
                  'spiralsquare.arff',
                  'donut2.arff',
                  'diamond9.arff',
                  'dartboard2.arff',
                  'birch-rg2.arff',
                  '2d-4c.arff',
                  '2d-4c-no4.arff',
                  'twodiamonds.arff',
                  'mopsi-finland.arff',
                  'smile1.arff',
                  'long2.arff',
                  'zelnik3.arff',
                  'spherical_4_3.arff',
                  'sizes5.arff'
                  ]
    for file_name in interesing:
        file = loadarff(join(path, file_name))

        df = pd.DataFrame(file[0])
        df = fix_dtypes(df)
        if df.shape[0] >= 2000:
            df = df.sample(n=2000)
        X = df.values[:, 0:2]

        yield file_name, X


def preprocess(X, class_column):
    df = pd.DataFrame(X)
    df = fix_dtypes(df)

    if df.shape[0] >= 2000:
        df = df.sample(n=2000)

    df.drop(columns=[class_column], inplace=True)
    X = df.values
    X = StandardScaler().fit_transform(X)
    return X


def get_german():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'german.arff'
    class_column = 'CLASS'
    n_clusters = 2
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'german', X, n_clusters


def get_balance():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'balance-scale.arff'
    class_column = 'class'
    n_clusters = 3
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'balance', X, n_clusters


def get_iris():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'iris.arff'
    class_column = 'class'
    n_clusters = 3
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'iris', X, n_clusters


def get_vowel():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'vowel.arff'
    class_column = 'Class'
    n_clusters = 11
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'vowel', X, n_clusters


def get_vehicle():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'vehicle.arff'
    class_column = 'Class'
    n_clusters = 4
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'vehicle', X, n_clusters


def get_segment():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'segment.arff'
    class_column = 'class'
    n_clusters = 7
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'segment', X, n_clusters


def get_zoo():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'zoo.arff'
    class_column = 'class'
    n_clusters = 7
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'zoo', X, n_clusters


def get_cpu():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'cpu.arff'
    class_column = 'class'
    n_clusters = 20
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'cpu', X, n_clusters


def get_ecoli():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'ecoli.arff'
    class_column = 'class'
    n_clusters = 8
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'ecoli', X, n_clusters


def get_glass():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'glass.arff'
    class_column = 'Class'
    n_clusters = 7
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'glass', X, n_clusters


def get_sonar():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'sonar.arff'
    class_column = 'Class'
    n_clusters = 2
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'sonar', X, n_clusters


def get_wine():
    path = '../data/clustering_benchmark/real-world/'
    file_name = 'wine.arff'
    class_column = 'class'
    n_clusters = 3
    file = loadarff(join(path, file_name))
    X = preprocess(file[0], class_column)

    return 'wine', X, n_clusters


datasets = [
    get_glass(),
    get_iris(),
    get_cpu(),
    get_ecoli(),
    get_segment(),
    get_vehicle(),
    get_wine(),
    get_zoo()
]


def get_datasets():
    for d in datasets:
        yield d
