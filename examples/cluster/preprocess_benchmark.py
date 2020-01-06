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


def fix_col_names(df):
    if df.shape[1] == 3:
        df.columns = ['x1', 'x2', 'y']
    elif df.shape[1] == 4:
        df.columns = ['x1', 'x2', 'x3', 'y']
    elif df.shape[1] == 2:
        df.columns = ['x1', 'x2']
    else:
        print(f'DF shape: {df.shape}')
        print(df.head())

    return df


def preprocess():
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
        df = fix_col_names(df)
        X = df.values[:, 0:2]

        yield file_name, X


def preprocess_real_world():
    path = '../data/clustering_benchmark/real-world/'

    for file_name in listdir(path):
        try:
            file = loadarff(join(path, file_name))
        except NotImplementedError:
            continue

        df = pd.DataFrame(file[0])
        df = fix_dtypes(df)
        if df.shape[0] >= 2000:
            df = df.sample(n=2000)
        if df.shape[1] > 35:
            continue
        df = fix_col_names(df)
        print(df.head())
        X = df.values

        yield file_name, X


'''path = '../data/clustering_benchmark/real-world/'
file_name = 'sonar.arff'
file = loadarff(join(path, file_name))
df = pd.DataFrame(file[0])
df = fix_dtypes(df)
if df.shape[0] >= 2000:
    df = df.sample(n=2000)

print(df.columns)
print(np.unique(df['Class']))'''
