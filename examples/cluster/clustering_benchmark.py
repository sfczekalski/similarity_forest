import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from os import listdir
from os.path import join
from simforest.cluster import SimilarityForestCluster
from scipy.spatial.distance import sqeuclidean
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score


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


def artificial():
    path = '../data/clustering_benchmark/artificial/'
    interesing = [
                  '2d-3c-no123.arff',
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
        try:
            file = loadarff(join(path, file_name))
        except NotImplementedError:
            # Some datasets include string attributes, and loadarff can't handle them
            continue
        df = pd.DataFrame(file[0])
        df = fix_dtypes(df)
        if df.shape[0] >= 2000:
            df = df.sample(n=2000)
        df = fix_col_names(df)
        X = df.values[:, 0:2]

        figure, axs = plt.subplots(2, 2, figsize=(10, 10))

        params = dict()
        params['n_clusters'] = 20
        params['max_depth'] = 5
        params['n_estimators'] = 20
        params['random_state'] = 1
        params['technique'] = 'hdbscan'
        sf = SimilarityForestCluster(**params)
        sf_clusters = sf.fit_predict(X)
        axs[0, 0].scatter(X[:, 0], X[:, 1], c=sf_clusters, cmap='Set1', alpha=0.8)
        axs[0, 0].set_title('SimilarityForestCluster')

        hdb_clusters = hdbscan.HDBSCAN().fit_predict(X)
        axs[0, 1].scatter(X[:, 0], X[:, 1], c=hdb_clusters, cmap='Set1', alpha=0.8)
        axs[0, 1].set_title('HDBSCAN')

        ahc_clusters = AgglomerativeClustering(n_clusters=20).fit_predict(X)
        axs[1, 0].scatter(X[:, 0], X[:, 1], c=ahc_clusters, cmap='Set1', alpha=0.8)
        axs[1, 0].set_title('AHC')

        kmeans_clusters = KMeans(n_clusters=20).fit_predict(X)
        axs[1, 1].scatter(X[:, 0], X[:, 1], c=kmeans_clusters, cmap='Set1', alpha=0.8)
        axs[1, 1].set_title('KMeans')

        plt.show()


def real_world():
    path = '../data/clustering_benchmark/real-world/'
    for file_name in listdir(path):
        try:
            file = loadarff(join(path, file_name))
        except NotImplementedError:
            # Some datasets include string attributes, and loadarff can't handle them
            continue
        df = pd.DataFrame(file[0])
        if 'class' in df.columns:
            class_column = 'class'
            y, df = df.pop(class_column), df
        elif 'Class' in df.columns:
            class_column = 'Class'
            y, df = df.pop(class_column), df
        elif 'CLASS' in df.columns:
            class_column = 'CLASS'
            y, df = df.pop(class_column), df
        else:
            pass

        df.fillna(0, inplace=True)
        df = fix_dtypes(df)
        '''if df.shape[0] >= 1000:
            df = df.sample(n=1000)'''

        if not file_name == 'balance-scale.arff':
            df = StandardScaler().fit_transform(df)

        sf = SimilarityForestCluster()
        clusters = sf.fit_predict(df)

        x = PCA(n_components=2, random_state=42).fit_transform(df)
        plt.scatter(x[:, 0], x[:, 1], c=clusters)
        plt.show()


artificial()
