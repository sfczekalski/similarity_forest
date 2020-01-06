import neptune
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from simforest.cluster import SimilarityForestCluster
from examples.cluster.preprocess_benchmark import preprocess, fix_dtypes
from sklearn.metrics import davies_bouldin_score, silhouette_score
import hdbscan
from scipy.stats import ttest_ind
from scipy.io.arff import loadarff
from os.path import join
import pandas as pd

path = '../data/clustering_benchmark/real-world/'

datasets = [
    {'file_name': 'glass.arff',
     'class_col': 'Class',
     'n_clusters': 7},
    {'file_name': 'iris.arff',
     'class_col': 'class',
     'n_clusters': 3},
    {'file_name': 'sonar.arff',
     'class_col': 'Class',
     'n_clusters': 2},
    {'file_name': 'wine.arff',
     'class_col': 'class',
     'n_clusters': 3}
]


def get_datasets(datasets):
    for d in datasets:
        yield d['file_name'], d['class_col'], d['n_clusters']


for file_name, class_col, n_clusters in get_datasets(datasets):
    file = loadarff(join(path, file_name))
    df = pd.DataFrame(file[0])
    df = fix_dtypes(df)
    if df.shape[0] >= 2000:
        df = df.sample(n=2000)

    df.drop(columns=[class_col], inplace=True)
    X = df.values

    # select project
    neptune.init('sfczekalski/similarity-forest')

    params = dict()
    params['max_depth'] = None
    params['n_estimators'] = 100
    params['technique'] = 'ahc'
    params['n_clusters'] = n_clusters

    # set experiment properties
    n_iterations = 30

    # create experiment
    neptune.create_experiment(name=f'Clustering {file_name}',
                              params=params,
                              properties={'n_iterations': n_iterations,
                                          'dataset': file_name,
                                          'n_clusters': params['n_clusters']})

    # store results
    sf_silhouette = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_db = np.zeros(shape=(n_iterations,), dtype=np.float32)

    ahc_silhouette = np.zeros(shape=(n_iterations,), dtype=np.float32)
    ahc_db = np.zeros(shape=(n_iterations,), dtype=np.float32)

    for i in range(n_iterations):
        sf = SimilarityForestCluster(**params)
        sf_clusters = sf.fit_predict(X)
        neptune.log_metric('SF Silhouette score', silhouette_score(X, sf_clusters))
        neptune.log_metric('SF Davies Bouldin score', davies_bouldin_score(X, sf_clusters))
        sf_silhouette[i] = silhouette_score(X, sf_clusters)
        sf_db[i] = davies_bouldin_score(X, sf_clusters)

        ahc_clusters = AgglomerativeClustering(n_clusters=params['n_clusters']).fit_predict(X)
        neptune.log_metric('AHC Silhouette score', silhouette_score(X, ahc_clusters))
        neptune.log_metric('AHC Davies Bouldin score', davies_bouldin_score(X, ahc_clusters))
        ahc_silhouette[i] = silhouette_score(X, ahc_clusters)
        ahc_db[i] = davies_bouldin_score(X, ahc_clusters)


    # log results
    neptune.log_metric('SF mean silhouette', np.mean(sf_silhouette))
    neptune.log_metric('SF mean Davies Bouldin', np.mean(sf_db))

    neptune.log_metric('AHC mean silhouette', np.mean(ahc_silhouette))
    neptune.log_metric('AHC mean Davies Bouldin', np.mean(ahc_db))


    # compare
    t, p = ttest_ind(sf_silhouette, ahc_silhouette)
    neptune.log_metric('t-stat silhouette', t)
    neptune.log_metric('p-val silhouette', p)

    t, p = ttest_ind(sf_db, ahc_db)
    neptune.log_metric('t-stat db', t)
    neptune.log_metric('p-val db', p)

    neptune.stop()
