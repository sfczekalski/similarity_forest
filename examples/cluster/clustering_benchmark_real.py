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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path = '../data/clustering_benchmark/real-world/'

'''
    {'file_name': 'glass.arff',
     'class_col': 'Class',
     'n_clusters': 7},
    {'file_name': 'sonar.arff',
     'class_col': 'Class',
     'n_clusters': 2},
    {'file_name': 'wine.arff',
     'class_col': 'class',
     'n_clusters': 3},
    {'file_name': 'balance-scale.arff',
     'class_col': 'class',
     'n_clusters': 3},
    {'file_name': 'german.arff',
     'class_col': 'CLASS',
     'n_clusters': 2},
    
    {'file_name': 'segment.arff',
     'class_col': 'class',
     'n_clusters': 7},
    {'file_name': 'vehicle.arff',
     'class_col': 'Class',
     'n_clusters': 4},
    
'''

datasets = [

    {'file_name': 'iris.arff',
     'class_col': 'class',
     'n_clusters': 3},
    {'file_name': 'vowel.arff',
     'class_col': 'Class',
     'n_clusters': 11},
    {'file_name': 'zoo.arff',
     'class_col': 'class',
     'n_clusters': 7},
    {'file_name': 'cpu.arff',
     'class_col': 'class',
     'n_clusters': 20},
    {'file_name': 'ecoli.arff',
     'class_col': 'class',
     'n_clusters': 8}

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
    X = StandardScaler().fit_transform(X)

    # select project
    neptune.init('sfczekalski/similarity-forest')

    params = dict()
    params['max_depth'] = None
    params['n_estimators'] = 1
    params['technique'] = 'hdbscan'
    params['n_clusters'] = n_clusters

    # set experiment properties
    n_iterations = 20
    plot = True

    # create experiment
    neptune.create_experiment(name=f'Clustering {file_name} plot dot hdbscan',
                              params=params,
                              properties={'n_iterations': n_iterations,
                                          'dataset': file_name,
                                          'n_clusters': params['n_clusters']})

    # store results
    sf_silhouette = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_db = np.zeros(shape=(n_iterations,), dtype=np.float32)

    hdb_silhouette = np.zeros(shape=(n_iterations,), dtype=np.float32)
    hdb_db = np.zeros(shape=(n_iterations,), dtype=np.float32)

    for i in range(n_iterations):
        sf = SimilarityForestCluster(**params)
        sf_clusters = sf.fit_predict(X)
        neptune.log_metric('SF Silhouette score', silhouette_score(X, sf_clusters))
        neptune.log_metric('SF Davies Bouldin score', davies_bouldin_score(X, sf_clusters))
        sf_silhouette[i] = silhouette_score(X, sf_clusters)
        sf_db[i] = davies_bouldin_score(X, sf_clusters)

        hdb_clusters = hdbscan.HDBSCAN().fit_predict(X)
        neptune.log_metric('HDBSCAN Silhouette score', silhouette_score(X, hdb_clusters))
        neptune.log_metric('HDBSCAN Davies Bouldin score', davies_bouldin_score(X, hdb_clusters))
        hdb_silhouette[i] = silhouette_score(X, hdb_clusters)
        hdb_db[i] = davies_bouldin_score(X, hdb_clusters)

        if plot:
            if X.shape[1] > 2:
                X = PCA(random_state=42).fit_transform(X)
            figure, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].scatter(X[:, 0], X[:, 1], c=sf_clusters, cmap='Set1', alpha=0.6)
            axs[0].set_title('SimilarityForestCluster')
            axs[1].scatter(X[:, 0], X[:, 1], c=hdb_clusters, cmap='Set1', alpha=0.6)
            axs[1].set_title('HDBSCAN')
            neptune.log_image('Plot', plt.gcf())
            plt.clf()
            plt.close()


    # log results
    neptune.log_metric('SF mean silhouette', np.mean(sf_silhouette))
    neptune.log_metric('SF mean Davies Bouldin', np.mean(sf_db))

    neptune.log_metric('HDBSCAN mean silhouette', np.mean(hdb_silhouette))
    neptune.log_metric('HDBSCAN mean Davies Bouldin', np.mean(hdb_db))


    # compare
    t, p = ttest_ind(sf_silhouette, hdb_silhouette)
    neptune.log_metric('t-stat silhouette', t)
    neptune.log_metric('p-val silhouette', p)

    t, p = ttest_ind(sf_db, hdb_db)
    neptune.log_metric('t-stat db', t)
    neptune.log_metric('p-val db', p)

    neptune.stop()
