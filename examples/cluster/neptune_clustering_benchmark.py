import neptune
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from simforest.cluster import SimilarityForestCluster
from examples.cluster.preprocess_benchmark import preprocess
from sklearn.metrics import davies_bouldin_score, silhouette_score
import hdbscan

# select project
neptune.init('sfczekalski/similarity-forest')

params = dict()
params['max_depth'] = None
params['n_estimators'] = 100
params['random_state'] = 1
params['technique'] = 'hdbscan'

# create experiment
neptune.create_experiment(name='clustering_benchmark',
                          params=params)


for file_name, X in preprocess():

    figure, axs = plt.subplots(1, 2, figsize=(10, 5))

    sf = SimilarityForestCluster(**params)
    sf_clusters = sf.fit_predict(X)
    axs[0].scatter(X[:, 0], X[:, 1], c=sf_clusters, cmap='Set1', alpha=0.6)
    axs[0].set_title('SimilarityForestCluster + HDB')
    sf_silhouette = silhouette_score(X, sf_clusters)
    sf_davies_bouldin = davies_bouldin_score(X, sf_clusters)

    hdb_clusters = hdbscan.HDBSCAN().fit_predict(X)
    axs[1].scatter(X[:, 0], X[:, 1], c=hdb_clusters, cmap='Set1', alpha=0.6)
    axs[1].set_title('HDBSCAN')
    hdb_silhouette = silhouette_score(X, hdb_clusters)
    hdb_davies_bouldin = davies_bouldin_score(X, hdb_clusters)


    neptune.log_metric('SF+HDB Silhouette score', sf_silhouette)
    neptune.log_metric('SF+HDB Davies Bouldin score', sf_davies_bouldin)
    neptune.log_metric('HDB Silhouette score', hdb_silhouette)
    neptune.log_metric('HDB Davies Bouldin score', hdb_davies_bouldin)

    neptune.log_image('Plot', plt.gcf())
    plt.clf()
    plt.close()

neptune.set_property('model', 'SimilarityForestClusterer')


neptune.stop()
