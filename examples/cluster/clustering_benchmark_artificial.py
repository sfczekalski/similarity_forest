import neptune
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from simforest.cluster import SimilarityForestCluster
from examples.cluster.preprocess_benchmark import preprocess, preprocess_real_world, fix_dtypes
from sklearn.metrics import davies_bouldin_score, silhouette_score
import hdbscan

# select project
neptune.init('sfczekalski/similarity-forest')

params = dict()
params['max_depth'] = None
params['n_estimators'] = 100
params['random_state'] = 42
params['technique'] = 'ahc'
params['n_clusters'] = 5

# create experiment
neptune.create_experiment(name='Clustering real world',
                          params=params)

plot = False

for file_name, X in preprocess_real_world():

    sf = SimilarityForestCluster(**params)
    sf_clusters = sf.fit_predict(X)
    neptune.log_metric('SF Silhouette score', silhouette_score(X, sf_clusters))
    neptune.log_metric('SF Davies Bouldin score', davies_bouldin_score(X, sf_clusters))

    hdb_clusters = hdbscan.HDBSCAN().fit_predict(X)
    neptune.log_metric('HDB Silhouette score', silhouette_score(X, hdb_clusters))
    neptune.log_metric('HDB Davies Bouldin score', davies_bouldin_score(X, hdb_clusters))

    ahc_clusters = AgglomerativeClustering(n_clusters=5).fit_predict(X)
    neptune.log_metric('AHC Silhouette score', silhouette_score(X, ahc_clusters))
    neptune.log_metric('AHC Davies Bouldin score', davies_bouldin_score(X, ahc_clusters))

    kmeans_clusters = KMeans(random_state=42, n_clusters=5).fit_predict(X)
    neptune.log_metric('AHC Silhouette score', silhouette_score(X, kmeans_clusters))
    neptune.log_metric('AHC Davies Bouldin score', davies_bouldin_score(X, kmeans_clusters))

    if plot:
        figure, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].scatter(X[:, 0], X[:, 1], c=sf_clusters, cmap='Set1', alpha=0.6)
        axs[0].set_title('SimilarityForestCluster')
        axs[1].scatter(X[:, 0], X[:, 1], c=hdb_clusters, cmap='Set1', alpha=0.6)
        axs[1].set_title('HDBSCAN')
        neptune.log_image('Plot', plt.gcf())
        plt.clf()
        plt.close()


neptune.set_property('model', 'SimilarityForestClusterer')
neptune.stop()
