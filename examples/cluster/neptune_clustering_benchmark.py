import neptune
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from simforest.cluster import SimilarityForestCluster
from examples.cluster.preprocess_benchmark import preprocess
from sklearn.metrics import davies_bouldin_score, silhouette_score


# select project
neptune.init('sfczekalski/similarity-forest')

n_iterations = 3

# create experiment
neptune.create_experiment(name='clustering_benchmark',
                          params={'n_iterations': n_iterations})


params = dict()
params['n_clusters'] = 20
params['max_depth'] = int(np.ceil(np.log2(20)))

for file_name, X in preprocess():
    forest = SimilarityForestCluster(**params)
    silhouettes = np.zeros(shape=(n_iterations,), dtype=np.float32)
    davies_bouldins = np.zeros(shape=(n_iterations,), dtype=np.float32)
    clusters = np.zeros(shape=(X.shape[0], n_iterations), dtype=np.int32)

    for i in range(n_iterations):
        clusters[:, i] = forest.fit_predict(X)
        silhouettes[i] = silhouette_score(X, clusters[:, i])
        davies_bouldins[i] = davies_bouldin_score(X, clusters[:, i])

    best_indice = np.argmax(silhouettes)

    figure, axs = plt.subplots(1, 1, figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=clusters[:, best_indice], cmap='Set1', alpha=0.8)
    plt.title(file_name)

    neptune.log_metric('Silhouette score', silhouettes[best_indice])
    neptune.log_metric('Davies Bouldin score', davies_bouldins[best_indice])
    neptune.log_image('Plot', plt.gcf())
    plt.clf()
    plt.close()

neptune.set_property('model', 'SimilarityForestClusterer')


neptune.stop()
