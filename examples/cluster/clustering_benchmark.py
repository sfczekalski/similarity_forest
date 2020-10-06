import neptune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from simforest.cluster import SimilarityForestCluster
from sklearn.metrics import davies_bouldin_score, silhouette_score
import hdbscan
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from examples.cluster.clustering_datasets import get_datasets


# init project
neptune.set_project('sfczekalski/SimilarityForest')
neptune.init('sfczekalski/SimilarityForest')

# set model properties
params = dict()
params['max_depth'] = 5
params['n_estimators'] = 20
params['technique'] = 'hdbscan'
params['bootstrap'] = False
params['sim_function'] = 'dot'

# set experiment properties
n_iterations = 20
plot = True
other_algorithm = 'hdbscan'

# create experiment
neptune.create_experiment(name='Clustering - hdbscan',
                          params=params,
                          properties={'n_iterations': n_iterations,
                                      'plot': plot})

# init log
df = pd.DataFrame(columns=['dataset', 'SF silhouette', f'{other_algorithm} silhouette', 'p-val silhouette',
                           'sf davies bouldin', f'{other_algorithm} davies bouldin', 'p-val davies bouldin'])
log_name = 'logs/clustering_log.csv'

for d_idx, (dataset, X, n_clusters) in enumerate(get_datasets()):

    params['n_clusters'] = n_clusters

    # store results
    sf_silhouette = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_db = np.zeros(shape=(n_iterations,), dtype=np.float32)

    other_silhouette = np.zeros(shape=(n_iterations,), dtype=np.float32)
    other_db = np.zeros(shape=(n_iterations,), dtype=np.float32)

    for i in range(n_iterations):
        print(f'{dataset}, {i + 1} / {n_iterations}')

        # fit_predict Similarity Forest
        sf = SimilarityForestCluster(**params)
        try:
            sf_clusters = sf.fit_predict(X)
        except ValueError as err:
            # sometimes all cluster labels are the same
            print(f'ValueError: {err}')
            break
        sf_silhouette[i] = silhouette_score(X, sf_clusters)
        sf_db[i] = davies_bouldin_score(X, sf_clusters)

        # fit_predict second algorithm
        other_clusters = hdbscan.HDBSCAN().fit_predict(X)
        other_silhouette[i] = silhouette_score(X, other_clusters)
        other_db[i] = davies_bouldin_score(X, other_clusters)

        if plot:
            if X.shape[1] > 2:
                X = PCA(random_state=42, n_components=2).fit_transform(X)
            figure, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].scatter(X[:, 0], X[:, 1], c=sf_clusters, cmap='Set1', alpha=0.6)
            axs[0].set_title('SimilarityForestCluster')
            axs[1].scatter(X[:, 0], X[:, 1], c=other_clusters, cmap='Set1', alpha=0.6)
            axs[1].set_title(f'{other_algorithm}')
            neptune.log_image(f'{dataset} Plot', plt.gcf())
            plt.clf()
            plt.close()


    # log results
    sf_mean_silhouette = np.mean(sf_silhouette)
    sf_mean_db = np.mean(sf_db)
    neptune.log_metric(f'{dataset} SF silhouette', sf_mean_silhouette)
    neptune.log_metric(f'{dataset} SF Davies Bouldin', sf_mean_db)

    other_mean_silhouette = np.mean(other_silhouette)
    other_mean_db = np.mean(other_db)
    neptune.log_metric(f'{dataset} {other_algorithm} silhouette', other_mean_silhouette)
    neptune.log_metric(f'{dataset} {other_algorithm} Davies Bouldin', other_mean_db)

    # compare
    ts, ps = ttest_ind(sf_silhouette, other_silhouette)
    neptune.log_metric(f'{dataset} t-stat silhouette', ts)
    neptune.log_metric(f'{dataset} p-val silhouette', ps)

    tdb, pdb = ttest_ind(sf_db, other_db)
    neptune.log_metric(f'{dataset} t-stat db', tdb)
    neptune.log_metric(f'{dataset} p-val db', pdb)

    # log
    df.loc[d_idx] = [dataset, sf_mean_silhouette, other_mean_silhouette, ps, sf_mean_db, other_mean_db, pdb]
    df.to_csv(log_name, index=False)

neptune.log_artifact(log_name)
neptune.stop()
