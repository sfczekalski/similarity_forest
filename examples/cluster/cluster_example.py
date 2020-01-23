from simforest.cluster import SimilarityForestCluster
import numpy as np
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import sqeuclidean
from simforest.cluster import SimilarityTreeCluster, SimilarityForestCluster
from sklearn.preprocessing import StandardScaler
from scipy.io.arff import loadarff
import pandas as pd
from os.path import join
from examples.cluster.preprocess_benchmark import fix_dtypes

path = '../data/clustering_benchmark/real-world/'
file = loadarff(join(path, 'ecoli.arff'))
df = pd.DataFrame(file[0])
df = fix_dtypes(df)
if df.shape[0] >= 2000:
    df = df.sample(n=2000)

df.drop(columns=['class'], inplace=True)
X = df.values
X = StandardScaler().fit_transform(X)

params = dict()
params['max_depth'] = 10
params['n_estimators'] = 100
params['technique'] = 'hdbscan'
params['n_clusters'] = 8
params['bootstrap'] = False

csf = SimilarityForestCluster()
clusters = csf.fit_predict(X)

ahc_clusters = AgglomerativeClustering(n_clusters=3, linkage='single').fit_predict(X)

X = PCA(n_components=3, random_state=42).fit_transform(X)
figure, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(X[:, 0], X[:, 1], c=clusters, cmap='Set1', alpha=0.6)
axs[0].set_title('SimilarityForestCluster')
axs[1].scatter(X[:, 0], X[:, 1], c=ahc_clusters, cmap='Set1', alpha=0.6)
axs[1].set_title('KMeans')
plt.show()
