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
file = loadarff(join(path, 'vehicle.arff'))
df = pd.DataFrame(file[0])
df = fix_dtypes(df)
if df.shape[0] >= 2000:
    df = df.sample(n=2000)

df.drop(columns=['Class'], inplace=True)
X = df.values
X = StandardScaler().fit_transform(X)

csf = SimilarityForestCluster(bootstrap=False)
clusters = csf.fit_predict(X)
'''dendrogram(csf.links_)
plt.show()

pca = PCA(n_components=3, random_state=42).fit_transform(X, y)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], marker='o', c=clusters,
           s=50, alpha=0.7)
ax.set_title('Similarity Forest clusters')
plt.show()'''


'''ahc_clusters = AgglomerativeClustering(n_clusters=3, linkage='single').fit_predict(X)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], marker='o', c=ahc_clusters,
           s=50, alpha=0.7)
ax.set_title('AHC clusters')
plt.show()'''
