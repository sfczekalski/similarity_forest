import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from os import listdir
from os.path import join
from simforest.cluster import SimilarityForestCluster
from scipy.spatial.distance import sqeuclidean
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array
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
    if df.shape[0] >= 1000:
        df = df.sample(n=1000)

    if not file_name == 'balance-scale.arff':
        df = StandardScaler().fit_transform(df)

    sf = SimilarityForestCluster(random_state=42, max_depth=15, sim_function=sqeuclidean)
    clusters = sf.fit_predict(df)

    print([i.get_depth() for i in sf.estimators_])

    x = PCA(n_components=2, random_state=42).fit_transform(df)
    plt.scatter(x[:, 0], x[:, 1], c=clusters)
    plt.show()
