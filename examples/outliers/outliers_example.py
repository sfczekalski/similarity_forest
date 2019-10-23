from simforest import SimilarityForestRegressor, SimilarityTreeRegressor
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X, y = make_regression(n_features=3, n_informative=2, n_samples=1000, random_state=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit
sf = SimilarityForestRegressor(n_estimators=100, n_directions=1)
sf.fit(X_train, y_train)
sf_outlyingness = sf.outlyingness(X_test)


isol_forest = IsolationForest()
isol_forest.fit(X_train, y_train)
isol_outlyingness = isol_forest.score_samples(X_test)


'''
# Plot classifiers' predictions
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=sf_outlyingness,
            s=50, edgecolor='k', alpha=1.0, cmap='BuPu', lw=0, facecolor='0.5')
plt.title("Similarity Forest outlyingness measure")
plt.show()

# Plot classifiers' predictions
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=isol_outlyingness,
            s=50, edgecolor='k', alpha=1.0, cmap='BuPu', lw=0, facecolor='0.5')
plt.title("Isolation Forest outlyingness measure")
plt.show()'''

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=sf_outlyingness, cmap='BuPu')
plt.title("Similarity Forest outlyingness measure")
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=isol_outlyingness, cmap='BuPu')
plt.title("Isolation Forest outlyingness measure")
plt.show()
