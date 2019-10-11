from simforest import SimilarityTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression, load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt


X, y = make_regression(n_features=2, n_informative=1, n_samples=1000, random_state=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
'''X, y = load_svmlight_file('../data/mpg')
X = X.toarray()'''

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit predict for both classifiers
st = SimilarityTreeRegressor(sim_function=np.dot, max_depth=None, n_directions=1)
st.fit(X_train, y_train)
st_pred = st.predict(X_test)
print(f'Similarity Tree R2 score: {r2_score(y_test, st_pred)}')
print(f'Similarity Tree depth: {st.get_depth()}')


dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Compare regressors' accuracy
print(f'Decision Tree R2 score: {r2_score(y_test, dt_pred)}')
print(f'Decison Tree depth: {dt.get_depth()}')

# Scale predictions for plotting
st_pred = (st_pred - np.min(st_pred))/np.ptp(st_pred)
dt_pred = (dt_pred - np.min(dt_pred))/np.ptp(dt_pred)

# Plot classifiers' predictions
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=st_pred,
            s=50, edgecolor='k', alpha=1.0, cmap='BuPu', lw=0, facecolor='0.5')
plt.title("Similarity Tree")
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=dt_pred,
            s=50, edgecolor='k', alpha=1.0, cmap='BuPu', lw=0, facecolor='0.5')
plt.title("Decision Tree")
plt.show()
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_test,
            s=50, edgecolor='k', alpha=1.0, cmap='BuPu', lw=0, facecolor='0.5')
plt.title("True")
plt.show()
