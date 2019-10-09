from simforest import SimilarityTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

#X, y = make_blobs(n_samples=1000, centers=[(0, 0), (1, 1), (1.5, 1)], random_state=42)

X, y = make_regression(n_features=4, n_informative=2, n_samples=1000, random_state=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

# Fit predict for both classifiers
st = SimilarityTreeRegressor(sim_function=np.dot, max_depth=None, n_directions=1)
st.fit(X_train, y_train)
st_pred = st.predict(X_test)
print(f'Similarity Tree R2 score: {r2_score(y_test, st_pred)}')
print(f'Similarity Tree depth: {st.get_depth()}')


dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Compare classifiers' accuracy
print(f'Decision Tree R2 score: {r2_score(y_test, dt_pred)}')
print(f'Decison Tree depth: {dt.get_depth()}')
