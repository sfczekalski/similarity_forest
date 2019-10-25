from simforest import SimilarityTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs, make_classification, load_iris
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#X, y = make_blobs(n_samples=1000, centers=[(0, 0), (1, 1), (1.5, 1)], random_state=42)

"""X, y = make_classification(n_features=4, n_redundant=2, n_informative=2, n_samples=1000,
                           random_state=1, n_clusters_per_class=1, n_classes=2)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)"""

#X, y = load_iris(return_X_y=True)

df = pd.read_csv('data/dataset_glass.csv')
y = df.pop('Type')
X = df

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

# Fit predict for both classifiers
st = OutputCodeClassifier(SimilarityTreeClassifier(sim_function=np.dot, max_depth=None, n_directions=2))
st.fit(X_train, y_train)
st_pred = st.predict(X_test)
#st_prob = st.predict_proba(X_test)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)

# Compare classifiers' accuracy
print(f'Decision Tree accuracy score: {accuracy_score(y_test, dt_pred)}')
print(f'Similarity Tree accuracy score: {accuracy_score(y_test, st_pred)}')

#print(f'Decision Tree log loss: {log_loss(y_test, dt_prob)}')
#print(f'Similarity Tree log loss: {log_loss(y_test, st_prob)}')


'''# Plot classifiers' predictions
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=st_pred,
            s=25, edgecolor='k', alpha=0.5)
plt.title("Similarity Forest")
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=dt_pred,
            s=25, edgecolor='k', alpha=0.5, )
plt.title("Random Forest")
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_test,
            s=25, edgecolor='k', alpha=0.5, )
plt.title("True")
plt.show()'''
