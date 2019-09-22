from simforest import SimilarityTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs, load_iris, make_classification
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(n_samples=1000, centers=[(0, 0), (1, 1)], random_state=42)
X, y = load_iris(return_X_y=True)
'''X, y = make_classification(n_features=4, n_redundant=1, n_informative=2, n_samples=1000,
                           random_state=1, n_clusters_per_class=1, n_classes=2)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)'''


X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42)


# Fit predict for both classifiers
#st = OneVsOneClassifier(SimilarityTreeClassifier())
st = SimilarityTreeClassifier()
st.fit(X_train, y_train)
st_pred = st.predict(X_test)
st_prob = st.predict_proba(X_test)

#print(st.estimators_)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)

# Compare classifiers' accuracy
print(f'Similarity Tree accuracy score: {accuracy_score(y_test, st_pred)}')
print(f'Decision Tree accuracy score: {accuracy_score(y_test, dt_pred)}')


print(f'Similarity Tree log loss: {log_loss(y_test, st_prob)}')
print(f'Decision Tree log loss: {log_loss(y_test, dt_prob)}')

#st_log_probs = st.predict_log_proba(X_test)


# Plot classifiers' predictions
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=st_pred,
            s=25, edgecolor='k', alpha=0.5)
plt.title("Similarity Tree")
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=dt_pred,
            s=25, edgecolor='k', alpha=0.5, )
plt.title("Decision Tree")
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_test,
            s=25, edgecolor='k', alpha=0.5, )
plt.title("True")
#plt.show()
