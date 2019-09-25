from simforest import SimilarityTreeClassifier, SimilarityForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs, load_iris, make_classification
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

#X, y = make_blobs(n_samples=1000, centers=[(0, 0), (1, 1), (1.5, 1)], random_state=42)
#X, y = load_iris(return_X_y=True)
X, y = make_classification(n_features=4, n_redundant=2, n_informative=2, n_samples=1000,
                           random_state=1, n_clusters_per_class=1, n_classes=2)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)


X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

# Fit predict for both classifiers

import time
start_time = time.time()
#sf = SimilarityForestClassifier(n_trees=100, n_directions=1)
#sf = OneVsOneClassifier(SimilarityForestClassifier(n_trees=100, n_directions=1))

#sf = OutputCodeClassifier(SimilarityForestClassifier(n_trees=100, n_directions=3), random_state=42, n_jobs=3)

from sklearn.model_selection import GridSearchCV
params = {
    'n_trees': [10, 20, 30, 40, 50, 60, 70, 80, 90]
}
sf = GridSearchCV(SimilarityForestClassifier(),
                  param_grid=params, cv=5)

sf.fit(X_train, y_train)
sf_pred = sf.predict(X_test)
#sf_prob = sf.predict_proba(X_test)
elapsed_time = time.time() - start_time
print(f'Time elapsed: {elapsed_time}')
print(f'Similarity Forest accuracy score: {accuracy_score(y_test, sf_pred)}')

'''
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)

# Compare classifiers' accuracy

print(f'Random Forest accuracy score: {accuracy_score(y_test, rf_pred)}')


#print(f'Similarity Forest log loss: {log_loss(y_test, sf_prob)}')
#print(f'Random Forest log loss: {log_loss(y_test, rf_prob)}')

#st_log_probs = st.predict_log_proba(X_test)


# Plot classifiers' predictions
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=sf_pred,
            s=25, edgecolor='k', alpha=0.5)
plt.title("Similarity Forest")
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=rf_pred,
            s=25, edgecolor='k', alpha=0.5, )
plt.title("Random Forest")
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_test,
            s=25, edgecolor='k', alpha=0.5, )
plt.title("True")
plt.show()
'''