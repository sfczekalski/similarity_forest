from simforest import SimilarityTreeClassifier, SimilarityForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=1000, centers=[(0, 0), (1, 1), (1.5, 1)], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

# Fit predict for both classifiers
sf = SimilarityForestClassifier(n_trees=100)
sf.fit(X_train, y_train)
sf_pred = sf.predict(X_test)
sf_prob = sf.predict_proba(X_test)


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)


# Compare classifiers' accuracy
print(f'Random Forest accuracy score: {accuracy_score(y_test, rf_pred)}')
print(f'Similarity Forest accuracy score: {accuracy_score(y_test, sf_pred)}')

print(f'Similarity Forest log loss: {log_loss(y_test, sf_prob)}')
print(f'Random Forest log loss: {log_loss(y_test, rf_prob)}')


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
