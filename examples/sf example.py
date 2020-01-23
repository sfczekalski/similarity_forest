from simforest import SimilarityTreeClassifier, SimilarityForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs, load_svmlight_file
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

#X, y = make_blobs(n_samples=1000, centers=[(0, 0), (1, 1), (1.5, 1)], random_state=42)

X, y = load_svmlight_file('data/a1a')
X = X.toarray()


X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42)

'''scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)'''

# Fit predict for both classifiers
sf = SimilarityForestClassifier(n_estimators=100, random_state=42)
sf.fit(X_train, y_train)
sf_pred = sf.predict(X_test)
sf_pred_train = sf.predict(X_train)
sf_prob = sf.predict_proba(X_test)


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_pred_train = rf.predict(X_train)
rf_prob = rf.predict_proba(X_test)


# Compare classifiers' accuracy
print(f'Random Forest accuracy score: {accuracy_score(y_test, rf_pred)}')
print(f'Similarity Forest accuracy score: {accuracy_score(y_test, sf_pred)}')


print(f'Similarity Forest log loss: {log_loss(y_test, sf_prob)}')
print(f'Random Forest log loss: {log_loss(y_test, rf_prob)}')

# Plot classifiers' predictions
'''plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=sf_pred,
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
plt.show()'''
