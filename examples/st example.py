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
st = SimilarityTreeClassifier()
st.fit(X_train, y_train)
st_pred = st.predict(X_test)
st_prob = st.predict_proba(X_test)


start_time = time.time()
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)

# Compare classifiers' accuracy
print(f'Decision Tree accuracy score: {accuracy_score(y_test, dt_pred)}')
print(f'Similarity Tree accuracy score: {accuracy_score(y_test, st_pred)}')

print(f'Decision Tree log loss: {log_loss(y_test, dt_prob)}')
print(f'Similarity Tree log loss: {log_loss(y_test, st_prob)}')


# Plot classifiers' predictions
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
plt.show()
