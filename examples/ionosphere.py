from simforest import SimilarityTreeClassifier, SimilarityForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file

X, y = load_svmlight_file('ionosphere_scale')
X = X.toarray()

X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

'''
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
'''

#params = {'n_trees': [20, 30, 40, 50, 60, 70, 80, 90]}
sf = SimilarityForestClassifier(n_trees=100)
sf.fit(X_train, y_train)
sf_pred = sf.predict(X_test)

'''rf = RandomForestClassifier(random_state=42, max_features='sqrt')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(f'Random Forest accuracy score: {accuracy_score(y_test, rf_pred)}')'''
print(f'Similarity Forest accuracy score: {accuracy_score(y_test, sf_pred)}')

