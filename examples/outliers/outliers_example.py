from simforest.isolation_simforest import IsolationSimilarityForest
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_openml, load_svmlight_file
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd

# fetch data
X, y = fetch_kddcup99(subset='http', random_state=42, return_X_y=True)
X, y = X.astype(np.float32), y.astype('str')

# fix classes
y_df = pd.DataFrame(y, columns=['class'])
y_df.loc[y_df['class'] != 'normal.', 'class'] = -1
y_df.loc[y_df['class'] == 'normal.', 'class'] = 1
y = y_df.values

# split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# IF
IF = IsolationForest()
IF.fit(X_train, y_train)
IF_preds = IF.predict(X_test)
IF_decision_f = IF.decision_function(X_test)
print(f'IF AUC: {roc_auc_score(y_test, IF_preds)}')
print('IF consfusion matrix:')
print(confusion_matrix(y_test, IF_preds))

# SF
SF = IsolationSimilarityForest(n_estimators=100)
SF.fit(X_train, y_train)
SF_preds = SF.predict(X_test)
SF_decision_f = SF.decision_function(X_test)
print(f'SF AUC: {roc_auc_score(y_test, SF_decision_f)}')
print(f'SF consfusion matrix:')
print(confusion_matrix(y_test, SF_preds))
