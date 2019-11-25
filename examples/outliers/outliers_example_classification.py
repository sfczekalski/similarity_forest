from simforest import SimilarityForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.utils import shuffle as sh

# fetch data
X, y = fetch_kddcup99(subset='http', random_state=42, return_X_y=True)
X, y = X.astype(np.float32), y.astype('str')

# fix classes
y_df = pd.DataFrame(y, columns=['class'])
y_df.loc[y_df['class'] != 'normal.', 'class'] = -1
y_df.loc[y_df['class'] == 'normal.', 'class'] = 1
y = y_df.values

# smtp take all outliers aside
'''outliers_indices = np.where(y == -1)[0]
inliers_indices = np.where(y == 1)[0]
y_outliers = y[outliers_indices]
y = y[inliers_indices]

X_outliers = X[outliers_indices]
X = X[inliers_indices]'''

# kddcup99 SF subset preprocessing
'''X, y = fetch_kddcup99(subset='SF', random_state=42, return_X_y=True)
lb = LabelBinarizer()
x1 = lb.fit_transform(X[:, 1].astype(str))
X = np.c_[X[:, :1], x1, X[:, 2:]]
y = y.astype('str')
y_df = pd.DataFrame(y, columns=['class'])
y_df.loc[y_df['class'] != 'normal.', 'class'] = -1
y_df.loc[y_df['class'] == 'normal.', 'class'] = 1
y = y_df.values'''

# kddcup99 SA subset preprocessing
'''X, y = fetch_kddcup99(subset='SA', random_state=42, return_X_y=True)
lb = LabelBinarizer()
x1 = lb.fit_transform(X[:, 1].astype(str))
x2 = lb.fit_transform(X[:, 2].astype(str))
x3 = lb.fit_transform(X[:, 3].astype(str))
X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
y = y.astype('str')
y_df = pd.DataFrame(y, columns=['class'])
y_df.loc[y_df['class'] != 'normal.', 'class'] = -1
y_df.loc[y_df['class'] == 'normal.', 'class'] = 1
y = y_df.values'''

# shuttle
'''dataset = fetch_openml('shuttle')
X = dataset.data
y = dataset.target
X, y = sh(X, y, random_state=1)
y = y.astype(int)
# we remove data with label 4
# normal data are then those of class 1
s = (y != 4)
X = X[s, :]
y = y[s]
y[(y == 1)] = 1
y[(y != 1)] = -1'''

# forestcover
'''dataset = fetch_covtype(shuffle=True, random_state=1)
X = dataset.data
y = dataset.target
# normal data are those with attribute 2
# abnormal those with attribute 4
s = (y == 2) + (y == 4)
X = X[s, :]
y = y[s]
y[(y == 2)] = 1
y[(y == 4)] = -1'''

# split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

# smtp concateate outliers to test set
'''y_test = np.append(y_test, y_outliers)
X_test = np.append(X_test, X_outliers, axis=0)'''

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# IF
IF = IsolationForest(random_state=42, behaviour='new')
IF.fit(X_train, y_train)
IF_preds = IF.predict(X_test)
IF_decision_f = IF.decision_function(X_test)
print(IF_decision_f[:200])
print(f'IF AUC: {roc_auc_score(y_test, IF_preds)}')
print('IF consfusion matrix:')
print(confusion_matrix(y_test, IF_preds))

# SF
max_samples = 256
max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
SF = SimilarityForestClassifier(sim_function=np.dot, n_estimators=20, random_state=42, bootstrap=False,
                                max_samples=256, max_depth=max_depth, discriminative_sampling=False)
SF.fit(X_train, y_train)
SF_preds = SF.predict_outliers(X_test)
SF_decision_f = SF.decision_function_outliers(X_test)
print(SF_decision_f[:200])
print(f'SF AUC: {roc_auc_score(y_test, SF_decision_f)}')
print(f'SF consfusion matrix:')
print(confusion_matrix(y_test, SF_preds))

