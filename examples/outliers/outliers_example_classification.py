from simforest import SimilarityForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from scipy.spatial import distance

# prepare data
X, y = fetch_kddcup99(subset='http', random_state=42, return_X_y=True)
X, y = X.astype(np.float32), y.astype('str')
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
IF = IsolationForest(random_state=42, behaviour='new')
IF.fit(X_train, y_train)
IF_preds = IF.predict(X_test)
print('IF consfusion matrix:')
print(confusion_matrix(y_test, IF_preds))

# SF
max_samples = 256
max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
SF = SimilarityForestClassifier(sim_function=distance.euclidean, n_estimators=20, random_state=42, bootstrap=False,
                                max_samples=max_samples, max_depth=None, discriminative_sampling=False)
SF.fit(X_train, y_train)
SF_preds = SF.predict_outliers(X_test)
print(f'SF consfusion matrix:')
print(confusion_matrix(y_test, SF_preds))

