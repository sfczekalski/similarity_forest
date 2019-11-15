from simforest import SimilarityForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
from scipy.spatial import distance

# prepare data
X, y = fetch_kddcup99(subset='http', random_state=42, return_X_y=True)
X, y = X.astype(np.float32), y.astype('str')
y_df = pd.DataFrame(y, columns=['class'])
y_df.loc[y_df['class'] != 'normal.', 'class'] = 'attack'
y = y_df.values
y_code = np.array([1 if yi == 'attack' else 0 for yi in y])

'''idxs = np.random.choice(range(len(y)), 100000)
X, y = X[idxs], y[idxs]'''

# split
X_train, X_test, y_train, y_test = train_test_split(
        X, y_code, test_size=0.3, random_state=42)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# IF
IF = IsolationForest(random_state=42, max_samples=256, behaviour='new')
IF.fit(X_train, y_train)
IF_preds = IF.decision_function(X_test)
IF_scores = IF.score_samples(X_test)
print(f'IF max depth: {IF.get_params()}')
print(f'IsolationForest AUC score: {roc_auc_score(y_test, IF_preds)}')
#print(f'IsolationForest confusion matrix: {confusion_matrix(y_test, IF_preds)}')
print(f'IF depths: {np.array([t.get_depth() for t in IF.estimators_])}')
#print(f'IF predictions: {IF_preds[:1000]}')
#print(f'IF scores: {IF_scores[:1000]}')

# SF
max_depth = int(np.ceil(np.log2(y.size)))
SF = SimilarityForestClassifier(sim_function=distance.euclidean, n_estimators=20, random_state=42, bootstrap=False,
                                max_samples=256, max_depth=max_depth, discriminative_sampling=False)
SF.fit(X_train, y_train)
SF_preds = SF.decision_function_outliers(X_test)
print(f'SimilarityForest AUC score: {roc_auc_score(y_test, SF_preds)}')
#print(f'SimilarityForest confusion matrix: {confusion_matrix(y_test, SF_preds)}')
print(f'SF depths: {np.array([t.get_depth() for t in SF.estimators_])}')
print(SF_preds[:1000])
print(f'std: {np.std(SF_preds)}')

print(f"Outlierów: {np.where(y == 'attack')[0].size}")
