import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, make_scorer, roc_auc_score
from scipy.stats import ttest_ind
import neptune
from sklearn.ensemble import RandomForestClassifier
from simforest import SimilarityForestClassifier
from simforest.distance import rbf, dot_product
from examples.classification.classification_datasets import get_datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV


sf_params = {
    'max_depth': [8, 10, 12, 14, None],
    'n_directions': [1, 2, 3],
    'sim_function': [rbf, dot_product]
}
rf_params = {
    'max_depth': [8, 10, 12, 14, None]
}

# init log
df = pd.DataFrame(columns=['dataset',
                           'SF f1', 'SF roc-auc', 'SF params',
                           'RF f1', 'RF roc-auc', 'RF params'])

log_name = 'logs/multiclass_classification_tuning.csv'

binary = False

# load and prepare data
for d_idx, d in enumerate(get_datasets()):
    X_train, X_test, y_train, y_test, dataset = d
    print(dataset)

    # shuffle training data - greed search does not do it automatically
    random_state = np.random.RandomState(42)
    shuffled_indices = random_state.permutation(len(y_train))
    X_train, y_train = X_train[shuffled_indices], y_train[shuffled_indices]

    # RF
    rf = GridSearchCV(RandomForestClassifier(), param_grid=rf_params, cv=3)
    rf.fit(X_train, y_train)
    rf_dec_f = rf.predict_proba(X_test)
    rf_pred = rf.classes_[np.argmax(rf_dec_f, axis=1)]

    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    if binary:
        rf_roc = roc_auc_score(y_test, rf_dec_f[:, 1])
    else:
        rf_roc = roc_auc_score(y_test, rf_dec_f, average='weighted', multi_class='ovr')

    # SF
    sf = GridSearchCV(SimilarityForestClassifier(), param_grid=sf_params, cv=3)
    sf.fit(X_train, y_train)
    sf_dec_f = sf.predict_proba(X_test)
    sf_pred = sf.classes_[np.argmax(sf_dec_f, axis=1)]

    sf_f1 = f1_score(y_test, sf_pred, average='weighted')
    if binary:
        sf_roc = roc_auc_score(y_test, sf_dec_f[:, 1])
    else:
        sf_roc = roc_auc_score(y_test, sf_dec_f, average='weighted', multi_class='ovr')

    df.loc[d_idx] = [dataset, sf_f1, sf_roc, sf.best_params_, rf_f1, rf_roc, rf.best_params_]
    print(df.loc[d_idx])
    df.to_csv(log_name, index=False)
