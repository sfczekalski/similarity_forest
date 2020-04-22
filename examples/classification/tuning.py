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
    #'max_depth': [8, 10, 12, 14, None],
    'max_depth': [8, 10, 12],
    'n_directions': [3],
    'sim_function': [rbf],
    #'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'gamma': [0.0000001, 0.000001],
    'n_estimators': [150, 200],
}

ovr_sf_params = {
    'estimator__max_depth': [8, 10, 12, 14, None],
    'estimator__n_directions': [1, 2, 3],
    'estimator__gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'estimator__sim_function': [dot_product]
}

rf_params = {
    'max_depth': [10, 12, 14, None],
    'n_estimators': [25, 50, 75, 100, 200]
}

# init log
df = pd.DataFrame(columns=['dataset',
                           'SF f1',
                           'SF roc-auc',
                           'SF max_depth',
                           'SF n_dir',
                           'SF s_fun',
                           'SF gamma',
                           'SF n_estimators',
                           'RF f1',
                           'RF roc-auc',
                           'RF max_depth',
                           'RF n_estimators'])

binary = False
multiclass_strategy = False

if binary:
    log_name = 'logs/rbf_binary_classification_tuning8.csv'
else:
    log_name = 'logs/rbf_multiclass_classification_tuning8.csv'


# load and prepare data
for d_idx, d in enumerate(get_datasets()):
    X_train, X_test, y_train, y_test, dataset, _, _ = d
    print(dataset)

    if binary:
        scorer = 'f1'
    else:
        scorer = 'f1_weighted'

    # RF
    rf = GridSearchCV(RandomForestClassifier(), param_grid=rf_params, cv=5, scoring=scorer, refit=scorer, n_jobs=4)
    rf.fit(X_train, y_train)
    rf_dec_f = rf.predict_proba(X_test)
    rf_pred = rf.classes_[np.argmax(rf_dec_f, axis=1)]

    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    if binary:
        rf_roc = roc_auc_score(y_test, rf_dec_f[:, 1])
    else:
        rf_roc = roc_auc_score(y_test, rf_dec_f, average='weighted', multi_class='ovr')

    # SF
    if multiclass_strategy:
        ovr = OneVsRestClassifier(SimilarityForestClassifier())
        sf = GridSearchCV(ovr, param_grid=ovr_sf_params, cv=5, verbose=11, n_jobs=4, scoring='f1', refit='f1')
    else:
        sf = GridSearchCV(SimilarityForestClassifier(), param_grid=sf_params, cv=5, verbose=11, n_jobs=4, scoring=scorer, refit=scorer)

    sf.fit(X_train, y_train)
    sf_dec_f = sf.predict_proba(X_test)
    sf_pred = sf.classes_[np.argmax(sf_dec_f, axis=1)]

    sf_f1 = f1_score(y_test, sf_pred, average='weighted')
    if binary:
        sf_roc = roc_auc_score(y_test, sf_dec_f[:, 1])
    else:
        sf_roc = roc_auc_score(y_test, sf_dec_f, average='weighted', multi_class='ovr')

    print(sf.best_params_)

    if multiclass_strategy:
        df_new = pd.DataFrame({'dataset': [dataset],
                               'SF f1': [sf_f1],
                               'SF roc-auc': [sf_roc],
                               'SF max_depth': [sf.best_params_['estimator__max_depth']],
                               'SF n_dir': [sf.best_params_['estimator__n_directions']],
                               'SF s_fun': [sf.best_params_['estimator__sim_function']],
                               'SF gamma': [sf.best_params_['estimator__gamma']],
                               'SF n_estimators': [sf.best_params_['estimator__n_estimators']],
                               'RF f1': [rf_f1],
                               'RF roc-auc': [rf_roc],
                               'RF max_depth': [rf.best_params_['max_depth']],
                               'RF n_estimators': [rf.best_params_['n_estimators']]}, index=[d_idx])
    else:
        df_new = pd.DataFrame({'dataset': [dataset],
                               'SF f1': [sf_f1],
                               'SF roc-auc': [sf_roc],
                               'SF max_depth': [sf.best_params_['max_depth']],
                               'SF n_dir': [sf.best_params_['n_directions']],
                               'SF s_fun': [sf.best_params_['sim_function']],
                               'SF gamma': [sf.best_params_['gamma']],
                               'SF n_estimators': [sf.best_params_['n_estimators']],
                               'RF f1': [rf_f1],
                               'RF roc-auc': [rf_roc],
                               'RF max_depth': [rf.best_params_['max_depth']],
                               'RF n_estimators': [rf.best_params_['n_estimators']]}, index=[d_idx])

    df = df.append(df_new)
    print((df.loc[d_idx]))

    df.to_csv(log_name, index=False)
