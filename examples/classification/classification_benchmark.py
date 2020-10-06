import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.stats import ttest_ind
import neptune
from sklearn.ensemble import RandomForestClassifier
from simforest import SimilarityForestClassifier
from simforest.distance import rbf, dot_product
from examples.classification.classification_datasets import get_datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV


use_neptune = True
binary = True
multiclass_strategy = False

sf_params = {
    'max_depth': [10, 12, 14, None],
    'n_directions': [1, 2, 3],
    'sim_function': [dot_product],
    #'gamma': [0.0001, 0.001, 0.01, 0.1],
    'n_estimators': [25, 50, 100]
    
}

ovr_sf_params = {
    'estimator__max_depth': [10, 12, 14, None],
    'estimator__n_directions': [1, 2, 3],
    'estimator__gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'estimator__sim_function': [dot_product]
}

rf_params = {
    'max_depth': [10, 12, 14, None],
    'n_estimators': [25, 50, 100],
    'min_samples_split': [2, 3, 4]
}

if use_neptune:
    neptune.set_project('sfczekalski/SimilarityForest')
    neptune.init('sfczekalski/SimilarityForest')


# set experiment properties
n_iterations = 20
n_folds = 10


# init log
df = pd.DataFrame(columns=['dataset',
                           'SF f1', 'SF roc-auc', 'SF acc', 'std SF f1', 'std SF roc-auc', 'std SF acc',
                           'RF f1', 'RF roc-auc', 'RF acc', 'std RF f1', 'std RF roc-auc', 'std RF acc',
                           'p-val f1', 'p-val roc-auc', 'p-val acc',
                           'sf params', 'rf params'])

df_std_cv_score = pd.DataFrame(columns=['dataset', 'SF cv score std', 'RF cv score std'])

if binary:
    log_name = 'logs/classification_results_variance'
    # create experiment
    if use_neptune:
        neptune.create_experiment(name='classification results variance',
                                  properties={'n_iterations': n_iterations, 'n_folds': n_folds})
else:
    log_name = 'logs/classification_results_variance'
    # create experiment
    if use_neptune:
        neptune.create_experiment(name='classification results variance',
                                  properties={'n_iterations': n_iterations, 'n_folds': n_folds})


# load and prepare data
for d_idx, d in enumerate(get_datasets()):
    X_train, X_test, y_train, y_test, dataset = d

    if binary:
        scorer = 'f1'
    else:
        scorer = 'f1_weighted'

    # Find parameters
    # RF
    rf = GridSearchCV(RandomForestClassifier(), param_grid=rf_params, cv=5, scoring=scorer, refit=scorer, n_jobs=4)
    rf.fit(X_train, y_train)

    # SF
    if multiclass_strategy:
        ovr = OneVsRestClassifier(SimilarityForestClassifier())
        sf = GridSearchCV(ovr, param_grid=ovr_sf_params, cv=n_folds, verbose=5, n_jobs=4, scoring='f1', refit='f1')
    else:
        sf = GridSearchCV(SimilarityForestClassifier(), param_grid=sf_params, cv=n_folds, verbose=5, n_jobs=4, scoring=scorer, refit=scorer)

    sf.fit(X_train, y_train)

    # Log std of CV scores for different parameters
    rf_std_score = np.std(rf.cv_results_['mean_test_score'])
    if use_neptune:
        neptune.log_metric(f'{dataset} RF std cv score', rf_std_score)

    sf_std_score = np.std(sf.cv_results_['mean_test_score'])
    if use_neptune:
        neptune.log_metric(f'{dataset} SF std cv score', sf_std_score)

    df_std_cv_score.loc[d_idx] = [dataset, sf_std_score, rf_std_score]
    print(df_std_cv_score.loc[d_idx])
    df_std_cv_score.to_csv(log_name + '_cv_std_score.csv', index=False)

    # Get best params found
    sf_best_params = sf.best_params_
    rf_best_params = rf.best_params_

    # Prepare for test scoring
    rf_f1 = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_f1 = np.zeros(shape=(n_iterations,), dtype=np.float32)

    rf_roc = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_roc = np.zeros(shape=(n_iterations,), dtype=np.float32)

    rf_acc = np.zeros(shape=(n_iterations,), dtype=np.float32)
    sf_acc = np.zeros(shape=(n_iterations,), dtype=np.float32)

    # Score models
    for i in range(n_iterations):
        print(f'{dataset}, {i+1} / {n_iterations}')

        rf = RandomForestClassifier(**rf_best_params)
        rf.fit(X_train, y_train)
        rf_dec_f = rf.predict_proba(X_test)
        rf_pred = rf.classes_[np.argmax(rf_dec_f, axis=1)]

        rf_f1[i] = f1_score(y_test, rf_pred, average='weighted')
        if binary:
            rf_roc[i] = roc_auc_score(y_test, rf_dec_f[:, 1])
        else:
            rf_roc[i] = roc_auc_score(y_test, rf_dec_f, average='weighted', multi_class='ovr')
        rf_acc[i] = accuracy_score(y_test, rf_pred)

        sf = SimilarityForestClassifier(**sf_best_params)
        sf.fit(X_train, y_train)
        sf_dec_f = sf.predict_proba(X_test)
        sf_pred = sf.classes_[np.argmax(sf_dec_f, axis=1)]

        sf_f1[i] = f1_score(y_test, sf_pred, average='weighted')
        if binary:
            sf_roc[i] = roc_auc_score(y_test, sf_dec_f[:, 1])
        else:
            sf_roc[i] = roc_auc_score(y_test, sf_dec_f, average='weighted', multi_class='ovr')
        sf_acc[i] = accuracy_score(y_test, sf_pred)

    # Average results across n_iterations runs
    mean_rf_f1 = np.mean(rf_f1)
    mean_sf_f1 = np.mean(sf_f1)

    mean_rf_roc = np.mean(rf_roc)
    mean_sf_roc = np.mean(sf_roc)

    mean_rf_acc = np.mean(rf_acc)
    mean_sf_acc = np.mean(sf_acc)

    # Calculate std across the runs
    std_rf_f1 = np.std(rf_f1)
    std_sf_f1 = np.std(sf_f1)

    std_rf_roc = np.std(rf_roc)
    std_sf_roc = np.std(sf_roc)

    std_rf_acc = np.std(rf_acc)
    std_sf_acc = np.std(sf_acc)

    # log results
    if use_neptune:
        # Metrics
        neptune.log_metric(f'{dataset} RF f1', mean_rf_f1)
        neptune.log_metric(f'{dataset} SF f1', mean_sf_f1)

        neptune.log_metric(f'{dataset} RF roc-auc', mean_rf_roc)
        neptune.log_metric(f'{dataset} SF roc-auc', mean_sf_roc)

        neptune.log_metric(f'{dataset} RF accuracy', mean_rf_acc)
        neptune.log_metric(f'{dataset} SF accuracy', mean_sf_acc)

        # Std
        neptune.log_metric(f'std {dataset} RF f1', std_rf_f1)
        neptune.log_metric(f'std {dataset} SF f1', std_sf_f1)

        neptune.log_metric(f'std {dataset} RF roc-auc', std_rf_roc)
        neptune.log_metric(f'std {dataset} SF roc-auc', std_sf_roc)

        neptune.log_metric(f'std {dataset} RF accuracy', std_rf_acc)
        neptune.log_metric(f'std {dataset} SF accuracy', std_sf_acc)

    t_f1, p_f1 = ttest_ind(rf_f1, sf_f1)
    t_roc, p_roc = ttest_ind(rf_roc, sf_roc)
    t_acc, p_acc = ttest_ind(rf_acc, sf_acc)

    if use_neptune:
        neptune.log_metric(f'{dataset} t-stat', t_f1)
        neptune.log_metric(f'{dataset} p-val', p_f1)

        neptune.log_metric(f'{dataset} t-stat', t_roc)
        neptune.log_metric(f'{dataset} p-val', p_roc)

        neptune.log_metric(f'{dataset} t-stat', t_acc)
        neptune.log_metric(f'{dataset} p-val', p_acc)

    df.loc[d_idx] = [dataset,
                     mean_sf_f1, mean_sf_roc, mean_sf_acc, std_sf_f1, std_sf_roc, std_sf_acc,
                     mean_rf_f1, mean_rf_roc, mean_rf_acc, std_rf_f1, std_rf_roc, std_rf_acc,
                     p_f1, p_roc, p_acc,
                     sf.get_params(), rf.get_params()]

    print(df.loc[d_idx])
    df.to_csv(log_name + '.csv', index=False)

if use_neptune:
    neptune.log_artifact(log_name + '.csv')
    neptune.log_artifact(log_name + '_cv_std_score.csv')
    neptune.stop()
