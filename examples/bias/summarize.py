import pandas as pd
import numpy as np
import json
import glob
import re


logs = glob.glob('./logs/*.json')

for log in logs:
    results = []
    with open(log, 'r') as f:
        json_res = json.load(f)
        dict_res = json.loads(json_res)

    # remove beginning and ending of log name
    file_pattern = re.compile('./logs/')
    log = re.sub(file_pattern, '', log)
    end_pattern = re.compile('results.json')
    log = re.sub(end_pattern, '', log)
    log = log.replace('_', ' ').strip().split()
    feature = log[-1]
    task = log[0]
    dataset_name = ' '.join(log[1:len(log)-1])

    # reconstruct log name
    log = task + ' ' + feature + ' ' + dataset_name

    df_f_imp = pd.DataFrame(dict_res['permutation_importances'])
    score_diffs = np.array(dict_res['rf_scores']) - np.array(dict_res['sf_scores'])
    f_imp_diffs_train = df_f_imp['rf_train'] - df_f_imp['sf_train']
    f_imp_diffs_test = df_f_imp['rf_test'] - df_f_imp['sf_test']
    indices = [0, 2, 4, 7]
    results.append({
        'task': task,
        'feature': feature,
        'dataset': dataset_name,
        'score diff shuffle 0.0': score_diffs[0],
        'score diff shuffle 0.1': score_diffs[2],
        'score diff shuffle 0.2': score_diffs[4],
        'score diff shuffle 0.5': score_diffs[7],
        'feat importance train diff 0.0': f_imp_diffs_train.values[0],
        'feat importance train diff 0.1': f_imp_diffs_train.values[2],
        'feat importance train diff 0.2': f_imp_diffs_train.values[4],
        'feat importance train diff 0.5': f_imp_diffs_train.values[7],
        'feat importance test diff 0.0': f_imp_diffs_test.values[0],
        'feat importance test diff 0.1': f_imp_diffs_test.values[2],
        'feat importance test diff 0.2': f_imp_diffs_test.values[4],
        'feat importance test diff 0.5': f_imp_diffs_test.values[7],
    })


    df_results = pd.DataFrame(results)
    # index
    df_results.set_index(['task', 'feature', 'dataset'], inplace=True)
    df_results.sort_index(inplace=True)
    # header
    correlations = np.array(dict_res['correlations'])[indices]
    header = pd.MultiIndex.from_product([['score', 'feat importance train', 'feat importance test'],
                                         correlations],
                                        names=['difference in', 'correlation'])
    df_results.columns = header
    print(df_results.head())
    df_results.to_csv(f'./summaries/{task} {feature} {dataset_name}.csv')
