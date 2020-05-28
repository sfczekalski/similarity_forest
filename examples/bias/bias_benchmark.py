from examples.classification.classification_datasets import get_datasets as get_classification_datasets
from examples.regression.regression_datasets import get_datasets as get_regression_datasets
from examples.bias.bias import *

import json
import neptune
import seaborn as sns
sns.set_style('whitegrid')

task = 'regression'
feature = 'numerical'
if task == 'classification':
    get_datasets = get_classification_datasets
else:
    get_datasets = get_regression_datasets

neptune.set_project('sfczekalski/BiasSF')
neptune.init('sfczekalski/BiasSF')
neptune.create_experiment(name=f'{task} {feature}', properties={'task': task, 'feature': feature})

SEED = 42
fraction_range = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

for d in get_datasets():
    X_train, X_test, y_train, y_test, dataset_name = d
    print(dataset_name)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    features = [f'f{i + 1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=features)
    image_path = f'./logs/{task}_{dataset_name}_{feature}'

    correlations, rf_scores, sf_scores, permutation_importances = bias_experiment(df, y,
                                                                                  task, feature,
                                                                                  fraction_range, SEED)
    plot_bias(fraction_range, correlations, rf_scores, sf_scores,
              permutation_importances, dataset_name, image_path+'.png')

    # Log chart and raw results into Neptune
    neptune.send_image(f'{dataset_name}', image_path+'.png')
    results_dict = {
        'correlations': correlations,
        'rf_scores': rf_scores,
        'sf_scores': sf_scores,
        'permutation_importances': permutation_importances
    }
    results_json = json.dumps(results_dict)
    result_file_path = f'./logs/{task}_{dataset_name}_{feature}_results.json'
    with open(result_file_path, 'w+') as f:
        json.dump(results_json, f)
    neptune.log_artifact(result_file_path)

neptune.stop()
