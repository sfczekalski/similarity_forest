from examples.classification.classification_datasets import get_datasets as get_classification_datasets
from examples.regression.regression_datasets import get_datasets as get_regression_datasets
from examples.bias.bias import *

import json
import neptune
import seaborn as sns
sns.set_style('whitegrid')


def bias_benchmark(task, feature, fraction_range, plot=True):

    if task == 'classification':
        get_datasets = get_classification_datasets
    else:
        get_datasets = get_regression_datasets

    SEED = 42

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
        if plot:
            plot_bias(fraction_range, correlations, rf_scores, sf_scores,
                      permutation_importances, dataset_name, image_path+'.png')

            # Log chart and raw results into Neptune
            neptune.send_image(f'{dataset_name}', image_path+'.png')

        results_dict = {
            'correlations': correlations.tolist(),
            'rf_scores': rf_scores.tolist(),
            'sf_scores': sf_scores.tolist(),
            'permutation_importances': permutation_importances
        }
        results_json = json.dumps(results_dict)
        result_file_path = f'./logs/{task}_{dataset_name}_{feature}_results.json'
        with open(result_file_path, 'w+') as f:
            json.dump(results_json, f)
            print(f'Saved results to: {result_file_path}')
        neptune.log_artifact(result_file_path)


def main(tasks, features):
    neptune.set_project('sfczekalski/BiasSF')
    neptune.init('sfczekalski/BiasSF')
    neptune.create_experiment(name='Bias summary, raw')

    fraction_range = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    for task in tasks:
        print(f'{task}'.upper())
        for feature in features:
            print(f'{feature}'.upper())
            bias_benchmark(task, feature, fraction_range, plot=False)

    neptune.stop()


if __name__ == '__main__':
    tasks = ['regression'] #, 'classification'
    features = ['categorical', 'numerical']

    main(tasks, features)

