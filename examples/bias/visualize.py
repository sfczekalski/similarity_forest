import matplotlib.pyplot as plt
import pandas as pd
import json
from bias import plot_bias
import click


def tick_function(correlations):
    return [round(c, 2) for c in correlations]


@click.command()
@click.option('--task', prompt='Task to perform',
              help='classification or regression')
@click.option('--feat', prompt='Feature type',
              help='categorical or numerical')
def visualize_bias(task, feat):
    """
    Simple CLI application to visualize how synthetic column influences fitting RF and SF.
    Two plots are prepared:
        first with a scores, depending on synthetic column correlation
        second with permutation feature importances, depending on synthetic column correlation
    :param task: task to perfom, either classification or regression
    :param feat: type of synthetic feature to be generated, either numerical or categorical
    :return: None
    """
    assert task in ['classification', 'regression'], \
        f'task should be either classification or regression, found: {task}'
    assert feat in ['categorical', 'numerical'], \
        f'feat should be either categorical or numerical, found: {feat}'

    if task == 'regression':
        dataset = 'boston'
    else:
        dataset = 'heart'

    with open(f'./logs/{task}_{dataset}_{feat}_results.json', 'r') as f:
        json_res = json.load(f)
        dict_res = json.loads(json_res)

    fraction_range = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

    plot_bias(fraction_range=fraction_range,
              correlations=dict_res['correlations'],
              rf_scores=dict_res['rf_scores'],
              sf_scores=dict_res['sf_scores'],
              permutation_importances=dict_res['permutation_importances'],
              dataset_name=dataset,
              image_path=f'./figures/{task} {feat} {dataset}')


if __name__ == '__main__':
    visualize_bias()

