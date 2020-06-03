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
    # Axis for scores
    # Set figure and first axis
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.xticks(rotation=90)
    ax1.set_xticks(fraction_range)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xlabel('Fraction of shuffled instances')

    # Set second axis
    ax2 = ax1.twiny()
    plt.xticks(rotation=90)
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xticklabels(tick_function(dict_res['correlations']))
    ax2.set_xlabel('New feature correlation')

    # Plot scores
    plt.plot(fraction_range, dict_res['rf_scores'], label='Random Forest', color='black', linestyle='dashdot')
    plt.plot(fraction_range, dict_res['sf_scores'], label='Similarity Forest', color='black', linestyle='solid')

    # Set legend and titles
    plt.legend()
    ax1.set_ylabel('Score')
    plt.title(f'Scores, dataset', fontsize=16)

    # Axis for importances
    df_permutation_importances = pd.DataFrame(dict_res['permutation_importances'])

    # Set figure and first axis
    ax3 = fig.add_subplot(1, 2, 2)
    plt.xticks(rotation=90)
    ax3.set_xticks(fraction_range)
    ax3.set_xlim(0.0, 1.0)
    ax3.set_xlabel('Fraction of shuffled instances')

    # Set second axis
    ax4 = ax3.twiny()
    plt.xticks(rotation=90)
    ax4.set_xticks(ax3.get_xticks())
    ax4.set_xlim(0.0, 1.0)
    ax4.set_xticklabels(tick_function(dict_res['correlations']))
    ax4.set_xlabel('New feature correlation')

    # Plot importances
    plt.plot(fraction_range, df_permutation_importances['rf_train'].values,
             label='Random Forest, train', color='black', linestyle='dashdot')
    plt.plot(fraction_range, df_permutation_importances['rf_test'].values,
             label='Random Forest, test', color='black', linestyle='dotted')
    plt.plot(fraction_range, df_permutation_importances['sf_train'].values,
             label='Similarity Forest, train', color='black', linestyle='solid')
    plt.plot(fraction_range, df_permutation_importances['sf_test'].values,
             label='Similarity Forest, test', color='black', linestyle='dashed')

    # Set legend and titles
    plt.legend()
    ax3.set_ylabel('New feature importance')
    plt.title(f'Permutation importance', fontsize=16)
    plt.tight_layout()

    plt.savefig(f'./figures/{task} {feat} {dataset}', dpi=100)
    plt.show()


if __name__ == '__main__':
    visualize_bias()
