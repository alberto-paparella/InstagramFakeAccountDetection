import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy import arange


def data_plot(fake, correct, title, ylabel, feature, name, xlabel='ID account', zeros=True):
    '''
    Function to create a plot about a feature in the original dataset useful for data analysis.
    '''
    plt.cla()
    plt.clf()
    plt.title(f"{title} - {name}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    fake = pd.DataFrame.from_dict(fake)
    correct = pd.DataFrame.from_dict(correct)
    fake_dim = fake.shape[0]

    plt.plot(fake.loc[fake[feature] < 1000000].index.values,
             fake.loc[fake[feature] < 1000000][feature],
             '.', color='r', markersize=1, label='fake')
    plt.plot(correct.loc[correct[feature] < 1000000].index.values + fake_dim,
             correct.loc[correct[feature] < 1000000][feature],
             '.', color='b', markersize=1, label='non fake')
    if zeros:
        plt.plot(fake.loc[fake[feature] == 0].index.values,
                 fake.loc[fake[feature] == 0][feature],
                 '.', color='orange', markersize=1, label='0-fake')
        plt.plot(correct.loc[correct[feature] == 0].index.values + fake_dim,
                 correct.loc[correct[feature] == 0][feature],
                 '.', color='g', markersize=1, label='0-non fake')
    plt.legend(loc='best')
    plt.savefig(f'./visualization/plots/{name}_{ylabel}.png')


def print_all_plots(fake_if, correct_if, fake_IJECE, correct_IJECE):
    '''
    Function to print all the plots relative to the features in both datasets.
    '''
    try:
        os.makedirs('./visualization/plots')
    except FileExistsError:
        pass

    data_plot(fake_if, correct_if, 'Numero di media per account', 'N media', 'nmedia', 'InstaFake')
    data_plot(fake_if, correct_if, 'Numero di follower per account', 'N follower', 'nfollower', 'InstaFake')
    data_plot(fake_if, correct_if, 'Numero di following per account', 'N following', 'nfollowing', 'InstaFake')
    data_plot(fake_if, correct_if, 'Lunghezza bio per account', 'Lunghezza bio', 'biol', 'InstaFake')
    data_plot(fake_if, correct_if, 'Engagement rate - like per account', 'ERL', 'mediaLikeNumbers', 'InstaFake')
    data_plot(fake_if, correct_if, 'Intervallo medio fra post per account', 'Tempo medio', 'avgtime', 'InstaFake')

    data_plot(fake_IJECE, correct_IJECE, 'Numero di media per account', 'N media', 'nmedia', 'IJECE')
    data_plot(fake_IJECE, correct_IJECE, 'Numero di follower per account', 'N follower', 'nfollower', 'IJECE')
    data_plot(fake_IJECE, correct_IJECE, 'Numero di following per account', 'N following', 'nfollowing', 'IJECE')
    data_plot(fake_IJECE, correct_IJECE, 'Lunghezza bio per account', 'Lunghezza bio', 'biol', 'IJECE')
    data_plot(fake_IJECE, correct_IJECE, 'Engagement rate - like per account', 'ERL', 'erl', 'IJECE')
    data_plot(fake_IJECE, correct_IJECE, 'Intervallo medio fra post per account', 'Tempo medio', 'avgtime', 'IJECE')


def result_plot(results, exp_list, n_iter):
    '''
    Function to print plots representing the performance scores of the models provided by the experiments.
    '''
    try:
        os.makedirs('./visualization/plots_results')
    except FileExistsError:
        pass

    methods = {'dt': 'Decision Tree', 'rf': 'Random Forest', 'svm': 'Support Vector Machine',
               'nbb': 'Naive Bayes (Bernoulli dist.)', 'nbg': 'Naive Bayes (Gaussian dist.)',
               'lr': 'Logistic Regression', 'mp': 'Multilayer Perceptron'}

    for method in methods.keys():
        if method not in exp_list:
            continue
        plt.cla()
        plt.clf()
        dataset_labels = [
            'IJ_Pap', 'IJ_Def', 'IJ_Cus', 'IJ_Com',
            'IF_Pap', 'IF_Def', 'IF_Cus', 'IF_Com',
            'Combo_P', 'Combo_F']
        metrics_labels = ['accuracy', 'precision', 'recall', 'f1']
        scores = dict()
        for m in metrics_labels:
            scores[m] = [
                results['IJECEPaper'][method]['default'][m],
                results['IJECE'][method]['default'][m],
                results['IJECE'][method]['custom'][m],
                results['CompIJECE'][method]['custom'][m],
                results['IFPaper'][method]['default'][m],
                results['InstaFake'][method]['default'][m],
                results['InstaFake'][method]['custom'][m],
                results['CompInstaFake'][method]['custom'][m],
                results['ComboPar'][method]['custom'][m],
                results['ComboFull'][method]['custom'][m]
            ]
        x = arange(len(dataset_labels))

        width = 0.10  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')
        plt.gcf().set_size_inches(8, 6)

        for metric, score in scores.items():
            offset = width * multiplier
            ax.bar(x + offset, score, width, label=metric)
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.set_ylabel('Score')
        ax.set_title(f'Scores for every dataset with {methods[method]} in {n_iter} iterations')
        ax.set_xticks(x + width, dataset_labels)
        ax.legend(loc='best', ncols=4)
        ax.set_ylim(0.6, 1)

        plt.savefig(f'./visualization/plots_results/{methods[method]}_{n_iter} iter.png')
