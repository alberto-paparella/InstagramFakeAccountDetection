import os
import pandas as pd
import matplotlib.pyplot as plt

def my_plot(fake, correct, title, ylabel, feature, name, xlabel='ID account', zeros=True):
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
    plt.savefig(f'./dataset/visualization/plots/{name}_{ylabel}.png')


def print_all_plots(fake_if, correct_if, fake_IJECE, correct_IJECE):
    try:
        os.makedirs('./dataset/visualization/plots')
    except FileExistsError:
        pass

    my_plot(fake_if, correct_if, 'Numero di media per account', 'N media', 'nmedia', 'InstaFake')
    my_plot(fake_if, correct_if, 'Numero di follower per account', 'N follower', 'nfollower', 'InstaFake')
    my_plot(fake_if, correct_if, 'Numero di following per account', 'N following', 'nfollowing', 'InstaFake')
    my_plot(fake_if, correct_if, 'Lunghezza bio per account', 'Lunghezza bio', 'biol', 'InstaFake')
    my_plot(fake_if, correct_if, 'Engagement rate - like per account', 'ERL', 'mediaLikeNumbers', 'InstaFake')
    my_plot(fake_if, correct_if, 'Intervallo medio fra post per account', 'Tempo medio', 'avgtime', 'InstaFake')

    my_plot(fake_IJECE, correct_IJECE, 'Numero di media per account', 'N media', 'nmedia', 'IJECE')
    my_plot(fake_IJECE, correct_IJECE, 'Numero di follower per account', 'N follower', 'nfollower', 'IJECE')
    my_plot(fake_IJECE, correct_IJECE, 'Numero di following per account', 'N following', 'nfollowing', 'IJECE')
    my_plot(fake_IJECE, correct_IJECE, 'Lunghezza bio per account', 'Lunghezza bio', 'biol', 'IJECE')
    my_plot(fake_IJECE, correct_IJECE, 'Engagement rate - like per account', 'ERL', 'erl', 'IJECE')
    my_plot(fake_IJECE, correct_IJECE, 'Intervallo medio fra post per account', 'Tempo medio', 'avgtime', 'IJECE')
