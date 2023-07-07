import os
import pandas as pd
import matplotlib.pyplot as plt

from deep.common import get_dataset_combined

(partial_train, partial_validation) = get_dataset_combined(False)
(full_train, full_validation) = get_dataset_combined(True)

features = ['nmedia', 'biol', 'url', 'nfollowing', 'nfollower', 'erl', 'erc',
            'avgtime', 'mediaHashtagNumbers', 'followerToFollowing', 'hasMedia',
            'fake']

for feature in features:
    plt.cla()
    plt.clf()

    train_dim = full_train.shape[0]

    plt.plot(full_train.loc[full_train[feature] < 1000000].index.values,
             full_train.loc[full_train[feature] < 1000000][feature],
             '.', color='r', markersize=1, label='fake')
    plt.plot(full_validation.loc[full_validation[feature] < 1000000].index.values + train_dim,
             full_validation.loc[full_validation[feature] < 1000000][feature],
             '.', color='b', markersize=1, label='non fake')
    plt.plot(full_train.loc[full_train[feature] == 0].index.values,
             full_train.loc[full_train[feature] == 0][feature],
             '.', color='orange', markersize=1, label='0-fake')
    plt.plot(full_validation.loc[full_validation[feature] == 0].index.values + train_dim,
             full_validation.loc[full_validation[feature] == 0][feature],
             '.', color='g', markersize=1, label='0-non fake')
    plt.legend(loc='best')
    plt.title(f'Full {feature}')
    plt.show()


