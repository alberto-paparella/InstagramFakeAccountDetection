from dataset.normalizer import csv_importer_full, json_importer_full
from dataset.utils import find_demarcator, shuffle_and_split, get_custom_dataset
import os, sys

class LayerConfiguration:
    def __init__(self, size, activation_function="relu"):
        self.size = size
        self.activation_function = activation_function

def get_dataset_IJCE():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not path in sys.path:
        sys.path.insert(1, path)
    os.chdir(path)
    del path
    default_dataset = csv_importer_full("dataset/sources/user_fake_authentic_2class.csv")
    idx = find_demarcator(default_dataset)
    fake = default_dataset[:idx]
    correct = default_dataset[idx:]
    train_df, validation_df = shuffle_and_split(fake, correct)
    custom_train_df, custom_validation_df = get_custom_dataset(train_df, validation_df, True)
    return (train_df, validation_df), (custom_train_df, custom_validation_df)


def get_dataset_spz():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(path)
    fake = json_importer_full("dataset/sources/automatedAccountData.json", True)
    correct = json_importer_full("dataset/sources/nonautomatedAccountData.json", False)
    train, validation = shuffle_and_split(fake, correct)

    return (train, validation),(get_custom_dataset(train, validation, False))
