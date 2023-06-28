from dataset.normalizer import csv_importer_full, json_importer_full
from dataset.utils import find_demarcator, shuffle_and_split, get_custom_dataset
import os, sys
import pandas as pd

class LayerConfiguration:
    def __init__(self, size, activation_function="relu"):
        self.size = size
        self.activation_function = activation_function

def get_dataset_IJECE():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not path in sys.path:
        sys.path.insert(1, path)
    os.chdir(path)
    del path
    train_df = pd.read_csv("dataset/deep/IJECE_df_train.csv")
    validation_df = pd.read_csv("dataset/deep/IJECE_df_val.csv")
    return (train_df, validation_df), (get_custom_dataset(train_df, validation_df, True))


def get_dataset_spz():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(path)
    train = pd.read_json("dataset/deep/SPZ_df_train.json")
    validation = pd.read_json("dataset/deep/SPZ_df_val.json")
    return (train, validation),(get_custom_dataset(train, validation, False))
