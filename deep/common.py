from dataset.normalizer import csv_importer_full, json_importer_full
from dataset.utils import find_demarcator, shuffle_and_split, get_custom_dataset, get_default_dataset
import os, sys
import pandas as pd


class LayerConfiguration:
    def __init__(self, size, activation_function="relu"):
        self.size = size
        self.activation_function = activation_function

    def __repr__(self):
        return f"{self.size}-{self.activation_function} "


def get_dataset_IJECE():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not path in sys.path:
        sys.path.insert(1, path)
    os.chdir(path)
    del path
    train_df = pd.read_csv("dataset/deep/IJECE_df_train.csv")
    validation_df = pd.read_csv("dataset/deep/IJECE_df_val.csv")
    return get_default_dataset(train_df, validation_df, True), get_custom_dataset(train_df, validation_df, True)


def get_dataset_spz():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(path)
    train_df = pd.read_json("dataset/deep/SPZ_df_train.json")
    validation_df = pd.read_json("dataset/deep/SPZ_df_val.json")
    return get_default_dataset(train_df, validation_df, False), get_custom_dataset(train_df, validation_df, False)


def train_save(name, train, runner, folder):
    import ctypes  # An included library with Python install.

    import datetime
    print(f"Now training {name} model...")
    (model, history, layers) = runner(train)
    #ctypes.windll.user32.MessageBoxW(0, "Training done!", "TrainingProgress", 1)
    #ans = input("Do you want to save this model? (y/any)")
    #if ans == "y":
    composition = ""
    for elem in layers:
        composition += str(elem) + "|"
    model.save_weights(f"{folder}/{name}_{datetime.datetime.now().timestamp()}/{name}_{datetime.datetime.now().timestamp()}")
    logfile = open(f'{folder}/log.csv', 'a+')
    try:
        f1_score = 2 * (history['precision'][-1] * history['recall'][-1]) / (
                    history['precision'][-1] + history['recall'][-1])
        logfile.write(
            f"\n{name}_{datetime.datetime.now().timestamp()};{history['accuracy'][-1]};{history['loss'][-1]};{history['precision'][-1]}; {history['recall'][-1]}; {f1_score};{composition};")
    except Exception:
        f1_score = 2 * (history['precision_1'][-1] * history['recall_1'][-1]) / (
                history['precision_1'][-1] + history['recall_1'][-1])
        logfile.write(
            f"\n{name}_{datetime.datetime.now().timestamp()};{history['accuracy'][-1]};{history['loss'][-1]};{history['precision_1'][-1]}; {history['recall_1'][-1]}; {f1_score};{composition};")
    return model
