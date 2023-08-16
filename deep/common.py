# A set of common utilities used by the deep module
from dataset.utils import get_custom_dataset, get_default_dataset
import os, sys
import pandas as pd


class LayerConfiguration:
    """
    Layer abstraction class, used by individual run_model methods to define the neural network.
    """
    def __init__(self, size, activation_function="relu"):
        self.size = size
        self.activation_function = activation_function

    def __repr__(self):
        return f"{self.size}-{self.activation_function} "


def get_dataset_IJECE():
    """
    Loads the IJECE dataset for deep learning
    :return: ((default_train_df, default_validation_df), (custom_train_df, custom_validation_df))
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not path in sys.path:
        sys.path.insert(1, path)
    os.chdir(path)
    del path
    train_df = pd.read_csv("dataset/deep/IJECE_df_train.csv")
    validation_df = pd.read_csv("dataset/deep/IJECE_df_val.csv")
    return get_default_dataset(train_df, validation_df, True), get_custom_dataset(train_df, validation_df, True)


def get_dataset_instafake():
    """
    Loads the InstaFake dataset for deep learning
    :return: ((default_train_df, default_validation_df), (custom_train_df, custom_validation_df))
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(path)
    train_df = pd.read_json("dataset/deep/instafake_df_train.json")
    validation_df = pd.read_json("dataset/deep/instafake_df_val.json")
    return get_default_dataset(train_df, validation_df, False), get_custom_dataset(train_df, validation_df, False)


def get_dataset_combined(full=False):
    """
    Loads the combined dataset for deep learning
    :param full: boolean, whether or not to load the combo_full or the combo_par
    :return: (train_df, validation_df)
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not path in sys.path:
        sys.path.insert(1, path)
    os.chdir(path)
    del path
    if full:
        train_df = pd.read_json("dataset/deep/combo_full_df_train.json")
        validation_df = pd.read_json("dataset/deep/combo_full_df_val.json")
    else:
        train_df = pd.read_json("dataset/deep/combo_partial_df_train.json")
        validation_df = pd.read_json("dataset/deep/combo_partial_df_val.json")
    return train_df, validation_df


def get_compatible_dataset(mode="if"):
    """
    Loads the compatible dataset for deep learning
    :param mode: can either be if or ijece, chooses which dataset will get loaded up.
    :return: (train_df, validation_df)
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not path in sys.path:
        sys.path.insert(1, path)
    os.chdir(path)
    del path
    if mode == "if":
        train_df = pd.read_json("dataset/deep/comp_instafake_df_train.json")
        validation_df = pd.read_json("dataset/deep/comp_instafake_df_val.json")
    elif mode == "ijece":
        train_df = pd.read_json("dataset/deep/comp_ijece_df_train.json")
        validation_df = pd.read_json("dataset/deep/comp_ijece_df_val.json")
    else:
        return None, None
    return train_df, validation_df


def train_save(name, train, runner, folder, timestamp):
    """
    Trains and then saves a neural network, while keeping logs about progress.
    :param name: The name of the network
    :param train: The training set
    :param runner: The method that needs to be run in order to train the model
    :param folder: Base folder in which data will be saved
    :param timestamp: self explanatory, used as a primary key in order to avoid data being overwritten
    :return: the trained and saved model
    """
    import datetime
    print(f"Now training {name} model...")
    (model, history, layers, learning) = runner(train)
    composition = ""
    for elem in layers:
        composition += str(elem) + "|"
    model.save(f"{folder}/{name}_{timestamp}/{name}_{timestamp}.h5")
    logfile = open(f'{folder}/log.csv', 'a+')
    # For whatever reason, sometimes metrics come with a postfix _1. This try/catch deals with this oddity.
    try:
        f1_score = 2 * (history['precision'][-1] * history['recall'][-1]) / (
                history['precision'][-1] + history['recall'][-1])
        logfile.write(
            f"\n{name}_{timestamp};{history['accuracy'][-1]};{history['loss'][-1]};{history['precision'][-1]}; {history['recall'][-1]}; {f1_score};{composition};{learning['rate']}; {learning['epochs']}; {learning['batch_size']};")
    except Exception:
        f1_score = 2 * (history['precision_1'][-1] * history['recall_1'][-1]) / (
                history['precision_1'][-1] + history['recall_1'][-1])
        logfile.write(
            f"\n{name}_{timestamp};{history['accuracy'][-1]};{history['loss'][-1]};{history['precision_1'][-1]}; {history['recall_1'][-1]}; {f1_score};{composition};{learning['rate']}; {learning['epochs']}; {learning['batch_size']};")
    return model


def load_model(folder, name):
    """
    Loads up a previously saved model in the .h5 format
    :param folder: Base folder from which data needs to be loaded
    :param name: The name of the model
    :return: the loaded model
    """
    print(f"Attempting to load {name} model...")
    from tensorflow.keras.models import load_model
    model = load_model(f"{folder}/{name}/{name}.h5")
    print("Model loaded!")
    return model
