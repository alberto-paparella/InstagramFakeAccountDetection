from dataset.utils import get_custom_dataset, get_default_dataset
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


def get_dataset_instafake():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(path)
    train_df = pd.read_json("dataset/deep/instafake_df_train.json")
    validation_df = pd.read_json("dataset/deep/instafake_df_val.json")
    return get_default_dataset(train_df, validation_df, False), get_custom_dataset(train_df, validation_df, False)


def get_dataset_combined(full=False):
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


def train_save(name, train, runner, folder, timestamp):
    import datetime
    print(f"Now training {name} model...")
    (model, history, layers, learning) = runner(train)
    composition = ""
    for elem in layers:
        composition += str(elem) + "|"
    # model.save_weights(f"{folder}/{name}_{timestamp}/{name}_{timestamp}_w.h5")
    model.save(f"{folder}/{name}_{timestamp}/{name}_{timestamp}.h5")
    logfile = open(f'{folder}/log.csv', 'a+')
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
    print(f"Attempting to load {name} model...")
    from tensorflow.keras.models import load_model
    model = load_model(f"{folder}/{name}/{name}.h5")
    print("Model loaded!")
    return model
