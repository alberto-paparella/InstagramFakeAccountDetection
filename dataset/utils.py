from dataset.normalizer import csv_importer_full
import random
import pandas as pd
import os
import sys


PERCENT_TRAIN = 70


def set_path():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if path not in sys.path:
        sys.path.insert(1, path)
    return path


def shuffle_and_split(ds_fake, ds_correct):
    # print(f"Now splitting dataset with ratio {PERCENT_TRAIN}:{100 - PERCENT_TRAIN}")
    random.shuffle(ds_fake)
    random.shuffle(ds_correct)

    ds_train = ds_fake[:int(len(ds_fake) * (PERCENT_TRAIN / 100))]
    ds_train += ds_correct[:int(len(ds_correct) * (PERCENT_TRAIN / 100))]

    ds_validation = ds_fake[int(len(ds_fake) * (PERCENT_TRAIN / 100)):]
    ds_validation += ds_correct[int(len(ds_correct) * (PERCENT_TRAIN / 100)):]

    random.shuffle(ds_train)
    random.shuffle(ds_validation)

    # print("Loading complete.")

    df_train = pd.DataFrame.from_dict(ds_train)
    df_validation = pd.DataFrame.from_dict(ds_validation)
    # print(df_train)
    # print(df_validation)
    return df_train, df_validation


def find_demarcator(dataset):
    """
    Restituisce l'indice del primo elemento non fake
    :param dataset: il dataset
    :return: l'indice
    """
    idx = 0
    for elem in dataset:
        if elem['fake'] == 1:
            idx += 1
        else:
            break
    return idx


def get_default_dataset_csv():
    default_dataset = csv_importer_full("dataset/sources/user_fake_authentic_2class.csv")
    idx = find_demarcator(default_dataset)
    fake = default_dataset[:idx]
    correct = default_dataset[idx:]
    return shuffle_and_split(fake, correct)


def get_custom_dataset(train_df, validation_df, csv=False):
    if csv:
        custom_train_df = train_df.drop(["mediaLikeNumbers", "mediaCommentNumbers",
                                         "mediaCommentsAreDisabled", "mediaHashtagNumbers", "mediaHasLocationInfo",
                                         "userHasHighlighReels", "usernameLength", "usernameDigitCount"], axis=1)
        custom_validation_df = validation_df.drop(["mediaLikeNumbers", "mediaCommentNumbers",
                                                   "mediaCommentsAreDisabled", "mediaHashtagNumbers",
                                                   "mediaHasLocationInfo",
                                                   "userHasHighlighReels", "usernameLength", "usernameDigitCount"],
                                                  axis=1)
    else:
        custom_train_df = train_df.drop(["pic", "cl", "cz", "ni", "lt", "ahc", "pr", "fo", "cs"], axis=1)
        custom_validation_df = validation_df.drop(["pic", "cl", "cz", "ni", "lt", "ahc", "pr", "fo", "cs"], axis=1)
    return custom_train_df, custom_validation_df

