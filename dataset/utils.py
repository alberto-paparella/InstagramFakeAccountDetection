from dataset.normalizer import csv_importer_full, json_importer_full, compute_erl, compute_erc, compute_avg_time
import random
import pandas as pd
import os
import sys
import numpy as np

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

    df_train = pd.DataFrame.from_dict(ds_train)
    df_validation = pd.DataFrame.from_dict(ds_validation)
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


def get_fake_correct_default(csv):
    if csv:
        default_dataset = csv_importer_full("dataset/sources/user_fake_authentic_2class.csv")
        idx = find_demarcator(default_dataset)
        return default_dataset[:idx], default_dataset[idx:]
    else:
        return json_importer_full("dataset/sources/automatedAccountData.json", True), json_importer_full(
            "dataset/sources/nonautomatedAccountData.json", False)


def get_default_dataset_csv():
    fake, correct = get_fake_correct_default(True)
    return shuffle_and_split(fake, correct)

def get_internal_sum(series):
    res = np.zeros((series.shape[0]))
    for i,el in enumerate(series):
        res[i] = sum(el)
    return res


def augment_spz(dataframe: pd.DataFrame):
    dataframe["erl"] = compute_erl(get_internal_sum(dataframe["mediaLikeNumbers"]),
                                   dataframe["userMediaCount"],
                                   dataframe["userFollowerCount"])
    dataframe["erc"] = compute_erc(get_internal_sum(dataframe["mediaCommentNumbers"]),
                                   dataframe["userMediaCount"],
                                   dataframe["userFollowerCount"])
    dataframe["avgtime"] = compute_avg_time(dataframe["mediaUploadTimes"])
    dataframe["mediaLikeNumbersAvg"] = (
        sum(dataframe["mediaLikeNumbers"]) / dataframe["userMediaCount"] if dataframe[
                                                                                'userMediaCount'] != 0 else 0)
    dataframe["mediaCommentNumbersAvg"] = (
        sum(dataframe["mediaCommentNumbers"]) / dataframe["userMediaCount"] if dataframe[
                                                                                   'userMediaCount'] != 0 else 0),
    dataframe["mediaCommentsDisabledAvg"] = (
        sum(dataframe["mediaCommentsAreDisabled"]) / dataframe["userMediaCount"] if
        dataframe['userMediaCount'] != 0 else 0),
    dataframe["mediaHashtagNumbersAvg"] = (
        sum(dataframe["mediaHashtagNumbers"]) / dataframe["userMediaCount"] if dataframe[
                                                                                   'userMediaCount'] != 0 else 0),
    dataframe["mediaHasLocationInfoAvg"] = (
        sum(dataframe["mediaHasLocationInfo"]) / dataframe["userMediaCount"] if dataframe[
                                                                                    "userMediaCount"] != 0 else 0),
    return dataframe


def common_augment(dataframe: pd.DataFrame):
    dataframe["erl"] = compute_erl(get_internal_sum(dataframe["mediaLikeNumbers"]),
                                   dataframe["userMediaCount"],
                                   dataframe["userFollowerCount"])
    dataframe["erc"] = compute_erc(get_internal_sum(dataframe["mediaCommentNumbers"]),
                                   dataframe["userMediaCount"],
                                   dataframe["userFollowerCount"])
    dataframe["avgtime"] = compute_avg_time(dataframe["mediaUploadTimes"])
    return dataframe


def get_custom_dataset(train_df, validation_df, csv, compatibility=True):
    if csv:
        if compatibility:
            custom_train_df = train_df.drop(["pic", "cl", "cz", "ni", "lt", "ahc", "pr", "fo", "cs"],
                                            axis=1)
            custom_validation_df = validation_df.drop(
                ["pic", "cl", "cz", "ni", "lt", "ahc", "pr", "fo", "cs"], axis=1)
        else:
            custom_train_df = train_df.drop(["ni", "lt", "ahc", "avgtime"], axis=1)
            custom_validation_df = validation_df.drop(["ni", "lt", "ahc", "avgtime"], axis=1)
    else:
        train_df = common_augment(train_df)
        validation_df = common_augment(validation_df)
        if compatibility:
            custom_train_df = train_df.drop(["mediaLikeNumbers", "mediaCommentNumbers",
                                             "mediaCommentsAreDisabled", "mediaHashtagNumbers", "mediaHasLocationInfo",
                                             "userHasHighlighReels", "usernameLength", "usernameDigitCount"], axis=1)
            custom_validation_df = validation_df.drop(["mediaLikeNumbers", "mediaCommentNumbers",
                                                       "mediaCommentsAreDisabled", "mediaHashtagNumbers",
                                                       "mediaHasLocationInfo",
                                                       "userHasHighlighReels", "usernameLength", "usernameDigitCount"],
                                                      axis=1)
        else:
            train_df: pd.DataFrame
            custom_train_df = train_df
            custom_train_df = augment_spz(custom_train_df)
            custom_validation_df = validation_df
            custom_validation_df = augment_spz(custom_validation_df)
    return custom_train_df, custom_validation_df


def get_deep_learning_dataset():
    fake_csv, correct_csv = get_fake_correct_default(True)
    fake_json, correct_json = get_fake_correct_default(False)

    ijece_train, ijece_val = shuffle_and_split(fake_csv, correct_csv)
    ijece_train.to_csv(f'./dataset/deep/IJECE_df_train.csv')
    ijece_val.to_csv(f'./dataset/deep/IJECE_df_val.csv')

    spz_train, spz_val = shuffle_and_split(fake_json, correct_json)
    spz_train.to_json(f'./dataset/deep/spz_df_train.json')
    spz_val.to_json(f'./dataset/deep/spz_df_val.json')
