from dataset.normalizer import csv_importer_full, json_importer_full
import random
import pandas as pd

PERCENT_TRAIN = 70


def shuffle_and_split(ds_fake, ds_correct):
    """
    Shuffles and splits equally fake and correct rows of information from a dataset
    :param ds_fake: fake rows
    :param ds_correct: correct rows
    :return: two dataframes, train and val
    """
    random.shuffle(ds_fake)
    random.shuffle(ds_correct)

    ds_train = ds_fake[:int(len(ds_fake) * (PERCENT_TRAIN / 100))]
    ds_train += ds_correct[:int(len(ds_correct) * (PERCENT_TRAIN / 100))]

    ds_val = ds_fake[int(len(ds_fake) * (PERCENT_TRAIN / 100)):]
    ds_val += ds_correct[int(len(ds_correct) * (PERCENT_TRAIN / 100)):]

    random.shuffle(ds_train)
    random.shuffle(ds_val)

    df_train = pd.DataFrame.from_dict(ds_train)
    df_val = pd.DataFrame.from_dict(ds_val)
    return df_train, df_val


def find_demarcator(dataset):
    """
    Returns the first non-fake element's index
    :param dataset: the dataset
    :return: the index
    """
    idx = 0
    for elem in dataset:
        if elem['fake'] == 1:
            idx += 1
        else:
            break
    return idx


def get_fake_correct_default(csv):
    """
    Load up the dataset, already split up in fake/correct
    :param csv: if the file is in csv format (IJECE)
    :return: the datasets
    """
    if csv:
        default_dataset = csv_importer_full("dataset/sources/user_fake_authentic_2class.csv")
        idx = find_demarcator(default_dataset)
        return default_dataset[:idx], default_dataset[idx:]
    else:
        return json_importer_full("dataset/sources/automatedAccountData.json", True), json_importer_full(
            "dataset/sources/nonautomatedAccountData.json", False)


def get_combined_datasets():
    """
    Load up the two datasets and then combine them.
    :return: the datasets, combined in partial and full mode
    """
    fake_if = json_importer_full("./dataset/sources/automatedAccountData.json", True, False)
    correct_if = json_importer_full("./dataset/sources/nonautomatedAccountData.json", False, False)
    ijece = csv_importer_full("./dataset/sources/user_fake_authentic_2class.csv", False)
    idx = find_demarcator(ijece)
    dataset = {"full": dict(), "partial": dict()}
    dataset["full"]["fake"] = fake_if + ijece[:idx]
    dataset["full"]["correct"] = correct_if + ijece[idx:]
    ijece_fake = ijece[:idx]
    random.shuffle(ijece_fake)
    ijece_correct = ijece[idx:]
    random.shuffle(ijece_correct)
    dataset["partial"]["fake"] = fake_if + ijece_fake[:700]
    dataset["partial"]["correct"] = correct_if + ijece_correct[:700]
    return dataset


def get_compatible_dataset(train_df, val_df, csv):
    """
    Drops the needed keys to make datasets compatible
    :param train_df: train dataset
    :param val_df: val dataset
    :param csv: flag to tell if we're dealing with IJECE or IF
    :return: the compatible datasets
    """
    if csv:
        custom_train_df = train_df.drop(["pic", "cl", "cz", "ni", "lt", "pr", "fo", "cs"], axis=1)
        custom_val_df = val_df.drop(["pic", "cl", "cz", "ni", "lt", "pr", "fo", "cs"], axis=1)
    else:
        custom_train_df = train_df.drop(["mediaLikeNumbers",
                                         "userHasHighlighReels", "usernameLength", "usernameDigitCount",
                                         "mediaCommentNumbers", "mediaCommentsAreDisabled",
                                         "mediaHasLocationInfo", "userTagsCount"], axis=1)
        custom_val_df = val_df.drop(["mediaLikeNumbers",
                                                   "userHasHighlighReels", "usernameLength", "usernameDigitCount",
                                                   "mediaCommentNumbers", "mediaCommentsAreDisabled",
                                                   "mediaHasLocationInfo",
                                                   "userTagsCount"],
                                           axis=1)
    return custom_train_df, custom_val_df


def get_ijece_custom_dataset(train_df, val_df):
    """
    Get IJECE custom dataset
    :param train_df: train dataset
    :param val_df: val dataset
    :return: the dataset
    """
    custom_train_df = train_df.drop(["ni", "lt", "mediaHashtagNumbers"], axis=1)
    custom_val_df = val_df.drop(["ni", "lt", "mediaHashtagNumbers"], axis=1)
    return custom_train_df, custom_val_df


def get_custom_dataset(train_df, val_df, csv):
    """
    Routing function
    :param train_df: train dataset
    :param val_df: val dataset
    :param csv: flag to tell if we're dealing with IJECE or IF
    :return: the dataset
    """
    if csv:
        return get_ijece_custom_dataset(train_df, val_df)
    else:
        return train_df, val_df


def get_ijece_default_dataset(train_df, val_df):
    """
    Get the default IJECE dataset
    :param train_df: train dataset
    :param val_df: val dataset
    :return: the dataset
    """
    custom_train_df = train_df.drop(["followerToFollowing", "hasMedia"], axis=1)
    custom_val_df = val_df.drop(["followerToFollowing", "hasMedia"], axis=1)
    return custom_train_df, custom_val_df


def get_default_dataset(train_df, val_df, csv):
    """
    Routing function
    :param train_df: train dataset
    :param val_df: val dataset
    :param csv: flag to tell if we're dealing with IJECE or IF
    :return: the dataset
    """
    if csv:
        return get_ijece_default_dataset(train_df, val_df)
    else:
        return get_instafake_default_dataset(train_df, val_df)


def get_instafake_default_dataset(train_df, val_df):
    """
    Returns instafake default dataset
    :param train_df: train dataset
    :param val_df: val dataset
    :return:
    """
    default_train_df = train_df.drop(["biol",
                                      "url", "erl", "erc", "avgtime",
                                      "userHasHighlighReels", "usernameLength", "usernameDigitCount",
                                      "mediaCommentNumbers", "mediaCommentsAreDisabled", "mediaHasLocationInfo",
                                      "usernameLength", "usernameDigitCount"], axis=1)
    default_val_df = val_df.drop(["biol",
                                                "url", "erl", "erc", "avgtime",
                                                "userHasHighlighReels", "usernameLength", "usernameDigitCount",
                                                "mediaCommentNumbers", "mediaCommentsAreDisabled",
                                                "mediaHasLocationInfo",
                                                "usernameLength", "usernameDigitCount"],
                                        axis=1)
    return default_train_df, default_val_df


def get_deep_learning_dataset():
    """
    This function creates the fixed datasets for the Multilayer Perceptron experiments.
    :return:
    """
    fake_csv, correct_csv = get_fake_correct_default(True)
    fake_json, correct_json = get_fake_correct_default(False)

    # IJECE dataset
    ijece_train, ijece_val = shuffle_and_split(fake_csv, correct_csv)
    ijece_train.to_csv(f'./dataset/deep/IJECE_df_train.csv')
    ijece_val.to_csv(f'./dataset/deep/IJECE_df_val.csv')

    # InstaFake dataset
    if_train, if_val = shuffle_and_split(fake_json, correct_json)
    if_train.to_json(f'./dataset/deep/instafake_df_train.json')
    if_val.to_json(f'./dataset/deep/instafake_df_val.json')

    # Instafake compatible dataset
    if_train_c = if_train
    if_val_c = if_val
    if_train_c, if_val_c = get_compatible_dataset(if_train_c, if_val_c, False)
    if_train_c.to_json('./dataset/deep/comp_instafake_df_train.json')
    if_val_c.to_json('./dataset/deep/comp_instafake_df_val.json')

    # IJECE compatible dataset
    ijece_train_c = ijece_train
    ijece_val_c = ijece_val
    ijece_train_c, ijece_val_c = get_compatible_dataset(ijece_train_c, ijece_val_c, True)
    ijece_train_c.to_json('./dataset/deep/comp_ijece_df_train.json')
    ijece_train_c.to_json('./dataset/deep/comp_ijece_df_val.json')

    # Balanced joined dataset
    random.shuffle(fake_csv)
    random.shuffle(correct_csv)
    ijece_correct_part = correct_csv[:700]
    ijece_fake_part = fake_csv[:700]
    partial_fake = fake_json + ijece_fake_part
    partial_correct = correct_json + ijece_correct_part
    partial_train, partial_val = treat_combined(partial_fake, partial_correct)
    partial_train.to_json(f'./dataset/deep/combo_partial_df_train.json')
    partial_val.to_json(f'./dataset/deep/combo_partial_df_val.json')

    # Unbalanced joined dataset
    combined_fake = fake_json + fake_csv
    combined_correct = correct_json + correct_csv
    combined_train, combined_val = treat_combined(combined_fake, combined_correct)
    combined_train.to_json(f'./dataset/deep/combo_full_df_train.json')
    combined_val.to_json(f'./dataset/deep/combo_full_df_val.json')


def treat_combined(fake, correct, demarcator=700):
    if_dataset_fake, if_dataset_correct = get_compatible_dataset(pd.DataFrame(data=fake[:demarcator]),
                                                                 pd.DataFrame(data=correct[:demarcator]),
                                                                 False)
    ijece_dataset_fake, ijece_dataset_correct = get_compatible_dataset(pd.DataFrame(data=fake[demarcator:]),
                                                                       pd.DataFrame(data=correct[demarcator:]),
                                                                       True)
    custom_train_df, custom_val_df = shuffle_and_split(
        pd.concat([if_dataset_fake, ijece_dataset_fake]).to_dict('records'),
        pd.concat([if_dataset_correct, ijece_dataset_correct]).to_dict('records'))
    return custom_train_df, custom_val_df
