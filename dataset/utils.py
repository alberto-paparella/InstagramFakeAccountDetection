from dataset.normalizer import csv_importer_full, json_importer_full
import random
import pandas as pd

PERCENT_TRAIN = 70


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


def get_combined_datasets():
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


def get_compatible_dataset(train_df, validation_df, csv):
    if csv:
        custom_train_df = train_df.drop(["pic", "cl", "cz", "ni", "lt", "pr", "fo", "cs"], axis=1)
        custom_validation_df = validation_df.drop(["pic", "cl", "cz", "ni", "lt", "pr", "fo", "cs"], axis=1)
    else:
        custom_train_df = train_df.drop(["mediaLikeNumbers",
                                         "userHasHighlighReels", "usernameLength", "usernameDigitCount",
                                         "mediaCommentNumbers", "mediaCommentsAreDisabled",
                                         "mediaHasLocationInfo", "userTagsCount"], axis=1)
        custom_validation_df = validation_df.drop(["mediaLikeNumbers",
                                                   "userHasHighlighReels", "usernameLength", "usernameDigitCount",
                                                   "mediaCommentNumbers", "mediaCommentsAreDisabled",
                                                   "mediaHasLocationInfo",
                                                   "userTagsCount"],
                                                  axis=1)
    return custom_train_df, custom_validation_df


def get_ijece_custom_dataset(train_df, validation_df):
    custom_train_df = train_df.drop(["ni", "lt", "mediaHashtagNumbers"], axis=1)
    custom_validation_df = validation_df.drop(["ni", "lt", "mediaHashtagNumbers"], axis=1)
    return custom_train_df, custom_validation_df


# TODO: attributi in comune sui singoli dataset e guarda quali funzionano meglio tra IF_Cus, IJECE_Cus,
# IF_comp e IJECE_comp


def get_custom_dataset(train_df, validation_df, csv):
    if csv:
        return get_ijece_custom_dataset(train_df, validation_df)
    else:
        return train_df, validation_df


def get_ijece_default_dataset(train_df, validation_df):
    custom_train_df = train_df.drop(["followerToFollowing", "hasMedia"], axis=1)
    custom_validation_df = validation_df.drop(["followerToFollowing", "hasMedia"], axis=1)
    return custom_train_df, custom_validation_df


def get_default_dataset(train_df, validation_df, csv):
    if csv:
        return get_ijece_default_dataset(train_df, validation_df)
    else:
        return get_instafake_default_dataset(train_df, validation_df)


def get_instafake_default_dataset(train_df, validation_df):
    """
    Da tenere: nmedia, follower, following, HasHighlightReels, Number of tags,
    average hashtag numbers, Average media likes,
    FFR, media or not
    :param train_df:
    :param validation_df:
    :return:
    """
    default_train_df = train_df.drop(["biol",
                                      "url", "erl", "erc", "avgtime",
                                      "userHasHighlighReels", "usernameLength", "usernameDigitCount",
                                      "mediaCommentNumbers", "mediaCommentsAreDisabled", "mediaHasLocationInfo",
                                      "usernameLength", "usernameDigitCount"], axis=1)
    default_validation_df = validation_df.drop(["biol",
                                                "url", "erl", "erc", "avgtime",
                                                "userHasHighlighReels", "usernameLength", "usernameDigitCount",
                                                "mediaCommentNumbers", "mediaCommentsAreDisabled",
                                                "mediaHasLocationInfo",
                                                "usernameLength", "usernameDigitCount"],
                                               axis=1)
    return default_train_df, default_validation_df


def get_deep_learning_dataset():
    fake_csv, correct_csv = get_fake_correct_default(True)
    fake_json, correct_json = get_fake_correct_default(False)

    # IJECE dataset
    ijece_train, ijece_val = shuffle_and_split(fake_csv, correct_csv)
    # ijece_train.to_csv(f'./dataset/deep/IJECE_df_train.csv')
    # ijece_val.to_csv(f'./dataset/deep/IJECE_df_val.csv')

    # InstaFake dataset
    if_train, if_val = shuffle_and_split(fake_json, correct_json)
    # if_train.to_json(f'./dataset/deep/instafake_df_train.json')
    # if_val.to_json(f'./dataset/deep/instafake_df_val.json')

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
    partial_train, partial_validation = treat_combined(partial_fake, partial_correct)
    # partial_train.to_json(f'./dataset/deep/combo_partial_df_train.json')
    # partial_validation.to_json(f'./dataset/deep/combo_partial_df_val.json')

    # Unbalanced joined dataset
    combined_fake = fake_json + fake_csv
    combined_correct = correct_json + correct_csv
    combined_train, combined_validation = treat_combined(combined_fake, combined_correct)
    # combined_train.to_json(f'./dataset/deep/combo_full_df_train.json')
    # combined_validation.to_json(f'./dataset/deep/combo_full_df_val.json')


def treat_combined(fake, correct, demarcator=700):
    if_dataset_fake, if_dataset_correct = get_compatible_dataset(pd.DataFrame(data=fake[:demarcator]),
                                                                 pd.DataFrame(data=correct[:demarcator]),
                                                                 False)
    ijece_dataset_fake, ijece_dataset_correct = get_compatible_dataset(pd.DataFrame(data=fake[demarcator:]),
                                                                       pd.DataFrame(data=correct[demarcator:]),
                                                                       True)
    custom_train_df, custom_validation_df = shuffle_and_split(
        pd.concat([if_dataset_fake, ijece_dataset_fake]).to_dict('records'),
        pd.concat([if_dataset_correct, ijece_dataset_correct]).to_dict('records'))
    return custom_train_df, custom_validation_df
