"""
This script creates a single dataset from two different ones, and exports it in .json format.
"""
import csv
import json
import datetime
import random

"""
url: url in bio
biol: lunghezza bio
erc : numero commenti / numero media / numero follower
erl : numero like / numero media / numero follower
nmedia: numero media
avgtime: intervallo medio tra un post e un altro (in ore)
nfollowing: numero di persone seguite
nfollower: numero di follower - da mettere!
fake: se l'utente è fake o meno
"""


def csv_importer(filename: str) -> list:
    result = []
    print(f"Now loading from file {filename}...")
    with open(filename, 'r') as csv_source:
        reader = csv.reader(csv_source)
        counter = 0
        for row in reader:
            print(f"{counter}        ", end="\r")
            if counter == 0:
                counter += 1
                continue
            counter += 1
            result.append(
                {"nmedia": float(row[0]), "biol": float(row[3]), "url": float(row[5]), "erl": float(row[10]),
                 "erc": float(row[10]), "avgtime": float(row[16]), "nfollowing": float(row[2]),
                 "nfollower": float(row[1]), "fake": (1 if row[17] == "f" else 0)})
    print(f"Loaded {counter} entries from source {filename}")
    return result


def csv_importer_full(filename, verbose=True):
    result = []
    if verbose:
        print(f"Now loading from file {filename}...")
    with open(filename, 'r') as csv_source:
        reader = csv.reader(csv_source)
        counter = 0
        for row in reader:
            if verbose:
                print(f"{counter}        ", end="\r")
            if counter == 0:
                counter += 1
                continue
            counter += 1
            result.append(
                {"nmedia": float(row[0]), "nfollower": float(row[1]), "nfollowing": float(row[2]),
                 "biol": float(row[3]),
                 "pic": float(row[4]),
                 "url": float(row[5]), "cl": float(row[6]), "cz": float(row[7]), "ni": float(row[8]),
                 "erl": float(row[9]), "erc": float(row[10]),
                 "lt": float(row[11]), "mediaHashtagNumbers": float(row[12]), "pr": float(row[13]), "fo": float(row[14]),
                 "cs": float(row[15]), "avgtime": float(row[16]),
                 "followerToFollowing": (float(row[1])/float(row[2]) if float(row[2]) != 0 else 0),
                 "hasMedia": (1 if float(row[0]) else 0),
                 "fake": (1 if row[17] == "f" else 0)})
    print(f"Loaded {counter} entries from source {filename}")
    return result


def compute_erc(num_comments, num_media, num_followers) -> float:
    result = 0
    try:
        result = num_comments / num_media / num_followers
    except ZeroDivisionError:
        return 0
    return result


def compute_erl(num_likes, num_media, num_followers) -> float:
    result = 0
    try:
        result = num_likes / num_media / num_followers
    except ZeroDivisionError:
        return 0
    return result


def compute_ahc(ht_numbers, media_count) -> float:
    result = 0
    try:
        sum(ht_numbers) / media_count
    except ZeroDivisionError:
        return 0
    return result


def compute_avg_time(times) -> float:
    length = len(times)
    if not length:
        return 0
    acc = 0
    for i in range(len(times) - 1):
        time1 = datetime.datetime.fromtimestamp(times[i])
        time2 = datetime.datetime.fromtimestamp(times[i + 1])
        acc += (time1 - time2).total_seconds()
    return (acc / 3600) / length


def json_importer(filename: str, fake=False) -> list:
    result = []
    print(f"Now loading from file {filename}...")
    with open(filename, "r") as json_source:
        data = json.load(json_source)
        size = len(data)
        counter = 0
        for row in data:
            print(f"{counter}/{size}       ", end="\r")
            counter += 1
            result.append(
                {"nmedia": row["userMediaCount"], "biol": row["userBiographyLength"], "url": row["userHasExternalUrl"],
                 "erl": compute_erl(sum(row["mediaLikeNumbers"]), row["userMediaCount"], row["userFollowerCount"]),
                 "erc": compute_erc(sum(row["mediaCommentNumbers"]), row["userMediaCount"], row["userFollowerCount"]),
                 "avgtime": compute_avg_time(row["mediaUploadTimes"]), "nfollowing": row["userFollowingCount"],
                 "nfollower": row["userFollowerCount"], "fake": (1 if fake else 0)})
        print(f"Loaded {counter} entries from source {filename}")
    return result


def json_importer_full(filename: str, fake=False, verbose=True) -> list:
    result = []
    if verbose:
        print(f"Now loading from file {filename}...")
    with open(filename, "r") as json_source:
        data = json.load(json_source)
        size = len(data)
        counter = 0
        for row in data:
            print(f"{counter}/{size}       ", end="\r")
            counter += 1
            result.append(
                {"nmedia": row["userMediaCount"],
                 "biol": row["userBiographyLength"],
                 "url": row["userHasExternalUrl"],
                 "nfollowing": row["userFollowingCount"],
                 "nfollower": row["userFollowerCount"],
                 "erl": compute_erl(sum(row["mediaLikeNumbers"]), row["userMediaCount"], row["userFollowerCount"]),
                 "erc": compute_erc(sum(row["mediaCommentNumbers"]), row["userMediaCount"], row["userFollowerCount"]),
                 "avgtime": compute_avg_time(row["mediaUploadTimes"]),
                 "mediaLikeNumbers": (
                     sum(row["mediaLikeNumbers"]) / sum(row["mediaCommentNumbers"]) if sum(
                         row["mediaCommentNumbers"]) != 0 else 0),
                 "mediaHashtagNumbers": (
                     sum(row["mediaHashtagNumbers"]) / row["userMediaCount"] if row['userMediaCount'] != 0 else 0),
                 "followerToFollowing": (
                     row["userFollowerCount"] / row["userFollowingCount"] if row['userFollowingCount'] != 0 else 0),
                 "mediaCommentNumbers": (
                     sum(row["mediaCommentNumbers"]) / row["userMediaCount"] if row['userMediaCount'] != 0 else 0),
                 "mediaCommentsAreDisabled": (
                     sum(row["mediaCommentsAreDisabled"]) / row["userMediaCount"] if row['userMediaCount'] != 0 else 0),
                 "mediaHasLocationInfo": (
                     sum(row["mediaHasLocationInfo"]) / row["userMediaCount"] if row["userMediaCount"] != 0 else 0),
                 "hasMedia": (1 if row["userMediaCount"] else 0),
                 "userHasHighlighReels": row["userHasHighlighReels"],
                 "userTagsCount": row["userTagsCount"],
                 "usernameLength": row["usernameLength"],
                 "usernameDigitCount": row["usernameDigitCount"],
                 "fake": (1 if fake else 0)})
        if verbose:
            print(f"Loaded {counter} entries from source {filename}")
    return result


if __name__ == "__main__":
    result = csv_importer("./sources/user_fake_authentic_2class.csv")
    result += json_importer("./sources/automatedAccountData.json", True)
    result += json_importer("./sources/nonautomatedAccountData.json", False)
    print(f"Done loading. {len(result)} entries have been loaded up.")
    print("Now shuffling...")
    random.shuffle(result)
    train_ratio = 70
    print(f"Now splitting dataset into train and validation with ratio {train_ratio}:{100 - train_ratio}")
    train_dataset = result[:int(len(result) * (train_ratio / 100))]
    validation_dataset = result[int(len(result) * (train_ratio / 100)):]

    print("Now saving to output file...")
    with open("train.json", "w") as output_file:
        json.dump(train_dataset, output_file)

    with open("validation.json", "w") as output_file:
        json.dump(validation_dataset, output_file)

    print("Done.")
