# This module is responsable of importing and pre-treating the datasets.

import csv
import json
import datetime
import random


def csv_importer_full(filename, verbose=True):
    """
    Importer for the IJECE dataset
    :param filename: file from which it loads
    :param verbose: be quiet
    :return: dataset
    """
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
    """
    Computes erc
    :param num_comments:
    :param num_media:
    :param num_followers:
    :return: erc
    """
    result = 0
    try:
        result = num_comments / num_media / num_followers
    except ZeroDivisionError:
        return 0
    return result


def compute_erl(num_likes, num_media, num_followers) -> float:
    """
    Computes erl
    :param num_likes:
    :param num_media:
    :param num_followers:
    :return: erl
    """
    result = 0
    try:
        result = num_likes / num_media / num_followers
    except ZeroDivisionError:
        return 0
    return result


def compute_avg_time(times) -> float:
    """
    Computes average times
    :param times: array of timestamps
    :return: the average
    """
    length = len(times)
    if not length:
        return 0
    acc = 0
    for i in range(len(times) - 1):
        time1 = datetime.datetime.fromtimestamp(times[i])
        time2 = datetime.datetime.fromtimestamp(times[i + 1])
        acc += (time1 - time2).total_seconds()
    return (acc / 3600) / length


def json_importer_full(filename: str, fake=False, verbose=True) -> list:
    """
    Importer for IF
    :param filename: the file from which it loads from
    :param fake: should this data be treated as fake or correct?
    :param verbose: be silent or not
    :return: the dataset
    """
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
