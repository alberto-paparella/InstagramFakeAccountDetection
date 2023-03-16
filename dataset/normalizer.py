"""
This script creates a single dataset from two different ones, and exports it in .json format.
"""
import csv
import json

"""
url: url in bio
biol: lunghezza bio
erc : numero commenti / numero media / numero follower
ahc: numero di hashtag medi
nmedia: numero media
avgtime: intervallo medio tra un post e un altro (in ore)
nfollowing: numero di persone seguite
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
                {"nmedia": row[0], "biol": row[3], "url": row[5], "erc": row[10], "ahc": row[12], "avgtime": row[16],
                 "nfollowing": row[2], "fake": (True if row[17] == "f" else False)})
    print(f"Loaded {counter} entries from source {filename}")
    return result


def compute_erc(num_comments, num_media, num_followers) -> float:
    result = 0
    try:
        result = num_comments / num_media / num_followers
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
        acc += times[i] - times[i + 1]
    return acc / length


def json_importer(filename: str, fake=False) -> list:
    result = []
    print(f"Now loading from file {filename}...")
    with open(filename, "r") as json_source:
        data = json.load(json_source)
        size = len(data)
        counter = 0
        for row in data:
            print(f"{counter}/{size}       ", end="\r")
            counter+=1
            result.append(
                {"nmedia": row["userMediaCount"], "biol": row["userBiographyLength"], "url": row["userHasExternalUrl"],
                 "erc": compute_erc(sum(row["mediaCommentNumbers"]), row["userMediaCount"], row["userFollowerCount"]),
                 "ahc": compute_ahc(row["mediaHashtagNumbers"], row["userMediaCount"]),
                 "avgtime": compute_avg_time(row["mediaUploadTimes"]), "nfollowing": row["userFollowingCount"],
                 "fake": fake})
        print(f"Loaded {counter} entries from source {filename}")
    return result


result = csv_importer("./sources/user_fake_authentic_2class.csv")
result += json_importer("./sources/automatedAccountData.json", True)
result += json_importer("./sources/nonautomatedAccountData.json", False)
print(f"Done loading. {len(result)} entries have been loaded up.")
print("Now saving to output file...")
with open("output.json", "w") as output_file:
    json.dump(result, output_file)
print("Done.")
