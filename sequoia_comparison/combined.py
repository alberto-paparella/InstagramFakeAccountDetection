from dataset.normalizer import json_importer_full, csv_importer_full
from dataset.utils import find_demarcator
from utils.utils import experiment
import random

n_exp = 5

fake_if = json_importer_full("../dataset/sources/automatedAccountData.json", True)
correct_if = json_importer_full("../dataset/sources/nonautomatedAccountData.json", False)

ijece = csv_importer_full("../dataset/sources/user_fake_authentic_2class.csv")
idx = find_demarcator(ijece)

experiments = ["dt", "lr", "nb", "rf", "dl"]

dataset = {"full": dict(), "partial": dict()}

dataset["full"]["fake"] = fake_if + ijece[:idx]
dataset["full"]["correct"] = correct_if + ijece[idx:]
ijece_fake = ijece[:idx]
random.shuffle(ijece_fake)
ijece_correct = ijece[idx:]
random.shuffle(ijece_correct)
dataset["partial"]["fake"] = fake_if + ijece_fake[:700]
dataset["partial"]["correct"] = correct_if + ijece_correct[:700]

print("\n================\n Running experiment on limited dataset... \n================\n")

for exp in experiments:
    experiment(dataset["partial"]["fake"], dataset["partial"]["correct"],
               csv=False, mode=exp, n_iter=n_exp, combine=True)

print("\n================\n Running experiment on full dataset... \n================\n")

for exp in experiments:
    experiment(dataset["full"]["fake"], dataset["full"]["correct"],
               csv=False, mode=exp, n_iter=n_exp, combine=True)
