from dataset.normalizer import json_importer_full, csv_importer_full
from dataset.utils import find_demarcator
from utils.utils import experiment
import random

n_exp = 5

fake = json_importer_full("../dataset/sources/automatedAccountData.json", True)
correct = json_importer_full("../dataset/sources/nonautomatedAccountData.json", False)

default_dataset = csv_importer_full("../dataset/sources/user_fake_authentic_2class.csv")
idx = find_demarcator(default_dataset)

experiments = ["dt", "lr", "nb", "rf", "dl"]

dataset = {"full": dict(), "partial": dict()}

dataset["full"]["fake"] = fake + default_dataset[:idx]
dataset["full"]["correct"] = correct + default_dataset[idx:]
d_d_f = default_dataset[:idx]
random.shuffle(d_d_f)
d_d_c = default_dataset[idx:]
random.shuffle(d_d_c)
dataset["partial"]["fake"] = fake + d_d_f[:700]
dataset["partial"]["correct"] = correct + d_d_c[:700]

print("\n================\n Running experiment on limited dataset... \n================\n")

for exp in experiments:
    experiment(dataset["partial"]["fake"], dataset["partial"]["correct"],
               csv=False, mode=exp, n_iter=n_exp, combine=True)

print("\n================\n Running experiment on full dataset... \n================\n")

for exp in experiments:
    experiment(dataset["full"]["fake"], dataset["full"]["correct"],
               csv=False, mode=exp, n_iter=n_exp, combine=True)
