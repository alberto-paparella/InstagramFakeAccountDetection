from dataset.normalizer import json_importer_full, csv_importer_full
from dataset.utils import find_demarcator
from utils.utils import experiment

n_exp = 5

fake = json_importer_full("../dataset/sources/automatedAccountData.json", True)
correct = json_importer_full("../dataset/sources/nonautomatedAccountData.json", False)

default_dataset = csv_importer_full("../dataset/sources/user_fake_authentic_2class.csv")
idx = find_demarcator(default_dataset)

fake = fake + default_dataset[:idx]
correct = correct + default_dataset[idx:]
# experiment(fake, correct, csv=False, mode="dt", n_iter=n_exp, combine=True)
# experiment(fake, correct, csv=False, mode="lr", n_iter=n_exp, combine=True)
# experiment(fake, correct, csv=False, mode="nb", n_iter=n_exp, combine=True)
# experiment(fake, correct, csv=False, mode="rf", n_iter=n_exp, combine=True)
experiment(fake, correct, csv=False, mode="dl", n_iter=n_exp, combine=True)