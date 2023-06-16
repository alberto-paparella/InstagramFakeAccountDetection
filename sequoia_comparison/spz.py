import os,sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from dataset.normalizer import json_importer_full
from utils.utils import experiment

n_exp = 5

fake = json_importer_full("../dataset/sources/automatedAccountData.json", True)
correct = json_importer_full("../dataset/sources/nonautomatedAccountData.json", False)

# Experiments
#experiment(fake, correct, csv=False, mode="dt", n_iter=n_exp)   # DecisionTree
#experiment(fake, correct, csv=False, mode="lr", n_iter=n_exp)   # LogisticRegression
#experiment(fake, correct, csv=False, mode="nb", n_iter=n_exp)   # NaiveBayes (LogisticRegression)
experiment(fake, correct, csv=False, mode="rf", n_iter=n_exp)   # RandomForest
#experiment(fake, correct, csv=False, mode="dl", n_iter=n_exp)    # Deep Learning

input("Press ENTER to exit...")