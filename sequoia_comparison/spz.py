import os,sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from dataset.normalizer import json_importer_full
from sequoia_comparison.utils import experiment

n_exp = 500

fake = json_importer_full("dataset/sources/automatedAccountData.json", True)
correct = json_importer_full("dataset/sources/nonautomatedAccountData.json", False)

# Experiments
experiment(fake, correct, csv=False, mode="dt", n_iter=n_exp)   # DecisionTree
experiment(fake, correct, csv=False, mode="lr", n_iter=n_exp)   # LogisticRegression
experiment(fake, correct, csv=False, mode="nb", n_iter=n_exp)   # NaiveBayes (LogisticRegression)