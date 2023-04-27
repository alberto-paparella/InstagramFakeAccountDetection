import os,sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from dataset.normalizer import csv_importer_full
from dataset.utils import find_demarcator
from sequoia_comparison.utils import experiment

n_exp = 50 # Number of experiments

default_dataset = csv_importer_full("dataset/sources/user_fake_authentic_2class.csv")
idx = find_demarcator(default_dataset)

fake = default_dataset[:idx]
correct = default_dataset[idx:]

# Experiments
experiment(fake, correct, csv=True, mode="dt", n_iter=n_exp)    # DecisionTree
experiment(fake, correct, csv=True, mode="lr", n_iter=n_exp)    # LogisticRegression
experiment(fake, correct, csv=True, mode="nb", n_iter=n_exp)    # NaiveBayes (LogisticRegression)