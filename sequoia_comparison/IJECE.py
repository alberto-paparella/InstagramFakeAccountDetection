import os,sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from dataset.normalizer import csv_importer_full
from sequoia_comparison.utils import find_demarcator, experiment

LOOPS = 500

default_dataset = csv_importer_full("dataset/sources/user_fake_authentic_2class.csv")
idx = find_demarcator(default_dataset)

fake = default_dataset[:idx]
real = default_dataset[idx:]

# Experiments
experiment(fake, real, "dt", 20)   # DecisionTree
experiment(fake, real, "lr", 20)   # LogisticRegression