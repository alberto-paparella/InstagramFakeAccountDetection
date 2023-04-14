# Caricare il dataset nel formato .csv
# Caricare il dataset nel formato parziale
from dataset.normalizer import csv_importer_full
from sequoia_comparison.utils import find_demarcator, print_scores

default_dataset = csv_importer_full("dataset/sources/user_fake_authentic_2class.csv")
idx = find_demarcator(default_dataset)

fake = default_dataset[:idx]
correct = default_dataset[idx:]

print_scores(fake, correct)
