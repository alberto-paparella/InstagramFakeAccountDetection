from dataset.normalizer import json_importer_full
from sequoia_comparison.utils import shuffle_and_split, print_scores

fake = json_importer_full("dataset/sources/automatedAccountData.json", True)
correct = json_importer_full("dataset/sources/nonautomatedAccountData.json", False)

print_scores(fake, correct, True)
