import os,sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path

from dataset.normalizer import csv_importer_full
from sequoia_comparison.utils import find_demarcator, shuffle_and_split, get_custom_dataset, get_scores

LOOPS = 500

default_dataset = csv_importer_full("dataset/sources/user_fake_authentic_2class.csv")
idx = find_demarcator(default_dataset)

fake = default_dataset[:idx]
correct = default_dataset[idx:]

### EXPERIMENT ###
avg_scores = {
    'default': {'precision': 0, 'accuracy': 0},
    'custom': {'precision': 0, 'accuracy': 0}
}

print(f"Calculating precision and accuracy {LOOPS} times")

for i in range(LOOPS):
    # Get new train_df and validation_df, same for default and custom
    train_df, validation_df = shuffle_and_split(fake, correct)
    custom_train_df, custom_validation_df = get_custom_dataset(train_df, validation_df)

    # Default scores
    scores = get_scores(train_df, validation_df)
    avg_scores['default']['precision'] += scores['precision']
    avg_scores['default']['accuracy'] += scores['accuracy']

    # Custom scores
    scores = get_scores(custom_train_df, custom_validation_df)
    avg_scores['custom']['precision'] += scores['precision']
    avg_scores['custom']['accuracy'] += scores['accuracy']

    print(f"{i+1}/{LOOPS}", end="\r")

# Averaging
for t in avg_scores.keys():
    for s in avg_scores[t].keys():
        avg_scores[t][s] /= LOOPS

print('Done!\n\n')

print('default avg precision:', "{:.3f}".format(avg_scores['default']['precision']))
print('default avg accuracy:', "{:.3f}".format(avg_scores['default']['accuracy']))

print('custom avg precision:', "{:.3f}".format(avg_scores['custom']['precision']))
print('custom avg accuracy:', "{:.3f}".format(avg_scores['custom']['accuracy']))