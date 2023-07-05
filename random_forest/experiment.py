from sklearn.ensemble import RandomForestClassifier
from dataset.utils import shuffle_and_split, get_fake_correct_default, get_custom_dataset, get_default_dataset
import pandas as pd

from utils.utils import get_scores

LOOPS = 5


def my_get_scores(csv, combine=False):
    # Experiment
    avg_scores = {
        'default': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
        'custom': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    }
    fake, correct = get_fake_correct_default(csv)
    on = 'IJECE' if csv else 'spz'
    c_mode = 'activated' if combine else 'not activated'
    print(
        f"\nCalculating precision, accuracy, recall and f1 score {LOOPS} times on {on} and with combined mode {c_mode} "
        f"with the Random Forest algorithm."
    )

    for i in range(LOOPS):
        # Get new train_df and validation_df, same for default and custom
        if not combine:
            train_df, validation_df = shuffle_and_split(fake, correct)
            custom_train_df, custom_validation_df = get_custom_dataset(train_df, validation_df, csv)
            default_train_df, default_validation_df = get_default_dataset(train_df, validation_df, csv)

            clf = RandomForestClassifier(max_depth=2, random_state=0)
            clf = clf.fit(default_train_df.iloc[:, :-1], default_train_df.iloc[:, -1])

            X_val, y_val = default_validation_df.iloc[:, :-1], default_validation_df.iloc[:, -1]

            y_pred = clf.predict(X_val)

            # Default scores
            scores = get_scores(y_val, y_pred)
            avg_scores['default']['accuracy'] += scores['accuracy']
            avg_scores['default']['precision'] += scores['precision']
            avg_scores['default']['recall'] += scores['recall']
            avg_scores['default']['f1'] += scores['f1']

        else:
            demarcator = 700
            spz_dataset_fake, spz_dataset_correct = get_custom_dataset(pd.DataFrame(data=fake[:demarcator]),
                                                                       pd.DataFrame(data=correct[:demarcator]), False)
            ijece_dataset_fake, ijece_dataset_correct = get_custom_dataset(pd.DataFrame(data=fake[demarcator:]),
                                                                           pd.DataFrame(data=correct[demarcator:]),
                                                                           True)
            custom_train_df, custom_validation_df = shuffle_and_split(
                pd.concat([spz_dataset_fake, ijece_dataset_fake]).to_dict('records'),
                pd.concat([spz_dataset_correct, ijece_dataset_correct]).to_dict('records'))

        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf = clf.fit(custom_train_df.iloc[:, :-1], custom_train_df.iloc[:, -1])

        X_val, y_val = custom_validation_df.iloc[:, :-1], custom_validation_df.iloc[:, -1]

        y_pred = clf.predict(X_val)

        # Custom scores
        scores = get_scores(y_val, y_pred)
        avg_scores['custom']['accuracy'] += scores['accuracy']
        avg_scores['custom']['precision'] += scores['precision']
        avg_scores['custom']['recall'] += scores['recall']
        avg_scores['custom']['f1'] += scores['f1']
        print(f"{i + 1}/{LOOPS}", end="\r")

    # Averaging
    for t in avg_scores.keys():
        for s in avg_scores[t].keys():
            avg_scores[t][s] /= LOOPS

    return avg_scores


def print_avg_scores(csv):
    avg_scores = my_get_scores(csv)

    print('PRECISION\tDefault:', "{:.2f}".format(avg_scores['default']['precision']), '\tCustom:',
          "{:.2f}".format(avg_scores['custom']['precision']))
    print('ACCURACY\tDefault:', "{:.2f}".format(avg_scores['default']['accuracy']), '\tCustom:',
          "{:.2f}".format(avg_scores['custom']['accuracy']))
    print('RECALL\t\tDefault:', "{:.2f}".format(avg_scores['default']['recall']), '\tCustom:',
          "{:.2f}".format(avg_scores['custom']['recall']))
    print('F1 SCORE\tDefault:', "{:.2f}".format(avg_scores['default']['f1']), '\tCustom:',
          "{:.2f}".format(avg_scores['custom']['f1']))


print_avg_scores(True)
print_avg_scores(False)
# my_get_scores(True, True)
