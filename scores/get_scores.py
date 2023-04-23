from dataset.utils import shuffle_and_split, get_fake_correct_default, get_custom_dataset

LOOPS = 10


def get_scores(get_single_scores, csv):
    # Experiment
    avg_scores = {
        'default': {'precision': 0, 'accuracy': 0},
        'custom': {'precision': 0, 'accuracy': 0}
    }
    fake, correct = get_fake_correct_default(csv)
    print(f"Calculating precision and accuracy {LOOPS} times")

    for i in range(LOOPS):
        train_df, validation_df = shuffle_and_split(fake, correct)
        custom_train_df, custom_validation_df = get_custom_dataset(train_df, validation_df, csv)

        # Default scores
        scores = get_single_scores(train_df, validation_df)
        avg_scores['default']['precision'] += scores['precision']
        avg_scores['default']['accuracy'] += scores['accuracy']

        # Custom scores
        scores = get_single_scores(custom_train_df, custom_validation_df)
        avg_scores['custom']['precision'] += scores['precision']
        avg_scores['custom']['accuracy'] += scores['accuracy']

        print(f"{i+1}/{LOOPS}", end="\r")

    # Averaging
    for t in avg_scores.keys():
        for s in avg_scores[t].keys():
            avg_scores[t][s] /= LOOPS

    return avg_scores


def print_avg_scores(get_single_scores, csv):
    avg_scores = get_scores(get_single_scores, csv)
    print('Done!\n\n')

    print('default avg precision:', "{:.3f}".format(avg_scores['default']['precision']))
    print('default avg accuracy:', "{:.3f}".format(avg_scores['default']['accuracy']))

    print('custom avg precision:', "{:.3f}".format(avg_scores['custom']['precision']))
    print('custom avg accuracy:', "{:.3f}".format(avg_scores['custom']['accuracy']))
