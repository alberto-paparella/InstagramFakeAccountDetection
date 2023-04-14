import random
import pandas as pd
from sklearn import tree, metrics

PERCENT_TRAIN = 70
LOOPS = 20


def shuffle_and_split(ds_fake, ds_correct):
    # print(f"Now splitting dataset with ratio {PERCENT_TRAIN}:{100 - PERCENT_TRAIN}")
    random.shuffle(ds_fake)
    random.shuffle(ds_correct)

    ds_train = ds_fake[:int(len(ds_fake) * (PERCENT_TRAIN / 100))]
    ds_train += ds_correct[:int(len(ds_correct) * (PERCENT_TRAIN / 100))]

    ds_validation = ds_fake[int(len(ds_fake) * (PERCENT_TRAIN / 100)):]
    ds_validation += ds_correct[int(len(ds_correct) * (PERCENT_TRAIN / 100)):]

    random.shuffle(ds_train)
    random.shuffle(ds_validation)

    # print("Loading complete.")

    df_train = pd.DataFrame.from_dict(ds_train)
    df_validation = pd.DataFrame.from_dict(ds_validation)
    # print(df_train)
    # print(df_validation)
    return df_train, df_validation


def find_demarcator(dataset):
    """
    Restituisce l'indice del primo elemento non fake
    :param dataset: il dataset
    :return: l'indice
    """
    idx = 0
    for elem in dataset:
        if elem['fake'] == 1:
            idx += 1
        else:
            break
    return idx


def fit_decision_tree(X, y, validation_df):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    # print("Fitting complete.")

    X_val, y_val = validation_df.iloc[:, :-2], validation_df.iloc[:, -1]
    y_pred = clf.predict(X_val)
    return y_val, y_pred


def print_classification_report(X, y, validation_df):
    y_val, y_pred = fit_decision_tree(X, y, validation_df)
    # y_compare = y_pred - y_val
    # print('accuracy =', 100 - (sum(abs(y_compare)) / len(validation_df.index)) * 100)
    print(metrics.classification_report(y_val, y_pred))
    print(type(metrics.classification_report(y_val, y_pred)))


def get_scores(fake, correct, csv):
    somme = {
        'custom': {'precision': 0, 'accuracy': 0},
        'default': {'precision': 0, 'accuracy': 0}
    }
    print(f"Calculating precision and accuracy {LOOPS} times")
    for i in range(LOOPS):
        train_df, validation_df = shuffle_and_split(fake, correct)
        if csv:
            custom_train_df = train_df.drop(["mediaLikeNumbers", "mediaCommentNumbers",
                                             "mediaCommentsAreDisabled", "mediaHashtagNumbers", "mediaHasLocationInfo",
                                             "userHasHighlighReels", "usernameLength", "usernameDigitCount"], axis=1)
            custom_validation_df = validation_df.drop(["mediaLikeNumbers", "mediaCommentNumbers",
                                                       "mediaCommentsAreDisabled", "mediaHashtagNumbers",
                                                       "mediaHasLocationInfo",
                                                       "userHasHighlighReels", "usernameLength", "usernameDigitCount"],
                                                      axis=1)
        else:
            custom_train_df = train_df.drop(["pic", "cl", "cz", "ni", "lt", "ahc", "pr", "fo", "cs"], axis=1)
            custom_validation_df = validation_df.drop(["pic", "cl", "cz", "ni", "lt", "ahc", "pr", "fo", "cs"], axis=1)
        y_val, y_pred = fit_decision_tree(train_df.iloc[:, :-2], train_df.iloc[:, -1], validation_df)
        cy_val, cy_pred = fit_decision_tree(custom_train_df.iloc[:, :-2], custom_train_df.iloc[:, -1],
                                            custom_validation_df)

        somme['default']['accuracy'] += metrics.accuracy_score(y_val, y_pred)
        somme['default']['precision'] += metrics.precision_score(y_val, y_pred)
        somme['custom']['accuracy'] += metrics.accuracy_score(cy_val, cy_pred)
        somme['custom']['precision'] += metrics.precision_score(cy_val, cy_pred)

        print(f"{i+1}/{LOOPS}", end="\r")

    for t in somme.keys():
        for s in somme[t].keys():
            somme[t][s] /= LOOPS

    print('Done!\n\n')
    return somme


def print_scores(fake, correct, csv=False):
    scores = get_scores(fake, correct, csv)
    print('default precision:', "{:.3f}".format(scores['default']['precision']))
    print('custom precision:', "{:.3f}".format(scores['custom']['precision']))
    print('default accuracy:', "{:.3f}".format(scores['default']['accuracy']))
    print('custom accuracy:', "{:.3f}".format(scores['custom']['accuracy']))
