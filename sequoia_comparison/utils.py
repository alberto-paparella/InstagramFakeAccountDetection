import random
import pandas as pd
from sklearn import tree, metrics
from sklearn.linear_model import LogisticRegression

PERCENT_TRAIN = 70

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


def get_custom_dataset(train_df, validation_df, csv=False):
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
    return custom_train_df, custom_validation_df


def get_scores(y_val, y_pred):
    scores = {
        'precision': 0, 'accuracy': 0
    }
    scores['accuracy'] += metrics.accuracy_score(y_val, y_pred)
    scores['precision'] += metrics.precision_score(y_val, y_pred)
    return scores


def print_scores(train_df, validation_df, csv=False):
    scores = get_scores(train_df, validation_df, csv)
    print('precision:', "{:.3f}".format(scores['precision']))
    print('accuracy:', "{:.3f}".format(scores['accuracy']))

'''
modes:
 - "dt" => DecisionTree
 - "lr" => LogisticRegression
'''
def experiment(fake, correct, csv=False, mode="dt", n_iter=20):
    avg_scores = {
        'default': {'precision': 0, 'accuracy': 0},
        'custom': {'precision': 0, 'accuracy': 0}
    }

    if mode == "dt":
        print(f"Calculating precision and accuracy metrics for Decision Trees over {n_iter} times")
    elif mode == "lr":
        print(f"Calculating precision and accuracy metrics for Decision Trees over {n_iter} times")
    else:
        return -1

    for i in range(n_iter):
        # Get new train_df and validation_df, same for default and custom
        train_df, validation_df = shuffle_and_split(fake, correct)
        custom_train_df, custom_validation_df = get_custom_dataset(train_df, validation_df, csv)

        # Default mode
        if mode == "dt":
            # Get new Decision Tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(train_df.iloc[:, :-2], train_df.iloc[:, -1])
        elif mode == "lr":
            # Get new Logistic Regressor
            clf = LogisticRegression(random_state=0, max_iter=5000)
            clf = clf.fit(train_df.iloc[:, :-2], train_df.iloc[:, -1])

        # Get ground truth and predictions to measure performance
        X_val, y_val = validation_df.iloc[:, :-2], validation_df.iloc[:, -1]
        y_pred = clf.predict(X_val)

        # Default scores
        scores = get_scores(y_val, y_pred)
        avg_scores['default']['precision'] += scores['precision']
        avg_scores['default']['accuracy'] += scores['accuracy']

        # Custom mode
        if mode == "dt":
            # Get new Decision Tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(custom_train_df.iloc[:, :-2], custom_train_df.iloc[:, -1])
        elif mode == "lr":
            # Get new Logistic Regressor
            clf = LogisticRegression(random_state=0, max_iter=2500)
            clf = clf.fit(custom_train_df.iloc[:, :-2], custom_train_df.iloc[:, -1])
        else:
            return -1
        
        # Get ground truth and predictions to measure performance
        X_val, y_val = custom_validation_df.iloc[:, :-2], custom_validation_df.iloc[:, -1]
        y_pred = clf.predict(X_val)

        # Custom scores
        scores = get_scores(y_val, y_pred)
        avg_scores['custom']['precision'] += scores['precision']
        avg_scores['custom']['accuracy'] += scores['accuracy']

        print(f"{i+1}/{n_iter}", end="\r")

    # Averaging
    for t in avg_scores.keys():
        for s in avg_scores[t].keys():
            avg_scores[t][s] /= n_iter

    print('Done!\n\n')

    print('default avg precision:', "{:.3f}".format(avg_scores['default']['precision']))
    print('default avg accuracy:', "{:.3f}".format(avg_scores['default']['accuracy']))

    print('custom avg precision:', "{:.3f}".format(avg_scores['custom']['precision']))
    print('custom avg accuracy:', "{:.3f}".format(avg_scores['custom']['accuracy']))