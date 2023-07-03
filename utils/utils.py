from dataset.utils import get_custom_dataset, shuffle_and_split, get_default_dataset, get_compatible_dataset
from sklearn import tree, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from deep.IJECE.IJECE_custom import run_model as run_ijce_custom
from deep.IJECE.IJECE_default import run_model as run_ijce_default
from deep.spz.spz_default import run_model as run_spz_default
from deep.spz.spz_custom import run_model as run_spz_custom

import numpy as np
import pandas as pd

PERCENT_TRAIN = 70
MAX_ITER = 5000  # Maximum number of iterations for Logistic Regressors


def fit_decision_tree(X, y, validation_df):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    # print("Fitting complete.")

    X_val, y_val = validation_df.iloc[:, :-1], validation_df.iloc[:, -1]
    y_pred = clf.predict(X_val)
    return y_val, y_pred


def print_classification_report(X, y, validation_df):
    y_val, y_pred = fit_decision_tree(X, y, validation_df)
    # y_compare = y_pred - y_val
    # print('accuracy =', 100 - (sum(abs(y_compare)) / len(validation_df.index)) * 100)
    print(metrics.classification_report(y_val, y_pred))
    print(type(metrics.classification_report(y_val, y_pred)))


def get_scores(y_val, y_pred):
    scores = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    scores['accuracy'] += metrics.accuracy_score(y_val, y_pred)
    scores['precision'] += metrics.precision_score(y_val, y_pred)
    scores['recall'] += metrics.recall_score(y_val, y_pred)
    scores['f1'] += metrics.f1_score(y_val, y_pred)
    return scores


def print_scores(train_df, validation_df):
    scores = get_scores(train_df, validation_df)
    print('precision:', "{:.3f}".format(scores['precision']))
    print('accuracy:', "{:.3f}".format(scores['accuracy']))


'''
basic naive bayes feature equation
'''


def naive_bayes(x, y_i, y):
    """Naive-Bayes function"""
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


def naive_bayes_support_vector_machine(x, y):
    y = y.values
    r = np.log(naive_bayes(x, 1, y) / naive_bayes(x, 0, y))
    # m = LogisticRegression(C=4, dual=True) # This gives an error
    m = LogisticRegression(C=4, dual=False, max_iter=MAX_ITER)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def experiment(fake, correct, csv, mode="dt", n_iter=20, combine=False, demarcator=700):
    '''
    A function which execution an experiment fitting a model `n_iter` times and giving
    back the `avg_scores` for various metrics such as `accuracy`, `precision`, `recall`, ...

    `modes`:
    - `dt` => DecisionTree
    - `lr` => LogisticRegression
    - `nb` => NaiveBayes (NB-SVM, but using LogisticRegression instead)
    - `rf` => RandomForest approach
    - `dl` => DeepLearning approach using neural networks
    '''
    avg_scores = {
        'default': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
        'custom': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    }

    if mode == "dt":
        print(f"Calculating metrics for Decision Trees over {n_iter} times")
    elif mode == "lr":
        print(f"Calculating metrics for Logistic Regression over {n_iter} times")
    elif mode == "nb":
        print(f"Calculating metrics for Naive Bayes (Logistic Regression) over {n_iter} times")
    elif mode == "rf":
        print(f"Calculating metrics for Random Forests over {n_iter} times")
    elif mode == "dl":
        print(f"Calculating metrics for Deep Learning over {n_iter} times")
    else:
        return -1

    for i in range(n_iter):
        # Get new train_df and validation_df, same for default and custom
        if not combine:
            train_df, validation_df = shuffle_and_split(fake, correct)
            custom_train_df, custom_validation_df = get_custom_dataset(train_df, validation_df, csv)
            default_train_df, default_validation_df = get_default_dataset(train_df, validation_df, csv)
        else:
            spz_dataset_fake, spz_dataset_correct = get_compatible_dataset(pd.DataFrame(data=fake[:demarcator]),
                                                                           pd.DataFrame(data=correct[:demarcator]),
                                                                           False)
            ijece_dataset_fake, ijece_dataset_correct = get_compatible_dataset(pd.DataFrame(data=fake[demarcator:]),
                                                                             pd.DataFrame(data=correct[demarcator:]),
                                                                             True)
            custom_train_df, custom_validation_df = shuffle_and_split(
                pd.concat([spz_dataset_fake, ijece_dataset_fake]).to_dict('records'),
                pd.concat([spz_dataset_correct, ijece_dataset_correct]).to_dict('records'))
        if not combine:
            # Default mode
            if mode == "dt":
                # Get new Decision Tree
                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(default_train_df.iloc[:, :-1], default_train_df.iloc[:, -1])
            elif mode == "lr":
                # Get new Logistic Regressor
                clf = LogisticRegression(random_state=0, max_iter=MAX_ITER)
                clf = clf.fit(default_train_df.iloc[:, :-1], default_train_df.iloc[:, -1])
            elif mode == "nb":
                '''
                Here we try using NBSVM (Naive Bayes - Support Vector Machine) but using sklearn's logistic regression rather than SVM,
                although in practice the two are nearly identical. NBSVM was introduced by Sida Wang and Chris Manning in the paper
                [Baselines and Bigrams: Simple, Good Sentiment and Topic Classiﬁcation](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf).
                '''
                clf, r = naive_bayes_support_vector_machine(default_train_df.iloc[:, :-1], default_train_df.iloc[:, -1])
            elif mode == "rf":
                clf = RandomForestClassifier(max_depth=2, random_state=0)
                clf = clf.fit(default_train_df.iloc[:, :-1], default_train_df.iloc[:, -1])
            elif mode == "dl":
                print(f"Training default model {i + 1}/{n_iter}      ", end="\r")
                # Get new DL
                if csv:
                    # Go with IJCE
                    clf = run_ijce_default(default_train_df)
                else:
                    clf = run_spz_default(default_train_df)

            # Get ground truth and predictions to measure performance
            X_val, y_val = default_validation_df.iloc[:, :-1], default_validation_df.iloc[:, -1]
            if mode == "dl":
                # TODO: Fix the precision counting
                '''
                precision = 0
                accuracy = 0
                for x in range(10):
                    loss, acc = clf.evaluate(x=X_val, y=y_val, verbose=0)
                    accuracy += acc
                avg_scores['default']['precision'] += accuracy / 10
                avg_scores['default']['accuracy'] += accuracy / 10
                '''
                accuracy = 0
                precision = 0
                recall = 0
                n = 100
                for _ in range(n):
                    _, acc, prc, rec = clf.evaluate(x=X_val, y=y_val, verbose=0)
                    accuracy += acc
                    precision += prc
                    recall += rec
                avg_scores['default']['accuracy'] += accuracy / n
                avg_scores['default']['precision'] += precision / n
                avg_scores['default']['recall'] += recall / n
                avg_scores['default']['f1'] += f1_score(precision / n, recall / n)
            else:
                if mode != "nb":
                    y_pred = clf.predict(X_val)
                else:
                    y_pred = clf.predict(X_val.multiply(r))

                # Default scores
                scores = get_scores(y_val, y_pred)
                avg_scores['default']['accuracy'] += scores['accuracy']
                avg_scores['default']['precision'] += scores['precision']
                avg_scores['default']['recall'] += scores['recall']
                avg_scores['default']['f1'] += scores['f1']

        # Custom mode
        if mode == "dt":
            # Get new Decision Tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(custom_train_df.iloc[:, :-1], custom_train_df.iloc[:, -1])
        elif mode == "lr":
            # Get new Logistic Regressor
            clf = LogisticRegression(random_state=0, max_iter=2500)
            clf = clf.fit(custom_train_df.iloc[:, :-1], custom_train_df.iloc[:, -1])
        elif mode == "nb":
            # Get new Naive Bayes (Logistic Regression)
            clf, r = naive_bayes_support_vector_machine(custom_train_df.iloc[:, :-1], custom_train_df.iloc[:, -1])
        elif mode == "rf":
            clf = RandomForestClassifier(max_depth=2, random_state=0)
            clf = clf.fit(custom_train_df.iloc[:, :-1], custom_train_df.iloc[:, -1])
        elif mode == "dl":
            print(f"Training custom model {i + 1}/{n_iter}      ", end="\r")
            # Get new DL
            if csv:
                # Go with IJCE
                clf = run_ijce_custom(custom_train_df)
            else:
                clf = run_spz_custom(custom_train_df)

        # Get ground truth and predictions to measure performance
        X_val, y_val = custom_validation_df.iloc[:, :-1], custom_validation_df.iloc[:, -1]
        if mode == "dl":
            # TODO Fix the precision counting!
            '''
            precision = 0
            accuracy = 0
            for x in range(10):
                loss, acc = clf.evaluate(x=X_val, y=y_val, verbose=0)
                accuracy += acc
            avg_scores['custom']['precision'] += accuracy / 10
            avg_scores['custom']['accuracy'] += accuracy / 10
            '''
            accuracy = 0
            precision = 0
            recall = 0
            n = 100
            for _ in range(n):
                _, acc, prc, rec = clf.evaluate(x=X_val, y=y_val, verbose=0)
                accuracy += acc
                precision += prc
                recall += rec
            avg_scores['custom']['accuracy'] += accuracy / n
            avg_scores['custom']['precision'] += precision / n
            avg_scores['custom']['recall'] += recall / n
            avg_scores['custom']['f1'] += f1_score(precision / n, recall / n)
        else:
            if mode != "nb":
                y_pred = clf.predict(X_val)
            else:
                y_pred = clf.predict(X_val.multiply(r))
            # Custom scores
            scores = get_scores(y_val, y_pred)
            avg_scores['custom']['accuracy'] += scores['accuracy']
            avg_scores['custom']['precision'] += scores['precision']
            avg_scores['custom']['recall'] += scores['recall']
            avg_scores['custom']['f1'] += scores['f1']

        print(f"{i + 1}/{n_iter}                            ", end="\r")

    # Averaging
    for t in avg_scores.keys():
        for s in avg_scores[t].keys():
            avg_scores[t][s] /= n_iter

    print('Done!')

    print("Accuracy - Default {:.3f}; Custom {:.3f}".format(avg_scores['default']['accuracy'],
                                                            avg_scores['custom']['accuracy']))
    print("Precision - Default {:.3f}; Custom {:.3f}".format(avg_scores['default']['precision'],
                                                             avg_scores['custom']['precision']))
    print("Recall - Default {:.3f}; Custom {:.3f}".format(avg_scores['default']['recall'],
                                                          avg_scores['custom']['recall']))
    print("F1 - Default {:.3f}; Custom {:.3f}".format(avg_scores['default']['f1'],
                                                      avg_scores['custom']['f1']))
    print("=============================")
    return avg_scores
