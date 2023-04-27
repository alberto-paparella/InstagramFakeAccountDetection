from dataset.utils import get_custom_dataset, shuffle_and_split
from sklearn import tree, metrics
from sklearn.linear_model import LogisticRegression

import numpy as np

PERCENT_TRAIN = 70
MAX_ITER = 5000 # Maximum number of iterations for Logistic Regressors


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


def get_scores(y_val, y_pred):
    scores = {
        'precision': 0, 'accuracy': 0
    }
    scores['accuracy'] += metrics.accuracy_score(y_val, y_pred)
    scores['precision'] += metrics.precision_score(y_val, y_pred)
    return scores


def print_scores(train_df, validation_df):
    scores = get_scores(train_df, validation_df)
    print('precision:', "{:.3f}".format(scores['precision']))
    print('accuracy:', "{:.3f}".format(scores['accuracy']))

'''
basic naive bayes feature equation
'''
def pr(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def nbsvm(x, y):
    y = y.values
    r = np.log(pr(x,1,y) / pr(x,0,y))
    #m = LogisticRegression(C=4, dual=True) # This gives an error
    m = LogisticRegression(C=4, dual=False, max_iter=MAX_ITER)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

'''
modes:
 - "dt" => DecisionTree
 - "lr" => LogisticRegression
 - "nb" => NaiveBayes (NB-SVM, but using LogisticRegression instead)
'''
def experiment(fake, correct, csv, mode="dt", n_iter=20):
    avg_scores = {
        'default': {'precision': 0, 'accuracy': 0},
        'custom': {'precision': 0, 'accuracy': 0}
    }

    if mode == "dt":
        print(f"Calculating precision and accuracy metrics for Decision Trees over {n_iter} times")
    elif mode == "lr":
        print(f"Calculating precision and accuracy metrics for Logistic Regression over {n_iter} times")
    elif mode == "nb":
        print(f"Calculating precision and accuracy metrics for Naive Bayes (Logistic Regression) over {n_iter} times")
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
            clf = LogisticRegression(random_state=0, max_iter=MAX_ITER)
            clf = clf.fit(train_df.iloc[:, :-2], train_df.iloc[:, -1])
        elif mode == "nb":
            '''
            Here we try using NBSVM (Naive Bayes - Support Vector Machine) but using sklearn's logistic regression rather than SVM,
            although in practice the two are nearly identical. NBSVM was introduced by Sida Wang and Chris Manning in the paper
            [Baselines and Bigrams: Simple, Good Sentiment and Topic Classiﬁcation](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf).
            '''
            clf,r = nbsvm(train_df.iloc[:, :-2], train_df.iloc[:, -1])

        # Get ground truth and predictions to measure performance
        X_val, y_val = validation_df.iloc[:, :-2], validation_df.iloc[:, -1]
        if mode != "nb":
            y_pred = clf.predict(X_val)
        else:
            y_pred = clf.predict(X_val.multiply(r))

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
        elif mode == "nb":
            # Get new Naive Bayes (Logistic Regression)
            clf,r = nbsvm(custom_train_df.iloc[:, :-2], custom_train_df.iloc[:, -1])

        # Get ground truth and predictions to measure performance
        X_val, y_val = custom_validation_df.iloc[:, :-2], custom_validation_df.iloc[:, -1]
        if mode != "nb":
            y_pred = clf.predict(X_val)
        else:
            y_pred = clf.predict(X_val.multiply(r))

        # Custom scores
        scores = get_scores(y_val, y_pred)
        avg_scores['custom']['precision'] += scores['precision']
        avg_scores['custom']['accuracy'] += scores['accuracy']

        print(f"{i + 1}/{n_iter}", end="\r")

    # Averaging
    for t in avg_scores.keys():
        for s in avg_scores[t].keys():
            avg_scores[t][s] /= n_iter

    print('Done!\n\n')

    print('default avg precision:', "{:.3f}".format(avg_scores['default']['precision']))
    print('default avg accuracy:', "{:.3f}".format(avg_scores['default']['accuracy']))

    print('custom avg precision:', "{:.3f}".format(avg_scores['custom']['precision']))
    print('custom avg accuracy:', "{:.3f}".format(avg_scores['custom']['accuracy']))
