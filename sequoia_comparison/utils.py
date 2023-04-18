from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


def fit_decision_tree(X, y, validation_df):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y)
    # print("Fitting complete.")

    X_val, y_val = validation_df.iloc[:, :-2], validation_df.iloc[:, -1]
    y_pred = clf.predict(X_val)
    return y_val, y_pred


def get_single_scores(train_df, validation_df):
    scores = {
        'precision': 0, 'accuracy': 0
    }
    y_val, y_pred = fit_decision_tree(train_df.iloc[:, :-2], train_df.iloc[:, -1], validation_df)

    scores['accuracy'] += metrics.accuracy_score(y_val, y_pred)
    scores['precision'] += metrics.precision_score(y_val, y_pred)
    return scores


def print_classification_report(X, y, validation_df):
    y_val, y_pred = fit_decision_tree(X, y, validation_df)
    # y_compare = y_pred - y_val
    # print('accuracy =', 100 - (sum(abs(y_compare)) / len(validation_df.index)) * 100)
    print(metrics.classification_report(y_val, y_pred))
    print(type(metrics.classification_report(y_val, y_pred)))
