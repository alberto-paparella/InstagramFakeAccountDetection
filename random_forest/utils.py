from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def fit_random_forest(X, y, validation_df):
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    rfc.fit(X, y)

    X_val, y_val = validation_df.iloc[:, :-2], validation_df.iloc[:, -1]
    y_pred = rfc.predict(X_val)
    return y_val, y_pred


def get_single_scores(train_df, validation_df):
    scores = {
        'precision': 0, 'accuracy': 0
    }
    y_val, y_pred = fit_random_forest(train_df.iloc[:, :-2], train_df.iloc[:, -1], validation_df)

    scores['accuracy'] += metrics.accuracy_score(y_val, y_pred)
    scores['precision'] += metrics.precision_score(y_val, y_pred)
    return scores
