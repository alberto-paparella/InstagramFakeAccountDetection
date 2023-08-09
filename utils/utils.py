from dataset.utils import get_custom_dataset, shuffle_and_split, get_default_dataset, get_compatible_dataset, treat_combined
from sklearn import svm, tree, metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from deep.IJECE.IJECE_custom import run_model as run_ijce_custom
from deep.IJECE.IJECE_default import run_model as run_ijce_default
from deep.InstaFake.instafake_default import run_model as run_if_default
from deep.InstaFake.instafake_custom import run_model as run_if_custom
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
   

def f1_score(precision, recall):
    '''
    Function that calculates the F1-score given precision and recall.
    '''
    return 2 * (precision * recall) / (precision + recall)


def get_scores(y_test, y_pred):
    '''
    Function that given the ground truth and the predicted values provides the
    metric scores (accuracy, precision, recall, f1-score).
    '''
    scores = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    scores['accuracy'] += metrics.accuracy_score(y_test, y_pred)
    scores['precision'] += metrics.precision_score(y_test, y_pred)
    scores['recall'] += metrics.recall_score(y_test, y_pred)
    scores['f1'] += metrics.f1_score(y_test, y_pred)
    return scores


def get_classifier(model="dt", X=[], y=[]):
    '''
    Function that creates and fits a classifier of type `model` based
    on inputs `X` and outputs `y` and returns the fitted model.

    `Available models`:
    - `dt`  => Decision Tree
    - `rf`  => Random Forest 
    - `svm` => Support Vector Machine
    - `nbb` => Naive Bayes (Bernoulli dist.)
    - `nbg` => Naive Bayes (Gaussian dist.)
    - `lr`  => Logistic Regression
    '''
    if model == "dt":
        # Get new Decision Tree
        clf = tree.DecisionTreeClassifier()
    elif model == "rf":
        # Get new Random Forest
        clf = RandomForestClassifier(max_depth=2, random_state=0)
    if model == "svm":
        # Get new Support Vector Machine
        clf = svm.SVC()
    elif model == "nbb":
        # Get new (Bernoulli dist.) Naive Bayes
        clf = BernoulliNB(force_alpha=True)
    elif model == "nbg":
        # Get new (Gaussian dist.) Naive Bayes
        clf = GaussianNB()
    elif model == "lr":
        # Get new Logistic Regression
        clf = LogisticRegression(random_state=0, max_iter=5000)
    return clf.fit(X, y)


@ignore_warnings(category=ConvergenceWarning)
def experiment(fake, correct, csv, model="dt", n_iter=20, combine=False, demarcator=700, compatibility=False):
    '''
    A function which executes an experiment fitting a specified model `n_iter` times and giving
    back the `avg_scores` for various metrics such as `accuracy`, `precision`, `recall`, `f1-score`.

    `Available models`:
    - `dt`  => Decision Tree
    - `rf`  => Random Forest 
    - `svm` => Support Vector Machine
    - `nbb` => Naive Bayes (Bernoulli dist.)
    - `nbg` => Naive Bayes (Gaussian dist.)
    - `lr`  => Logistic Regression
    '''
    avg_scores = {
        'default': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
        'custom': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    }

    if model == "dt":
        print(f"Calculating metrics for Decision Trees over {n_iter} times")
    elif model == "rf":
        print(f"Calculating metrics for Random Forests over {n_iter} times")
    elif model == "svm":
        print(f"Calculating metrics for Support Vector Machine over {n_iter} times")
    elif model == "nbb":
        print(f"Calculating metrics for Naive Bayes (Bernoulli dist.) over {n_iter} times")
    elif model == "nbg":
        print(f"Calculating metrics for Naive Bayes (Gaussian dist.) over {n_iter} times")
    elif model == "lr":
        print(f"Calculating metrics for Logistic Regression over {n_iter} times")
    else:
        print(f"The specified model is not supported at the moment.")
        return -1

    for i in range(n_iter):
        # Get new train_df and test_df, same for default and custom
        if not combine:
            train_df, test_df = shuffle_and_split(fake, correct)
            if compatibility:
                custom_train_df, custom_test_df = get_compatible_dataset(train_df, test_df, csv)
            else:
                custom_train_df, custom_test_df = get_custom_dataset(train_df, test_df, csv)
                default_train_df, default_test_df = get_default_dataset(train_df, test_df, csv)
        else:
            custom_train_df, custom_test_df = treat_combined(fake, correct, demarcator)

        # When running on combine or compatibility mode, default mode would not make any sense
        if not combine and not compatibility:
            # Default mode
            clf = get_classifier(model, default_train_df.iloc[:, :-1], default_train_df.iloc[:, -1])
            # Get ground truth and predictions to measure performance
            X_test, y_test = default_test_df.iloc[:, :-1], default_test_df.iloc[:, -1]
            y_pred = clf.predict(X_test)
            # Default scores
            scores = get_scores(y_test, y_pred)
            avg_scores['default']['accuracy'] += scores['accuracy']
            avg_scores['default']['precision'] += scores['precision']
            avg_scores['default']['recall'] += scores['recall']
            avg_scores['default']['f1'] += scores['f1']

        # Custom mode
        clf = get_classifier(model, custom_train_df.iloc[:, :-1], custom_train_df.iloc[:, -1])
        # Get ground truth and predictions to measure performance
        X_test, y_test = custom_test_df.iloc[:, :-1], custom_test_df.iloc[:, -1]
        y_pred = clf.predict(X_test)
        # Custom scores
        scores = get_scores(y_test, y_pred)
        avg_scores['custom']['accuracy'] += scores['accuracy']
        avg_scores['custom']['precision'] += scores['precision']
        avg_scores['custom']['recall'] += scores['recall']
        avg_scores['custom']['f1'] += scores['f1']

    # Averaging
    for t in avg_scores.keys():
        for s in avg_scores[t].keys():
            avg_scores[t][s] /= n_iter

    print('Done!')
    print("Accuracy  - Default {:.3f}; Custom {:.3f}".format(avg_scores['default']['accuracy'],
                                                             avg_scores['custom']['accuracy']))
    print("Precision - Default {:.3f}; Custom {:.3f}".format(avg_scores['default']['precision'],
                                                             avg_scores['custom']['precision']))
    print("Recall    - Default {:.3f}; Custom {:.3f}".format(avg_scores['default']['recall'],
                                                             avg_scores['custom']['recall']))
    print("F1-score  - Default {:.3f}; Custom {:.3f}".format(avg_scores['default']['f1'],
                                                             avg_scores['custom']['f1']))
    print("=============================")

    return avg_scores
