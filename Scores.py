from sklearn.metrics import log_loss, confusion_matrix
from numpy import nan, log

def sensitivity(clf, X, y) -> float:
    # [trueNegative,falsePositive, falseNegative, truePositive]
    y_pred = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # Used to remove dividing by 0 warnings
    if (tp + fn) == 0:
        return nan

    return tp / (tp + fn)


def specificity(clf, X, y) -> float:
    # [trueNegative,falsePositive, falseNegative, truePositive]
    y_pred = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # Used to remove dividing by 0 warnings
    if (tn + fp) == 0:
        return nan

    return tn / (tn + fp)


def PPV(clf, X, y) -> float:
    # [trueNegative,falsePositive, falseNegative, truePositive]
    y_pred = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # Used to remove dividing by 0 warnings
    if (tp + fp) == 0:
        return nan

    return tp / (tp + fp)


def NPV(clf, X, y) -> float:
    # [trueNegative,falsePositive, falseNegative, truePositive]
    y_pred = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # Used to remove dividing by 0 warnings
    if (tn + fn) == 0:
        return nan

    return tn / (tn + fn)


def calculate_bic2(clf, X, y, num_params=2) -> float:
    # Sources:
    # https://stackoverflow.com/questions/48185090/how-to-get-the-log-likelihood-for-a-logistic-regression-model-in-sklearn
    # https://en.wikipedia.org/wiki/Bayesian_information_criterion

    n = len(y)
    y_pred = clf.predict(X)
    log_likelihood = -log_loss(y, y_pred)

    bic = -2 * log_likelihood + num_params * log(n)
    return bic


scores = {
    "BIC": calculate_bic2,
    "sensitivity": sensitivity,  # (also called the true positive rate/recall),
    "specificity": specificity,  # (also called the true negative rate),
    "NPV": NPV,  # Negative predictive value
    "PPV": PPV,  # Precision or positive predictive value
    "balanced_accuracy": "balanced_accuracy",
    'f1': 'f1',
}
