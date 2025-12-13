from sklearn.metrics import accuracy_score


def evaluate_model(pipe, X_test, y_test):
    """
    This function uses a trained model to make predictions from X_test features and calculates its accuracy compared to the y_test targets.

    Parameters
    ----------
    pipe = the trained model
    X_test = the unseen features from the test dataset
    y_yest = the unseen target from the test dataset

    Returns:
        An accuracy score for a given model

    Examples
    --------
    >>> accuracy_lr = evaluate_model(lr_pipe, X_test, y_test)
    """
    prediction = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    return accuracy
