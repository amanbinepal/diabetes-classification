import pandas as pd

def read_data(x_test, y_test):
    """
    This function reads in x_test and y_test from a designation directory to be run in our trained model

    Parameters
    ----------
    x_test = unseen features
    y_test = unseen target

    Returns:
        X_test and y_test variables for testing

    Examples
    --------
    >>> X_test, y_test = read_data(x_test, y_test)
    """
    x = pd.read_csv(x_test)
    y = pd.read_csv(y_test)
    return x, y
