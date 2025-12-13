import pandas as pd

def read_data(x_test, y_test):
    x = pd.read_csv(x_test)
    y = pd.read_csv(y_test)
    return x, y