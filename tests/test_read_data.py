## test_evaluate_model.py
## author: Aman Binepal
## 2025-12-11

# This script tests the read_data function used in the \ 
# 18_model_testing.py script.

# Usage: pytest from command line root

import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.functions.read_data import read_data

def test_read_data():
    '''
    Docstring for test_read_data
    This function confirms that our X_test and y_test data are data frames and are not empty.
    '''
    x_test_data = "data/processed/X_test.csv"
    y_test_data = "data/processed/y_test.csv"

    X_test, y_test = read_data(x_test_data, y_test_data)

    assert isinstance(X_test, pd.DataFrame), "X_test is not a DataFrame"
    assert isinstance(y_test, pd.DataFrame), "y_test is not a DataFrame"
    assert not X_test.empty, "X_test is empty"
    assert not y_test.empty, "y_test is empty"
