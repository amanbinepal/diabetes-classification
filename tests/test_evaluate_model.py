## test_evaluate_model.py
## author: Emily Jin
## 2025-12-11

# This script test the evaluate_model function used in the 18_model_testing.py script

# Usage:

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.functions.read_data import read_data
from src.functions.evaluate_model import evaluate_model
from src.functions.read_model import read_model


def test_evaluate_model():
    '''
    Docstring for test_evaluate_model
    This function ensures that the accuracy score of our two model are between 0 and 1.
    '''
    X_test, y_test = read_data("src/objects/X_test.csv", "src/objects/y_test.csv")
    lr_pipe, linear_svc_pipe = read_model("results/models")

    accuracy_lr = evaluate_model(lr_pipe, X_test, y_test)
    accuracy_svc = evaluate_model(linear_svc_pipe, X_test, y_test)

    assert 0.0 <= accuracy_lr <= 1.0, f"The lr accuracy score ({accuracy_lr}) is not between 0 and 1"
    assert 0.0 <= accuracy_svc <= 1.0, f"The svc accuracy score ({accuracy_svc}) is not between 0 and 1"
