import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_testing import evaluate_model, read_model, read_data

def test_evaluate_model():
    X_test, y_test = read_data("src/objects/X_test.csv", "src/objects/y_test.csv")
    lr_pipe, linear_svc_pipe = read_model("results/models")

    accuracy_lr = evaluate_model(lr_pipe, X_test, y_test)
    accuracy_svc = evaluate_model(linear_svc_pipe, X_test, y_test)

    assert 0.0 <= accuracy_lr <= 1.0, f"The lr accuracy score ({accuracy_lr}) is not between 0 and 1"
    assert 0.0 <= accuracy_svc <= 1.0, f"The svc accuracy score ({accuracy_svc}) is not between 0 and 1"