import click
import pandas as pd
import os
import pickle

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

from functions.read_data import read_data

# from great_tables import GT

def read_model(out_dir):
    lr_pipe_path = os.path.join(out_dir, "trained_lr_pipe.pkl")
    linear_svc_pipe_path = os.path.join(out_dir, "trained_linear_svc_pipe.pkl")

    with open(lr_pipe_path, "rb") as f:
        lr_pipe = pickle.load(f)

    with open(linear_svc_pipe_path, "rb") as f:
        linear_svc_pipe = pickle.load(f)

    return lr_pipe, linear_svc_pipe

def evaluate_model(pipe, X_test, y_test):
    prediction = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    return accuracy


@click.command()
@click.option(
    "--x-test",
    default="src/objects/X_test.csv",
    help="Feature test dataset CSV file",
)
@click.option(
    "--y-test",
    default="src/objects/y_test.csv",
    help="Response test dataset CSV file",
)
def model_testing(x_test, y_test, out_dir="results/models"):
    """This function tests our models.
        1. Imports trained models
        2. Make predictions on X_test data
        3. Creates accuracy score table to "model_testing.png"

    Args:
        x_test (string): the path to the x_test data set Defaults to "src/objects/X_test.csv".
        y_test (string): the path to the y_test data set Defaults to "src/objects/y_test.csv".
        out_dir (str, optional): the folder to save the plots. Defaults to "results/models".
    """

    # ============================ read in test data
    #X_test = pd.read_csv(x_test)
    #y_test = pd.read_csv(y_test)
    X_test, y_test = read_data(x_test, y_test)

    # ============================ read in models
    #with open(os.path.join(out_dir, "trained_lr_pipe.pkl"), "rb") as f:
    #    lr_pipe = pickle.load(f)

    #with open(os.path.join(out_dir, "trained_linear_svc_pipe.pkl"), "rb") as f:
    #    linear_svc_pipe = pickle.load(f)

    lr_pipe, linear_svc_pipe = read_model(out_dir)

    # ============================ Test Models
    #prediction_lr = lr_pipe.predict(X_test)
    #accuracy_lr = accuracy_score(y_test, prediction_lr)

    #prediction_svc = linear_svc_pipe.predict(X_test)
    #accuracy_svc = accuracy_score(y_test, prediction_svc)

    accuracy_lr = evaluate_model(lr_pipe, X_test, y_test)
    accuracy_svc = evaluate_model(linear_svc_pipe, X_test, y_test)


    # ============================ Make table
    test_results = {"Logistic Regression": accuracy_lr, "Linear SVC": accuracy_svc}

    test_results = pd.DataFrame([test_results], index=['Accuracy'])
    # Also output as csv for score readin from quarto 
    test_results.to_csv(os.path.join(out_dir, "model_testing.csv"), index=False)

    # ============================ Export Great Table
    # great_table2 = (
    #     GT(test_results.rename(columns={"index": "Metric"}))
    #     .tab_header(
    #         title="Accuracy Score on Test Set for Candidate Models"
    #     )
    #     .fmt_number(columns=["Logistic Regression", "Linear SVC"], decimals=4)
    #     .opt_stylize(style=3, color="blue")
    # )

    # great_table2.save(os.path.join(out_dir, "model_testing.html"), scale=2)

    # ============================ Export Models

    click.echo("Models tested")

if __name__ == "__main__":
    model_testing()
