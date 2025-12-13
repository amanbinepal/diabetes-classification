from sklearn.metrics import accuracy_score


def evaluate_model(pipe, X_test, y_test):
    prediction = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    return accuracy
