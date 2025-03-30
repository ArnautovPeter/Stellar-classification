from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score


def metrics(X, Y, to_return=True):  # X predicted, Y real
    scores = {"Accuracy": accuracy_score(Y, X),
              "Precision": precision_score(Y, X),
              "Recall": recall_score(Y, X),
              "F1": f1_score(Y, X)}
    print("Accuracy:  ", scores["Accuracy"])
    print("Precision: ", scores["Precision"])
    print("Recall:    ", scores["Recall"])
    print("F1:        ", scores["F1"])
    print(confusion_matrix(Y, X))

    if to_return:
      return scores
