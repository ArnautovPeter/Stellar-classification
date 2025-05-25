from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import seaborn as sns
import matplotlib.pyplot as plt


def metrics(pred, real, to_return=True, output=True):  # X predicted, Y real
    X = pred
    Y = real
    scores = {"Accuracy": accuracy_score(Y, X),
              "Precision": precision_score(Y, X, zero_division=0),
              "Recall": recall_score(Y, X, zero_division=0),
              "F1": f1_score(Y, X, zero_division=0)}
    if output:
        print("Accuracy:  ", scores["Accuracy"])
        print("Precision: ", scores["Precision"])
        print("Recall:    ", scores["Recall"])
        print("F1:        ", scores["F1"])
        ax = sns.heatmap(confusion_matrix(Y, X), annot=True, vmax=X.size, vmin=0,
                         fmt="d", cmap="Blues")
        ax.set(xlabel="Predicted class", ylabel="Real class")
        plt.show()

    if to_return:
        return scores
