from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import seaborn as sns
import matplotlib.pyplot as plt


def metrics(X, Y, to_return=True):  # X predicted, Y real
    scores = {"Accuracy": accuracy_score(Y, X),
              "Precision": precision_score(Y, X),
              "Recall": recall_score(Y, X),
              "F1": f1_score(Y, X)}
    print("Accuracy:  ", scores["Accuracy"])
    print("Precision: ", scores["Precision"])
    print("Recall:    ", scores["Recall"])
    print("F1:        ", scores["F1"])
    ax = sns.heatmap(confusion_matrix(Y, X), annot=True, vmax=X.size, vmin=-X.size,
                     fmt="d", cmap="crest")
    ax.set(xlabel="Predicted class", ylabel="Real class")
    plt.show()

    if to_return:
      return scores
