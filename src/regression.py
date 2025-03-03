import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score,\
  confusion_matrix, roc_curve, precision_recall_curve


class BinaryRegression(LinearRegression):
    def __init__(self):
        super().__init__()

    def draw_dist(self, X) -> None:
        reg_pred = self.predict(X)

        l, r = np.min(reg_pred), np.max(reg_pred)

        plt.style.use('seaborn-v0_8-pastel')
        plt.hist(reg_pred, bins=50, range=(l, r), density=True,
                 edgecolor='black', alpha=0.5, label="Predicted values")

    def ROC_threshold(self, X, Y) -> int:
        reg_pred = self.predict(X)
        fpr, tpr, thresholds = roc_curve(Y, reg_pred)
        return thresholds[np.argmax(tpr - fpr)]

    def F1_threshold(self, X, Y) -> int:
        reg_pred = self.predict(X)
        precision, recall, thresholds = precision_recall_curve(Y, reg_pred)
        f1_scores = 2 * (precision * recall) / \
            (precision + recall)  # optimizing f1
        return thresholds[np.argmax(f1_scores)]
    
    def cost_threshold(self, X, Y) -> int:
        reg_pred = self.predict(X)
        l, r = np.min(reg_pred), np.max(reg_pred)

        thresholds = np.linspace(l, r, 2500)
        threshold = 0
        best_cost = 0

        cost_tp = 5

        for t in thresholds:
            cm = confusion_matrix((reg_pred >= t).astype(int), Y)
            tp, fp = cm[1, 1], cm[1, 0]
            cur_cost = tp * cost_tp + fp
            if cur_cost > best_cost:
              best_cost = cur_cost
              threshold = t

        return threshold


    # maybe it's better to has t as class field
    def metric(self, X, Y, t):
        pred_bin = (self.predict(X) >= t).astype(int)
        print("Accuracy: ", accuracy_score(Y, pred_bin))
        print("Recall:   ", recall_score(Y, pred_bin))
        print("F1:       ", f1_score(Y, pred_bin))

        # confusion has indexes like (real, predicted)
        print(confusion_matrix(Y, pred_bin))
