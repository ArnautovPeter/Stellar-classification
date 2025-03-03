import pandas as pd


class Data:
  @staticmethod
  def get_train():
    train = pd.read_csv("../data/train.csv", index_col=0)
    return train.iloc[:, :5], train.iloc[:, 5].to_numpy()
  
  @staticmethod
  def get_test():
    test = pd.read_csv("../data/test.csv", index_col=0)
    return test.iloc[:, :5], test.iloc[:, 5].to_numpy()