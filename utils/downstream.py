
import pickle
import argparse

from sklearn.linear_model import LogisticRegression
import torch

EPS = 1e-10

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_logistic_regression(model, X_test, y_test):
    probability = torch.tensor(model.predict_proba(X_test))
    pred_entropy = -(probability * torch.log(probability + EPS)).sum(dim=1)
    cross_entropy = -torch.log(probability[[idx for idx in range(len(y_test))], y_test] + EPS)
    brier = probability.clone()
    brier[range(len(y_test)), y_test] -= 1
    brier = (brier ** 2).sum(dim=1)
    return {
        'probability': probability,
        'pred_entropy': pred_entropy,
        'cross_entropy': cross_entropy,
        'brier': brier,
    }

if __name__ == "__main__":
    pass
