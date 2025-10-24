
import pickle
import argparse

from sklearn.linear_model import LogisticRegression
import torch


def train_multi_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=1234, max_iter=20)
    model.fit(X_train, y_train)
    return model

def train_binary_logistic_regression(X_train, y_train):
    # print("data:", X_train, y_train)
    torch.save(X_train, "X.pth")
    n_unique = len(torch.unique(y_train))
    models = {}
    for idx in range(n_unique):
        for jdx in range(idx+1, n_unique):
            model = LogisticRegression(random_state=1234, max_iter=20)
            model.fit(
                torch.cat([X_train[y_train == idx], X_train[y_train == jdx]]),
                [0] * sum(y_train == idx) + [1] * sum(y_train == jdx),
            )
            models[idx, jdx] = model
    return models

def evaluate_multi_logistic_regression(model, X_test, y_test):
    probability = torch.tensor(model.predict_proba(X_test))
    log2_prob = torch.log2(probability)
    log2_prob[probability == 0.0] = 0.0
    pred_entropy = -(probability * log2_prob).sum(dim=1)
    cross_entropy = -log2_prob[[idx for idx in range(len(y_test))], y_test]
    brier = probability.clone()
    brier[range(len(y_test)), y_test] -= 1
    brier = (brier ** 2).sum(dim=1)
    return {
        'probability': probability,
        'pred_entropy': pred_entropy,
        'cross_entropy': cross_entropy,
        'brier': brier,
    }

def evaluate_binary_logistic_regression(models, X_test, y_test):
    n_unique = len(torch.unique(y_test))
    n_samples = len(X_test)
    probabilities = torch.ones(n_samples, n_unique)

    # Calculate probability of being correct
    for idx in range(n_unique):
        for jdx in range(idx+1, n_unique):
            idx_proba = torch.tensor(
                models[idx, jdx].predict_proba(X_test[y_test == idx])[:, 0]
            ).to(torch.float32)
            jdx_proba = torch.tensor(
                models[idx, jdx].predict_proba(X_test[y_test == jdx])[:, 1]
            ).to(torch.float32)
            probabilities[y_test == idx, jdx] = idx_proba
            probabilities[y_test == jdx, idx] = jdx_proba

    # Evaluations
    brier = (2 * (1 - probabilities) ** 2).sum(dim=1) / (n_unique - 1)
    log2_prob = torch.log2(probabilities)
    log2_err = torch.log2(1 - probabilities)
    log2_prob[probabilities == 0.0] = 0.0
    log2_err[probabilities == 1.0] = 0.0
    cross_entropy = -log2_prob.sum(dim=1) / (n_unique - 1)
    pred_entropy = -(probabilities * log2_prob + (1 - probabilities) * log2_err).sum(dim=1) / (n_unique - 1)

    return {
        'probability': probabilities,
        'pred_entropy': pred_entropy,
        'cross_entropy': cross_entropy,
        'brier': brier,
    }

if __name__ == "__main__":
    pass
