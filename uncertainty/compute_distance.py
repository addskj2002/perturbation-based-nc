
import torch

def compute_cosine_distance(X, Y):
    X_normalized = X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    Y_normalized = Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-8)
    dist = 1 - Y_normalized @ X_normalized.T
    return dist.clamp(min=0)

def compute_euclidean_distance(X, Y):
    return torch.cdist(Y, X, p=2)

def compute_distance(X, Y, metric):
    if metric == 'cosine':
        return compute_cosine_distance(X, Y)
    elif metric == 'euclidean':
        return compute_euclidean_distance(X, Y)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
        