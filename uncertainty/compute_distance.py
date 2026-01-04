
import torch

def compute_cosine_distance(X, Y):
    X_norm = X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    Y_norm = Y.norm(dim=1, keepdim=True).clamp(min=1e-8)
    X_norm[X_norm.isinf()] = torch.nan
    Y_norm[Y_norm.isinf()] = torch.nan
    X_normalized = X / X_norm
    Y_normalized = Y / Y_norm
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
        