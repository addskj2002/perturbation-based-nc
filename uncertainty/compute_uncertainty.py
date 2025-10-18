
from tqdm import tqdm

import torch

from uncertainty.compute_distance import compute_distance

def compute_norm_uncertainty(Y):
    return torch.norm(Y, dim=1)

def compute_dist_uncertainty(X, Y, k=100, metric='cosine'):
    distances = compute_distance(X, Y, metric=metric).squeeze(0)
    knn_dist, _ = torch.topk(distances, k=k, largest=False, dim=1)
    return knn_dist.mean(dim=1)

def compute_knn_idxs(X, Y, k=100, metric='cosine'):
    distances = compute_distance(X.to('cpu'), Y.to('cpu'), metric=metric).squeeze(0)
    _, knn_idxs = torch.topk(distances, k=k, largest=False, dim=1)
    return [set(idxs.tolist()) for idxs in knn_idxs]

def compute_nc_uncertainty(Xs, Ys, k=100, metric='cosine'):
    N = len(Xs)
    M = Ys[0].shape[0]
    knn_idxs = [compute_knn_idxs(X, Y, k=k, metric=metric) for X, Y in tqdm(zip(Xs, Ys))]
    return torch.tensor([sum([
        (
            len(knn_idxs[0][jdx].intersection(knn_idxs[idx][jdx])) /
            len(knn_idxs[0][jdx].union(knn_idxs[idx][jdx]))
        ) # Jaccard index
        for idx in range(1, N)
    ]) / (N - 1) for jdx in range(M)])

if __name__ == "__main__":
    pass
    