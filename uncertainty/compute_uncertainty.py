
from tqdm import tqdm

import torch
from sklearn.mixture import GaussianMixture

from uncertainty.vonMF.von_mises_fisher_mixture import VonMisesFisherMixture
from uncertainty.compute_distance import compute_distance

def compute_norm_uncertainty(Y):
    return torch.norm(Y, dim=1)

def compute_dist_uncertainty(X, Y, k=100, metric='cosine'):
    distances = compute_distance(X, Y, metric=metric).squeeze(0)
    knn_dist, _ = torch.topk(distances, k=k, largest=False, dim=1)
    return knn_dist.mean(dim=1)

def compute_fv_uncertainty(Ys, metric='cosine'):
    if metric == "cosine":
        Ys = [Y / Y.norm(dim=1, keepdim=True) for Y in Ys]
    return torch.var(torch.stack(Ys), dim=0, correction=0).sum(dim=-1)

def compute_nc_uncertainty(Xs, Ys, k=100, metric='cosine'):
    N = len(Xs)
    M = Ys[0].shape[0]
    knn_idxs = []
    valid = torch.tensor([True for _ in range(M)])
    for X, Y in tqdm(zip(Xs, Ys)):
        distances = compute_distance(X, Y, metric=metric).squeeze(0)
        valid = valid & ~distances.isnan().any(dim=1).to("cpu")
        _, single_knn_idxs = torch.topk(distances, k=k, largest=False, dim=1)
        knn_idxs.append([set(idxs.tolist()) for idxs in single_knn_idxs])
    ret = torch.tensor([sum([
        (
            len(knn_idxs[kdx][jdx].intersection(knn_idxs[idx][jdx])) /
            len(knn_idxs[kdx][jdx].union(knn_idxs[idx][jdx]))
        ) # Jaccard index
        for idx in range(N)
        for kdx in range(idx+1, N)
    ]) / (N * (N - 1) / 2) for jdx in range(M)])
    ret[~valid] = torch.nan
    return ret

if __name__ == "__main__":
    pass
    