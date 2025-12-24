
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

def compute_knn_idxs(X, Y, k=100, metric='cosine'):
    distances = compute_distance(X, Y, metric=metric).squeeze(0)
    _, knn_idxs = torch.topk(distances, k=k, largest=False, dim=1)
    return [set(idxs.tolist()) for idxs in knn_idxs]

def compute_fv_uncertainty(Ys, metric='cosine'):
    if metric == "cosine":
        Ys = [Y / Y.norm(dim=1, keepdim=True) for Y in Ys]
    return torch.var(torch.stack(Ys), dim=0, correction=0).sum(dim=-1)

def compute_ll_uncertainty(X, Y, n_clusters=20, max_iters=30, metric='cosine'):
    kernel = (
        VonMisesFisherMixture(
            n_clusters=n_clusters,
            posterior_type='soft',
            max_iter=max_iters,
            normalize=False
        ) if metric == "cosine" else
        GaussianMixture(n_components=n_clusters, max_iter=max_iters)
    )
    kernel.fit(X.cpu().numpy())
    return (
        kernel.predict_log_proba(Y.cpu().numpy())
        if metric == "cosine" else
        kernel.score_samples(Y.cpu().numpy())
    )

def compute_nc_uncertainty(Xs, Ys, k=100, metric='cosine'):
    N = len(Xs)
    M = Ys[0].shape[0]
    knn_idxs = [compute_knn_idxs(X, Y, k=k, metric=metric) for X, Y in tqdm(zip(Xs, Ys))]
    return torch.tensor([sum([
        (
            len(knn_idxs[kdx][jdx].intersection(knn_idxs[idx][jdx])) /
            len(knn_idxs[kdx][jdx].union(knn_idxs[idx][jdx]))
        ) # Jaccard index
        for idx in range(N)
        for kdx in range(idx+1, N)
    ]) / (N * (N - 1) / 2) for jdx in range(M)])

if __name__ == "__main__":
    pass
    