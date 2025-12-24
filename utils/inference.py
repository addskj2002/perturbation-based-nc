
import torch

BATCH = 256

def infer(model, dataset):
    N = len(dataset)
    idx = 0
    embeddings = []
    with torch.no_grad():
        while idx < N:
            embeddings.append(model(dataset[idx:idx+BATCH]))
            idx += BATCH
        return torch.cat(embeddings)

def is_stable(model, dataset):
    N = len(dataset)
    idx = 0
    with torch.no_grad():
        while idx < N:
            single_emb = model(dataset[idx:idx+BATCH])
            if single_emb.isnan().any() or single_emb.isinf().any():
                return False
            idx += BATCH
    return True