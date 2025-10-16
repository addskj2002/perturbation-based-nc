
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
        