
import numpy as np

from utils import perturb, infer
from uncertainty import compute_nc_uncertainty


np.random.seed(1234)
SAMPLE_SIZE = 5000
DEFAULT_STDDEVS = [
    1e-5 * (10 ** (idx / 4)) for idx in range(17)
]
EPS = 1e-5

def tuned_perturb(model, num_perturb, method, trainset, device, stddevs=None, loss_fn=None, pert_inputs=None, cache=None, overwrite=True, k=100, metric="cosine", generator=None):
    best_unc_stddev, best_perts = 0.0, None
    pert_stddev_to_unc_stddev = {}
    if stddevs is None:
        stddevs = DEFAULT_STDDEVS

    # Investigate search space of perturbation standard deviations
    for stddev in stddevs:
        # Generate synthetic ensemble
        models = perturb(
            model,
            stddev,
            num_perturb,
            method,
            device,
            loss_fn=loss_fn,
            inputs=pert_inputs,
            cache=cache,
            overwrite=overwrite,
            generator=None,
        )
        # Evaluate uncertainty
        embeddings = [infer(model, trainset)] + [infer(m, trainset) for m in models]
        got_inf = False
        for emb in embeddings[1:]:
            if emb.isinf().any():
                got_inf = True
                break
        if got_inf:
            print(f"{stddev} is too large")
            continue
        ref_idx = np.random.choice(
            embeddings[0].shape[0], size=SAMPLE_SIZE, replace=False
        )
        uncertainties = compute_nc_uncertainty(
            [emb[ref_idx] for emb in embeddings], embeddings, k=k, metric=metric
        )
        # Take the first highest spread in uncertainty
        unc_stddev = uncertainties.std().item()
        pert_stddev_to_unc_stddev[stddev] = unc_stddev
        if unc_stddev > best_unc_stddev:
            best_unc_stddev = unc_stddev
            best_perts = models
    
    return best_perts, pert_stddev_to_unc_stddev
    
