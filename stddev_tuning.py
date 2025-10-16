
from utils import perturb, infer
from uncertainty import compute_nc_uncertainty

DEFAULT_STDDEVS = [
    1e-6 * (10 ** (idx / 4)) for idx in range(25)
]

def tuned_perturb(model, num_perturb, method, trainset, device, stddevs=None, loss_fn=None, pert_inputs=None, cache=None, overwrite=True, k=100, metric="cosine", generator=None):
    best_unc_stddev, best_perts = 0.0, None
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
        uncertainties = compute_nc_uncertainty(embeddings, embeddings, k=k, metric=metric)
        # Take highest spread in uncertainty
        unc_stddev = uncertainties.std().item()
        if unc_stddev > best_unc_stddev:
            best_unc_stddev = unc_stddev
            best_perts = models
    
    return best_perts
    
