
import os

import torch

from utils.params import params_to_model, model_to_params
from utils.gradients import get_gradients, yield_gradients
from utils.more_linalg import project

EPS = 1e-10

def in_space_perturb(
    model, stddev, num_perturb, device, loss_fn, inputs, cache=None, overwrite=True, generator=None
):
    params = model_to_params(model)
    perturbations = torch.normal(
        mean=0.0, std=stddev, size=(num_perturb, len(params)), generator=generator
    ).to(device)

    # Get gradients
    gradients = get_gradients(
        inputs, model, loss_fn, device, overwrite=overwrite, cache=cache
    )

    # Project to orthogonal space
    new_perturbation = project(
        perturbations, gradients, device, overwrite=overwrite, cache=cache
    )

    # Normalize
    new_perturbation *= (
        perturbations.norm(dim=1, keepdim=True) /
        (new_perturbation.norm(dim=1, keepdim=True) + EPS)
    )

    # Insert to models
    new_params = params.reshape(1, -1) + new_perturbation
    return [
        params_to_model(model, new_params[idx]) for idx in range(num_perturb)
    ]


def ortho_space_perturb(
    model, stddev, num_perturb, device, loss_fn, inputs, cache=None, overwrite=True, generator=None
):
    params = model_to_params(model)
    perturbations = torch.normal(
        mean=0.0, std=stddev, size=(num_perturb, len(params)), generator=generator
    ).to(device)

    # Get gradients
    gradients = get_gradients(
        inputs, model, loss_fn, device, overwrite=overwrite, cache=cache
    )

    # Project to orthogonal space
    new_perturbation = perturbations - project(
        perturbations, gradients, device, overwrite=overwrite, cache=cache
    )

    # Normalize
    new_perturbation *= (
        perturbations.norm(dim=1, keepdim=True) /
        (new_perturbation.norm(dim=1, keepdim=True) + EPS)
    )

    # Insert to models
    new_params = params.reshape(1, -1) + new_perturbation
    return [
        params_to_model(model, new_params[idx]) for idx in range(num_perturb)
    ]


def gradient_inverse_perturb(
    model, stddev, num_perturb, device, loss_fn, inputs, cache=None, overwrite=True, generator=None
):

    # Get gradient second moment
    cache_recovered = False
    if not overwrite and (cache is not None) and os.path.exists(cache):
        data = torch.load(cache, weights_only=False)
        if "desired_vars" in data:
            desired_vars = data["desired_vars"]
            cache_recovered = True
    
    if not cache_recovered:
        print("Cache not hit")
        second_moment = None
        N = 0
        for grad in yield_gradients(
            inputs, model, loss_fn, device, save=False, overwrite=overwrite, cache=cache
        ):
            if second_moment is None:
                second_moment = torch.zeros(grad.shape[0]).to(device)
            second_moment += grad.to(device) ** 2
            N += 1
        second_moment /= N
        second_moment[second_moment < EPS] = 0.0
        desired_vars = 1 / second_moment
        valid_entries = ~(torch.isnan(desired_vars) | torch.isinf(desired_vars))
        desired_vars[~valid_entries] = 0.0
        desired_vars *= (len(desired_vars[valid_entries]) / desired_vars.sum())
        if cache is not None:
            data = {} if not os.path.exists(cache) else torch.load(cache, weights_only=False)
            data["desired_vars"] = desired_vars
            torch.save(data, cache)
        
    desired_stddev = stddev * (desired_vars ** 0.5)
    desired_stddev = torch.stack([desired_stddev for _ in range(num_perturb)])
    
    # Perturb parameters
    params = model_to_params(model)
    perturbations = torch.normal(
        mean=0.0, std=desired_stddev.to("cpu"), generator=generator
    ).to(device)

    # Insert to models
    new_params = params.reshape(1, -1) + perturbations
    return [
        params_to_model(model, new_params[idx]) for idx in range(num_perturb)
    ]


def random_perturb(model, stddev, num_perturb, device, generator=None):
    # Perturb parameters
    params = model_to_params(model)
    perturbations = torch.normal(
        mean=0.0, std=stddev, size=(num_perturb, len(params)), generator=generator
    ).to(device)

    # Insert to models
    new_params = params.reshape(1, -1) + perturbations
    return [
        params_to_model(model, new_params[idx]) for idx in range(num_perturb)
    ]


def perturb(
    model, stddev, num_perturb, method, device, loss_fn=None, inputs=None, cache=None, overwrite=True, generator=None
):
    match method:
        case "in_space":
            return in_space_perturb(
                model, stddev, num_perturb, device, loss_fn, inputs, cache=cache, overwrite=overwrite, generator=generator
            )
        case "ortho_space":
            return ortho_space_perturb(
                model, stddev, num_perturb, device, loss_fn, inputs, cache=cache, overwrite=overwrite, generator=generator
            )
        case "random":
            return random_perturb(model, stddev, num_perturb, device, generator)
        case "gradinv":
            return gradient_inverse_perturb(
                model, stddev, num_perturb, device, loss_fn, inputs, cache=cache, overwrite=overwrite, generator=generator
            )
