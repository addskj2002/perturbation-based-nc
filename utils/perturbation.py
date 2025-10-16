
import torch

from utils.params import params_to_model, model_to_params
from utils.gradients import get_gradients
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
        perturbations, gradients, overwrite=overwrite, cache=cache
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
        perturbations, gradients, overwrite=overwrite, cache=cache
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
