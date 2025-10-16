
import copy

import torch

def params_to_model(model, params):
    new_model = copy.deepcopy(model)
    idx = 0
    for name, param in new_model.named_parameters():
        shape = param.shape
        size = 1
        for axis in shape:
            size *= axis
        with torch.no_grad():
            param.copy_(params[idx:idx+size].reshape(shape))
        idx += size
    return new_model

def model_to_params(model):
    params = [param.flatten() for param in model.parameters()]
    return torch.cat(params)
