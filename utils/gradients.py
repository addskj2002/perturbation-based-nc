
import os
from pathlib import Path

from tqdm import tqdm
import torch
from torch.func import functional_call, vmap, grad


def get_gradients(X, model, loss_fn, device, overwrite=True, cache=None):
    """
    Getting the individual gradients for each of the sample, given the model and the loss
    function

    Inputs:
    - X         : The inputs as a torch tensor
    - model     : The embedding model
    - loss_fn   : The loss function
    - overwrite : If True, perform gradient computation. Otherwise, extract gradient from
                filename if filename exists. In the case that filename does not exists,
                perform the computation anyway
    - cache     : String representing where we could extract the memoized gradients. If
                overwrite is True or filename doesn't exist, this will be the destination
                where the gradient will be saved

    Returns:
    The collection of gradients as a 2D torch tensor, each row being one of the gradients
    """

    # Caching
    if (not overwrite) and (
        (cache is not None) and os.path.exists(cache)
    ):
        filedata = torch.load(cache, weights_only=False)
        if "gradient" in filedata:
            print(f"gradient cache hit {cache}")
            return filedata["gradient"]
        
    # Setup directory if there will be too many gradients
    Ns = [len(loss_fn(model, single_batch, device).flatten()) for single_batch in X]
    N = sum(Ns)
    step = 50
    if N > step:
        output_dir_path = Path(f"{cache}_dir")
        output_dir_path.mkdir(parents=True, exist_ok=True)

    # Compute the gradients
    gradients = []
    kdx = 0
    for idx, single_N in enumerate(Ns):
        for jdx in tqdm(range(single_N)):
            loss = loss_fn(model, X[idx], device).flatten()[jdx]
            loss.backward()
            gradient = []
            for param_name, params in model.named_parameters():
                new_gradient = params.grad
                gradient.append(new_gradient.flatten())
            model.zero_grad()
            X[idx].requires_grad_(requires_grad=False)
            gradient = torch.cat(gradient).to("cpu")
            gradients.append(gradient)
            if (kdx + 1) % step == 0:
                temp = torch.stack(gradients)
                torch.save(temp, f"{cache}_dir/{kdx//step}.pth")
                gradients = []
            kdx += 1
            torch.cuda.empty_cache()
    if len(gradients) > 0:
        gradients = torch.stack(gradients)
        torch.save(gradients, f"{cache}_dir/{kdx//step}.pth")

    # Returns a directory name if we have too many gradients
    if N > step:
        gradients = f"{cache}_dir"
    
    # Caching part 2
    if cache is not None:
        filedata = {}
        if os.path.exists(cache):
            filedata = torch.load(cache, weights_only=False)
        filedata["gradient"] = gradients
        torch.save(filedata, cache)
    return gradients


if __name__ == "__main__":
    pass
