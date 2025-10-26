
from pathlib import Path
import random

import torchvision
import torch

from model_dataset.misc import STL_TRANSFORM
from model_dataset.simclr import get_model as get_model_simclr
from model_dataset.byol import get_model as get_model_byol
from model_dataset.moco import get_model as get_model_moco
from model_dataset.simclr import save_model as save_model_simclr
from model_dataset.byol import save_model as save_model_byol
from model_dataset.moco import save_model as save_model_moco
from model_dataset.simclr import get_transform as get_transform_simclr
from model_dataset.byol import get_transform as get_transform_byol
from model_dataset.moco import get_transform as get_transform_moco
from model_dataset.simclr import get_perturbation_transform as get_perturbation_transform_simclr
from model_dataset.tiny_imagenet import TinyImageNetDataset
from model_dataset.stl10 import STL10


DATASET_DIR = "./data"
ALL_DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
    "imagenet32": TinyImageNetDataset,
    "stl10": STL10,
}


def get_model(filename, ssl, arch):
    match ssl:
        case 'simclr':
            return get_model_simclr(filename, arch)
        case 'byol':
            return get_model_byol(filename, arch)
        case 'moco':
            return get_model_moco(filename, arch)
        case _:
            raise ValueError(f'SSL {ssl} not supported')


def get_dataset(ssl, pretrain, dataset_names):
    """
    Get the dataset as tensors

    Inputs:
    - ssl           : One of 'simclr', 'byol', 'moco', the self-supervised learning method
    - pretrain      : One of 'cifar10', 'cifar100', 'tinyimagenet32', 'stl10', the dataset
                    used for pretraining
    - dataset_names : List of tuples (name, train), where name is the dataset you want to
                    load (again, one of 'cifar10', 'cifar100', 'tinyimagenet32', 'stl10')
                    and train is a boolean indicating whether to load the training

    Returns:
    Dictionary mapping (name, train) to (inputs, labels), where inputs is a tensor
    """
    # Get the transformation
    match ssl:
        case 'simclr':
            transform = get_transform_simclr(pretrain)
        case 'byol':
            transform = get_transform_byol()
        case 'moco':
            transform = get_transform_moco()
        case _:
            raise ValueError(f'SSL {ssl} not supported')

    # Load the datasets
    ret = {}
    for name, train in dataset_names:
        dataset = ALL_DATASETS[name](
            root=DATASET_DIR,
            train=train,
            download=True,
            transform=(
                transform if name != 'stl10' else
                torchvision.transforms.Compose([STL_TRANSFORM, transform])
            ),
        )
        tensorized_input = torch.stack([dataset[i][0] for i in range(len(dataset))])
        tensorized_label = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        ret[name, train] = (tensorized_input, tensorized_label)
    return ret


def get_perturbation_input(
    ssl, pretrain, downstream, sample_size=200, label_uniform=True, seed=None
):
    """
    Get a subset of the dataset as tensors for perturbation

    Inputs:
    - ssl          : One of 'simclr', 'byol', 'moco', the self-supervised learning method
    - pretrain     : One of 'cifar10', 'cifar100', 'tinyimagenet32', 'stl10', the dataset
                   used for pretraining
    - downstream   : One of 'cifar10', 'cifar100', 'tinyimagenet32', 'stl10', the dataset
                   used for downstream tasks
    - sample_size  : The number of samples to use
    - label_uniform: Whether to sample uniformly across labels
    - seed         : Random seed for sampling

    Returns:
    A tensor for the input data for perturbation
    """
    sampled_idxs = []
    random.seed(seed)

    # Get transform
    match ssl:
        case 'simclr':
            transform = get_perturbation_transform_simclr(pretrain, downstream)
        case _:
            raise ValueError(f'SSL {ssl} not supported')
    
    # Get dataset
    dataset = ALL_DATASETS[downstream](
        root=DATASET_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    tensorized_input = torch.stack([dataset[i][0] for i in range(len(dataset))])

    # Choose by labels
    if label_uniform:
        all_labels = [dataset[i][1] for i in range(len(dataset))]
        label_to_idxs = {}
        for idx, label in enumerate(all_labels):
            if label not in label_to_idxs:
                label_to_idxs[label] = []
            label_to_idxs[label].append(idx)
        overflows = random.sample(
            list(label_to_idxs.keys()), sample_size % len(label_to_idxs)
        )
        for label, idxs in label_to_idxs.items():
            sampled_idxs += random.sample(
                idxs, sample_size // len(label_to_idxs) + (1 if label in overflows else 0)
            )
        random.shuffle(sampled_idxs)
    
    # Choose randomly
    else:
        sampled_idxs = random.sample(range(len(dataset)), sample_size)

    return [
        tensorized_input[sampled_idxs[jdx*50:(jdx+1)*50]]
        for jdx in range(1 + (sample_size - 1) // 50)
    ]


def save_model(model, filename, ssl, arch):
    """
    Save the model to a file

    Inputs:
    - model    : The model to save
    - filename : The filename to save the model to
    - ssl      : One of 'simclr', 'byol', 'moco', the self-supervised learning method
    """
    # Create output directory if it doesn't exist
    output_dir = '/'.join(filename.split('/')[:-1])
    if filename[0] == '/':
        output_dir = '/' + output_dir
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Save the model
    if ssl == "byol":
        save_model_byol(model, filename)
    elif ssl == "simclr":
        save_model_simclr(model, filename, arch)
    elif ssl == "moco":
        save_model_moco(model, filename, arch)
