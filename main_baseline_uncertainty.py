
import argparse
import json
from pathlib import Path

import torch
import numpy as np

from model_dataset import get_model, get_dataset
from uncertainty import (
    compute_norm_uncertainty,
    compute_dist_uncertainty,
    compute_ll_uncertainty,
    compute_fv_uncertainty,
)
from utils import infer


N_PRETRAIN = {
    "cifar10": 50_000,
    "cifar100": 50_000,
    "imagenet32": 100_000,
    "stl10": 5_000,
}
METRICS = ["cosine", "euclidean"]


def get_args_parser():
    # Parse
    parser = argparse.ArgumentParser(description='Set arguments for uncertainty quantification', add_help=False)

    # Models configurations
    parser.add_argument('--ensemble-dir', type=str)
    parser.add_argument('--n-ens', type=int)
    parser.add_argument('--ssl', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--pretrain', type=str)

    # Downstream task name
    parser.add_argument('--downstream', type=str)

    # Uncertainty configurations
    parser.add_argument('--k-list', type=str, default=None)
    parser.add_argument('--n-ref', type=int, default=5_000)

    # Output file
    parser.add_argument('--outfile', type=str)

    return parser


def main(args):
    # Get models and dataset
    print("Getting models and dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for idx in range(args.n_ens):
        model_path = f"{args.ensemble_dir}/{idx}.pth"
        new_model = get_model(model_path, args.ssl, args.arch)
        new_model.eval()
        new_model.to(device)
        models.append(new_model)
    dataset = get_dataset(
        args.ssl, args.pretrain, [(args.pretrain, True), (args.downstream, False)]
    )

    # Get uncertainties
    print("Getting inference")
    ref_idx = np.random.choice(N_PRETRAIN[args.pretrain], size=args.n_ref, replace=False)
    Xs = [
        infer(model, dataset[args.pretrain, True][0].to(device))[ref_idx]
        for model in models
    ]
    Ys = [infer(model, dataset[args.downstream, False][0].to(device)) for model in models]
    k_list = [1, 100]
    if args.k_list is not None:
        with open(args.k_list, 'r') as f:
            k_list = json.load(f)
    print("Computing uncertainties")
    uncertainties = {
        'norm': compute_norm_uncertainty(Ys[0]),
        'll': {
            metric: compute_ll_uncertainty(Xs[0], Ys[0], metric=metric)
            for metric in METRICS
        },
        'fv': {
            metric: compute_fv_uncertainty(Ys, metric=metric)
            for metric in METRICS
        },
        'dist': {
            metric: {
                k: compute_dist_uncertainty(Xs[0], Ys[0], k=k, metric=metric)
                for k in k_list
            }
            for metric in METRICS
        },
    }

    # Save
    output_dir = '/'.join(args.outfile.split('/')[:-1])
    if args.outfile[0] == '/':
        output_dir = '/' + output_dir
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(uncertainties, args.outfile)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Uncertainty quantification test script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)