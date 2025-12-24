
import argparse
import json
from pathlib import Path

import torch
import numpy as np

from model_dataset import get_model, get_dataset
from uncertainty import compute_nc_uncertainty
from utils import infer


N_PRETRAIN = {
    "cifar10": 50_000,
    "cifar100": 50_000,
    "imagenet32": 100_000,
    "stl10": 5_000,
}


def get_args_parser():
    # Parse
    parser = argparse.ArgumentParser(description='Set arguments for uncertainty quantification', add_help=False)

    # Models configurations
    parser.add_argument('--original-model', type=str)
    parser.add_argument('--ensemble-dir', type=str)
    parser.add_argument('--n-ens', type=int)
    parser.add_argument('--ssl', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--pretrain', type=str)

    # Downstream task name
    parser.add_argument('--downstream', type=str)
    parser.add_argument('--train', action='store_true')

    # Uncertainty configurations
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('--metric', type=str, default="cosine")
    parser.add_argument('--n-ref', type=int, default=5_000)
    parser.add_argument('--seed', type=int, default=0)

    # Output file
    parser.add_argument('--outfile', type=str)

    return parser


def main(args):
    # Get models and dataset
    print("Getting models and dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model = get_model(args.original_model, args.ssl, args.arch)
    original_model.eval()
    original_model.to(device)
    models = [original_model]
    for idx in range(args.n_ens):
        model_path = f"{args.ensemble_dir}/{idx}.pth"
        new_model = get_model(model_path, args.ssl, args.arch)
        new_model.eval()
        new_model.to(device)
        models.append(new_model)
    dataset = get_dataset(
        args.ssl, args.pretrain, [(args.pretrain, True), (args.downstream, args.train)]
    )

    # Get uncertainties
    print("Getting inference")
    np.random.seed(args.seed)
    ref_idx = np.random.choice(N_PRETRAIN[args.pretrain], size=args.n_ref, replace=False)
    Xs = [
        infer(model, dataset[args.pretrain, True][0][ref_idx].to(device))
        for model in models
    ]
    Ys = [infer(model, dataset[args.downstream, args.train][0].to(device)) for model in models]
    print("Computing uncertainties")
    uncertainties = compute_nc_uncertainty(Xs, Ys, k=args.k, metric=args.metric)

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