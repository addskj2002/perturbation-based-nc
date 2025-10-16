
import argparse
import json
from pathlib import Path

import torch

from model_dataset import get_model, get_dataset
from uncertainty import compute_norm_uncertainty, compute_dist_uncertainty
from utils import infer


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
    Xs = [infer(model, dataset[args.pretrain, True][0].to(device)) for model in models]
    Ys = [infer(model, dataset[args.downstream, False][0].to(device)) for model in models]
    k_list = [1, 100]
    if args.k_list is not None:
        with open(args.k_list, 'r') as f:
            k_list = json.load(f)
    print("Computing uncertainties")
    uncertainties = {
        'norm': compute_norm_uncertainty(Ys[0]),
        'dist_cosine': {
            k: compute_dist_uncertainty(Xs[0], Ys[0], k=k, metric='cosine')
            for k in k_list
        },
        'dist_euclidean': {
            k: compute_dist_uncertainty(Xs[0], Ys[0], k=k, metric='euclidean')
            for k in k_list
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