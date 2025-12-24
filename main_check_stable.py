
import os
import argparse
from pathlib import Path

import torch

from model_dataset import get_model, get_dataset
from utils import is_stable


def get_args_parser():
    # Parse
    parser = argparse.ArgumentParser(description='Set arguments for downstream tasks', add_help=False)

    # Original configurations
    parser.add_argument('--ssl', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--pretrain', type=str)
    parser.add_argument('--model', type=str)

    # Output filename
    parser.add_argument('--outfile', type=str)

    return parser

def main(args):
    # Get models and dataset
    print("Getting models and dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, args.ssl, args.arch)
    model.to(device)
    model.eval()
    dataset = get_dataset(
        args.ssl, args.pretrain, [(args.pretrain, True)]
    )

    # Check stability
    ret = is_stable(model, dataset[args.pretrain, True][0].to(device))

    # Save results
    output_dir = '/'.join(args.outfile.split('/')[:-1])
    if args.outfile[0] == '/':
        output_dir = '/' + output_dir
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(ret, args.outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Check stability of perturbed models', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    