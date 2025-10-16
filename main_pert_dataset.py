
import argparse
from pathlib import Path

import torch

from model_dataset import get_perturbation_input

def get_args_parser():
    # Parse
    parser = argparse.ArgumentParser(description='Set arguments for downstream tasks', add_help=False)

    # Original configurations
    parser.add_argument('--ssl', type=str)
    parser.add_argument('--pretrain', type=str)

    # Downstream config
    parser.add_argument('--downstream', type=str)
    parser.add_argument('--sample-size', type=int, default=200)
    parser.add_argument('--label-uniform', action='store_true')
    parser.add_argument('--seed', type=int, default=None)

    # Output filename
    parser.add_argument('--outfile', type=str)

    return parser


def main(args):
    # Get model and dataset
    dataset = get_perturbation_input(
        ssl=args.ssl,
        pretrain=args.pretrain,
        downstream=args.downstream,
        sample_size=args.sample_size,
        label_uniform=args.label_uniform,
        seed=args.seed
    )

    # Save
    output_dir = '/'.join(args.outfile.split('/')[:-1])
    if args.outfile[0] == '/':
        output_dir = '/' + output_dir
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.outfile)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Downstream evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
