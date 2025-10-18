
import argparse
from pathlib import Path

import torch

from model_dataset import get_model
from utils import get_simclr_loss_diff, get_byol_loss_diff, get_moco_loss_diff

LOSS_DIFF = {
    'simclr': get_simclr_loss_diff,
    'byol': get_byol_loss_diff,
    'moco': get_moco_loss_diff,
}

def get_args_parser():
    # Parse
    parser = argparse.ArgumentParser(description='Set arguments for loss preservation test', add_help=False)

    # Models configuration
    parser.add_argument('--model1', type=str)
    parser.add_argument('--model2', type=str)
    parser.add_argument('--ssl', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--pretrain', type=str)

    # Dataset name
    parser.add_argument('--dataset', type=str)

    # Output filename
    parser.add_argument('--outfile', type=str)

    return parser


def main(args):
    # Get models and dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = get_model(args.model1, args.ssl, args.arch)
    model2 = get_model(args.model2, args.ssl, args.arch)
    model1.eval()
    model2.eval()
    model1.to(device)
    model2.to(device)
    dataset = torch.load(args.dataset, weights_only=False, map_location=device)

    # Compute loss difference
    loss_diff = torch.cat([
        LOSS_DIFF[args.ssl](model1, model2, x, device)
        for x in dataset
    ])

    # Save
    output_dir = '/'.join(args.outfile.split('/')[:-1])
    if args.outfile[0] == '/':
        output_dir = '/' + output_dir
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(loss_diff, args.outfile)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Loss preservation test script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)