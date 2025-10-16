
import argparse
from pathlib import Path

import torch

from model_dataset import get_model, get_dataset
from utils import train_logistic_regression, evaluate_logistic_regression, infer

def get_args_parser():
    # Parse
    parser = argparse.ArgumentParser(description='Set arguments for downstream tasks', add_help=False)

    # Model configurations
    parser.add_argument('--filename', type=str)
    parser.add_argument('--ssl', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--pretrain', type=str)

    # Dataset name
    parser.add_argument('--dataset', type=str)

    # Output filename
    parser.add_argument('--outfile', type=str)

    return parser


def main(args):
    # Get model and dataset
    print("Loading model and dataset")
    embedding_model = get_model(args.filename, args.ssl, args.arch)
    embedding_model.eval()
    datasets = get_dataset(
        args.ssl, args.pretrain, [(args.dataset, True), (args.dataset, False)]
    )
    train_X, train_y = datasets[args.dataset, True]
    test_X, test_y = datasets[args.dataset, False]

    # Train and evaluate on test
    print("Training prediction head")
    prediction_head = train_logistic_regression(infer(embedding_model, train_X), train_y)
    print("Evaluating on test set")
    results = evaluate_logistic_regression(
        prediction_head, infer(embedding_model, test_X), test_y
    )

    # Save
    output_dir = '/'.join(args.outfile.split('/')[:-1])
    if args.outfile[0] == '/':
        output_dir = '/' + output_dir
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.outfile)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Downstream evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
