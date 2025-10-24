
import argparse
from pathlib import Path

import torch

from model_dataset import get_model, get_dataset
from utils import (
    train_binary_logistic_regression,
    evaluate_binary_logistic_regression,
    train_multi_logistic_regression,
    evaluate_multi_logistic_regression,
    infer,
)

TRAIN = {
    "binary": train_binary_logistic_regression,
    "multi": train_multi_logistic_regression,
}
EVAL = {
    "binary": evaluate_binary_logistic_regression,
    "multi": evaluate_multi_logistic_regression,
}
COARSE_LABELS = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13])


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

    # Task type
    parser.add_argument('--task', type=str)

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
    if args.pretrain == "cifar100":
        train_y = COARSE_LABELS[train_y]
        test_y = COARSE_LABELS[test_y]

    # Train and evaluate on test
    print("Training prediction head")
    prediction_head = TRAIN[args.task](infer(embedding_model, train_X), train_y)
    print("Evaluating on test set")
    results = EVAL[args.task](
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
