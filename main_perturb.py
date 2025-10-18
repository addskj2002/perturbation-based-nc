
import argparse
import json

import torch

from model_dataset import get_model, get_dataset, save_model
from utils import perturb, get_simclr_loss, get_byol_loss, get_moco_loss
from stddev_tuning import tuned_perturb


LOSS_FN = {
    'simclr': get_simclr_loss,
    'byol': get_byol_loss,
    'moco': get_moco_loss,
}


def get_args_parser():
    # Parse
    parser = argparse.ArgumentParser(description='Set arguments for generating ensembles', add_help=False)

    # Models configutations
    parser.add_argument('--original-model', type=str)
    parser.add_argument('--n-ens', type=int)
    parser.add_argument('--ssl', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--pretrain', type=str, default=None)

    # Synthetic ensemble configutations
    parser.add_argument('--method', type=str)
    parser.add_argument('--stddev', type=float, default=None)
    parser.add_argument('--stddev-list', type=str, default=None)
    parser.add_argument('--inputs', type=str)
    parser.add_argument('--cache', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('--metric', type=str, default="cosine")

    # Output directory
    parser.add_argument('--outdir', type=str)

    return parser


def main(args):
    # Get inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = get_model(args.original_model, args.ssl, args.arch)
    embedding_model.eval()
    embedding_model.to(device)
    pert_inputs = torch.load(args.inputs, weights_only=False)
    pert_inputs = [x.to(device) for x in pert_inputs]
    generator = (
        torch.manual_seed(args.seed)
        if args.seed is not None else torch.Generator()
    )

    # If standard deviation is given, directly perturb
    if args.stddev is not None:
        models = perturb(
            embedding_model,
            args.stddev,
            args.n_ens,
            args.method,
            device,
            loss_fn=LOSS_FN[args.ssl],
            inputs=pert_inputs,
            cache=args.cache,
            overwrite=args.overwrite,
            generator=generator
        )
        for idx, model in enumerate(models):
            save_model(model, f"{args.outdir}/{idx}.pth", args.ssl, args.arch)

    # Otherwise, tune standard deviation
    else:
        # Training set for finding standard deviation with the best spread
        trainset = get_dataset(
            args.ssl, args.pretrain, [(args.pretrain, True)]
        )[args.pretrain, True][0].to(device)
        # Search space of standard deviations
        stddevs = None
        if args.stddev_list is not None:
            with open(args.stddev_list, 'r') as f:
                stddevs = json.load(f)
        # Get ensemble
        models = tuned_perturb(
            embedding_model,
            args.n_ens,
            args.method,
            trainset,
            device,
            stddevs=stddevs,
            loss_fn=LOSS_FN[args.ssl],
            pert_inputs=pert_inputs,
            cache=args.cache,
            overwrite=args.overwrite,
            k=args.k,
            metric=args.metric,
            generator=generator
        )
        for idx, model in enumerate(models):
            save_model(model, f"{args.outdir}/{idx}.pth", args.ssl, args.arch)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Ensemble generation test script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)