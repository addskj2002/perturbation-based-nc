This is the repository for the master thesis project titled "Post-Training Ensemble-Free Uncertainty and Reliability Estimation in Self-Supervised Models"

# General Code Usage

The codebase has a directory `models` with all the pretrained SimCLR, BYOL, and MoCo SSL models, and it is intended to work with these models and their saved file formats specifically.
To use other file format and other model architectures, modify the `get_model` and `save_model` functions in the file `model_dataset/__init__.py`, they are responsible for loading in the model and saving perturbed models.
Similarly, this codebase also only supports the CIFAR-10, CIFAR-100, STL-10, and TinyImagenet datasets, and it will apply the necessary preprocessing needed to have it be fed into SimCLR, BYOL, and MoCo model.
To support the usage of datasets for other type of models, and to support the usage of other datasets, you would need to modify the `get_dataset` function in `model_dataset/__init__.py`.

This codebase also have five main scripts that you can use to run experiments, `main_baseline_uncertainty.py`, `main_downstream.py`, `main_nc_uncertainty.py`, `main_pert_dataset.py`, and `main_perturb.py`.
The following is the usage for `main_baseline_uncertainty.py`:

```python
python3 main_baseline_uncertainty.py \
    --ensemble-dir {string describing the directory where the ensemble is stored} \
    --n-ens {# of models in the ensemble} \
    --ssl {simclr, byol, or moco} \
    --arch {resnet18 or resnet50} \
    --pretrain {cifar10, cifar100, or imagenet32} \
    --downstream {cifar10, cifar100, stl10, or imagenet32} \
    --k-list {optional string describing location of json file containing the list of k} \
    --n-ref {# of ref points in subsample} \
    --seed {optional randomization seed, default 0} \
    --outfile {string representing output file location}
```

This script would compute the $\mathsf{Dist}_k$, $\mathsf{Norm}$, and $\mathsf{FV}$ formula for the downstream dataset on the ensemble, and the `k-list` argument is optional with a default list of `[1, 100]`.
Next, the following is the usage for `main_downstream.py`:

```python
python3 main_downstream.py \
    --filename {string describing the filename of model} \
    --ssl {simclr, byol, or moco} \
    --arch {resnet18 or resnet50} \
    --pretrain {cifar10, cifar100, or imagenet32} \
    --dataset {cifar10, cifar100, stl10, or imagenet32} \
    --task {binary or multi} \
    --eval-train {if tag is used, evaluate on training set of task} \
    --outfile {string representing output file location}
```

This script would first train linear classifier(s) prediction head on the training portion of the `dataset` argument, and evaluate it on each of the test point, specifically using the Brier score, prediction entropy score, and the cross entropy score.
The following is the usage for `main_nc_uncertainty.py`:

```python
python3 main_nc_uncertainty.py \
    --original-model {string describing the filename of model that we want to use to compute NC} \
    --ensemble-dir {string describing the directory where the rest of the ensemble is stored} \
    --n-ens {# of models in the ensemble} \
    --ssl {simclr, byol, or moco} \
    --arch {resnet18 or resnet50} \
    --pretrain {cifar10, cifar100, or imagenet32} \
    --downstream {cifar10, cifar100, stl10, or imagenet32} \
    --train {if used, evaluate NC on training portion of downstream, otherwise we use the test} \
    --k {optional int for k, default is 100} \
    --metric {optional string for distance metric for nearest neighbor, default cosine} \
    --n-ref {optional # of ref points in subsample, default 5000} \
    --seed {optional randomization seed default 0} \
    --outfile {string representing output file location}
```

This script's main purpose is to compute the $\mathsf{NC}_k$ score of the downstream dataset for an ensemble of model, but it can also be used for $\mathsf{PNC}_k$ when the ensemble directory is the synthetic ensemble that is sampled from the distribution $\mathcal{P}(\theta)$.
The last two scripts work hand-in-hand to create these ensembles.
The following is the usage for `main_pert_dataset.py`:

```python
python3 main_pert_dataset.py \
    --ssl {simclr, byol, or moco} \
    --pretrain {cifar10, cifar100, or imagenet32} \
    --downstream {cifar10, cifar100, stl10, or imagenet32} \
    --sample-size {optional subsampling size, defaults to 200} \
    --label-uniform {if used, there will be a roughly equal number of samples for each class} \
    --seed {optional randomization seed default None} \
    --outfile {string representing output file location}
```

This script subsamples from the training portion of a dataset and create a new dataset of pairs of augmented views.
The resulting new dataset can be used to be fed into `main_perturb.py` for perturbing model parameters.
The following is the usage for `main_perturb.py`:

```python
python3 main_perturb.py \
    --original-model {string describing the filename of model that we want to use to compute NC} \
    --n-ens {# of desired perturbations} \
    --ssl {simclr, byol, or moco} \
    --pretrain {cifar10, cifar100, or imagenet32} \
    --method {random, gradinv, in_space, ortho_space} \
    --stddev {float for perturbation magnitude \sigma_{pert}} \
    --inputs {optional filename of the input dataset that the script uses to compute loss gradients} \
    --cache {optional filename of the cache for large gradient computation} \
    --overwrite {if present, overwrite cache} \
    --seed {optional randomization seed}
    --outdir {string representing output directory location}
```

This script samples models from the distribution of perturbed models.
For clarity, the `random` method is the one where the distribution is a spherical Gaussian;
the `gradinv` method is the gradient inverse method;
the `in_space` method is the one where the distribution is along the gradients of the loss;
and the `ortho_space` method is the one where the distribution is orthogonal to the gradients of the loss.

# Replicating Experimental Results

To replicate the results from the paper, we provided bash scripts to run our experiment settings.
They are all in the `bash_scripts` directory, organized into several subdirectories:
- The first is the `baseline_uncertainty` directory, which contains the scripts to compute all the $\mathsf{Dist}_1$, $\mathsf{Norm}$, and $\mathsf{FV}$ scores.
- Next, we have the `downstream` directory that contains the scripts for training the downstream prediction heads and computing the downstream performance.
- We also have the `ensemble_uncertainty` directory, which contains the scripts that computes all the $\mathsf{NC}_{100}$ scores.
- The `perturb` directory contains the scripts to sample the synthetic ensemble of perturbed models, we recommend to run them carefully as the results can take up a lot of disk space.
- The `perturbated_uncertainty` directory contains two subdirectories, `perturbated_uncertainty/pure_random_methods` for computing the $\mathsf{PNC}_{100}$ scores that uses the spherical Gaussian distribution, and the `perturbated_uncertainty/grad_methods` that uses the other gradient-based distributions in the paper's Appendix
- Finally, `caching` contains the bash scripts that caches and summarizes all results, including the Kendall-$\tau$ scores between the estimation methods and the downstream performance and the spread of the $\mathsf{PNC}_{100}$ scores.

`playground.ipynb` can be used to visualize the results after running all the bash scripts.
