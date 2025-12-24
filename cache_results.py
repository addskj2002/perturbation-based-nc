
import torch
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
from matplotlib import pyplot as plt
from datasets import load_dataset, load_from_disk

ALL_STDDEVS = ["1e-05", "1.778e-05", "3.162e-05", "5.623e-05", "0.0001", "0.00017783", "0.00031623", "0.00056234", "0.001", "0.00177828", "0.00316228", "0.00562341", "0.01", "0.01778279", "0.03162278", "0.05623413", "0.1"]

def cache_indist_other_methods(ssl, pretrain, arch, downstream, methods):
    with torch.no_grad():
        for method in methods:
            # Test correlations
            test_correlation = {model_num: [] for model_num in range(10)}
            for model_num in range(10):
                downstream_performance = torch.load(f"downstream/{ssl}_{pretrain}_{arch}_{model_num}_{downstream}_binary_test.pth")
                downstream_performance = downstream_performance["brier"]
                for stddev in ALL_STDDEVS:
                    all_corrs = []
                    for seed in range(5):
                        uncertainty = torch.load(f"nc_uncertainty/{ssl}_{pretrain}_{arch}_{downstream}/test/ckpt_{model_num}/{method}_{stddev}_nc_{seed}.pth")
                        all_corrs.append(kendalltau(downstream_performance, -uncertainty).statistic)
                    test_correlation[model_num].append(all_corrs)
            torch.save(test_correlation, f"result_cache/{ssl}_{pretrain}_{arch}_{downstream}_{method}_test_corrs.pth")
            
            # Train correlations and stddevs
            train_stddevs = {model_num: [] for model_num in range(10)}
            for model_num in range(10):
                for stddev in ALL_STDDEVS:
                    seed = 1234
                    uncertainty = torch.load(f"nc_uncertainty/{ssl}_{pretrain}_{arch}_{downstream}/train/ckpt_{model_num}/{method}_{stddev}_nc_{seed}.pth")
                    train_stddevs[model_num].append(uncertainty.std())
            torch.save(train_stddevs, f"result_cache/{ssl}_{pretrain}_{arch}_{downstream}_{method}_train_stddevs.pth")

def cache_transfer_other_methods(ssl, pretrain, arch, downstream, methods):
    with torch.no_grad():
        for method in methods:
            # Test correlations
            test_correlation = {model_num: [] for model_num in range(10)}
            for model_num in range(10):
                downstream_performance = torch.load(f"downstream/{ssl}_{pretrain}_{arch}_{model_num}_{downstream}_binary_test.pth")
                downstream_performance = downstream_performance["brier"]
                for stddev in ALL_STDDEVS:
                    all_corrs = []
                    for seed in range(5):
                        uncertainty = torch.load(f"nc_uncertainty/{ssl}_{pretrain}_{arch}_{downstream}/test/ckpt_{model_num}/{method}_{stddev}_nc_{seed}.pth")
                        all_corrs.append(kendalltau(downstream_performance, -uncertainty).statistic)
                    test_correlation[model_num].append(all_corrs)
            torch.save(test_correlation, f"result_cache/{ssl}_{pretrain}_{arch}_{downstream}_{method}_test_corrs.pth")
            
            # Train correlations and stddevs
            train_stddevs = {model_num: [] for model_num in range(10)}
            for model_num in range(10):
                for stddev in ALL_STDDEVS:
                    seed = 1234
                    uncertainty = torch.load(f"nc_uncertainty/{ssl}_{pretrain}_{arch}_{pretrain}/train/ckpt_{model_num}/{method}_{stddev}_nc_{seed}.pth")
                    train_stddevs[model_num].append(uncertainty.std())
            torch.save(train_stddevs, f"result_cache/{ssl}_{pretrain}_{arch}_{pretrain}_{method}_train_stddevs.pth")

for ssl in ["simclr", "byol", "moco"]:
    for pretrain in ["cifar10", "cifar100"]:
        for arch in ["resnet18", "resnet50"]:
            print(ssl, pretrain, arch, pretrain)
            cache_indist_other_methods(ssl, pretrain, arch, pretrain, ["random"])

for ssl in ["simclr", "byol", "moco"]:
    for downstream in ["cifar10", "cifar100", "stl10"]:
        for arch in ["resnet18", "resnet50"]:
            print(ssl, "imagenet32", arch, downstream)
            cache_transfer_other_methods(ssl, "imagenet32", arch, downstream, ["random"])

ssl="simclr"
for pretrain in ["cifar10", "cifar100"]:
    for arch in ["resnet18", "resnet50"]:
        print(ssl, pretrain, arch, pretrain, "other_methods")
        cache_indist_other_methods(ssl, pretrain, arch, pretrain, ["in_space", "ortho_space", "gradinv"])

ssl="simclr"
for downstream in ["cifar10", "cifar100", "stl10"]:
    for arch in ["resnet18", "resnet50"]:
        print(ssl, "imagenet32", arch, downstream, "other_methods")
        cache_transfer_other_methods(ssl, "imagenet32", arch, downstream, ["in_space", "ortho_space", "gradinv"])

            
