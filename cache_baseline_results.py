
import torch
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
from matplotlib import pyplot as plt
from datasets import load_dataset, load_from_disk

ALL_STDDEVS = ["1e-05", "1.778e-05", "3.162e-05", "5.623e-05", "0.0001", "0.00017783", "0.00031623", "0.00056234", "0.001", "0.00177828", "0.00316228", "0.00562341", "0.01", "0.01778279", "0.03162278", "0.05623413", "0.1"]

def cache_baselines(ssl, pretrain, arch, downstream):
    ret = {
        model_num: {
            method: []
            for method in ["nc", "norm", "dist", "fv"]
        }
        for model_num in range(10)
    }
    for seed in range(5):
        baseline_uncertainty = torch.load(f"baseline_uncertainty/{ssl}_{pretrain}_{arch}_{downstream}/{seed}.pth", weights_only=False, map_location="cpu")
        nc_baseline_uncertainty = -torch.load(f"nc_uncertainty/{ssl}_{pretrain}_{arch}_{downstream}/baseline_nc_{seed}.pth", weights_only=False, map_location="cpu")
        for model_num in range(10):
            downstream_performance = torch.load(f"downstream/{ssl}_{pretrain}_{arch}_{model_num}_{downstream}_binary_test.pth")
            downstream_performance = downstream_performance["brier"]
            ret[model_num]["nc"].append(kendalltau(downstream_performance, nc_baseline_uncertainty).statistic)
            ret[model_num]["norm"].append(kendalltau(downstream_performance, baseline_uncertainty["norm"]).statistic)
            ret[model_num]["dist"].append(kendalltau(downstream_performance, baseline_uncertainty["dist"]["cosine"][1]).statistic)
            ret[model_num]["fv"].append(kendalltau(downstream_performance, baseline_uncertainty["fv"]["cosine"]).statistic)
    torch.save(ret, f"result_cache/{ssl}_{pretrain}_{arch}_{downstream}_baseline_corrs.pth")

for ssl in ["simclr", "byol", "moco"]:
    for arch in ["resnet18", "resnet50"]:
        for pretrain in ["cifar10", "cifar100"]:
            downstream = pretrain
            cache_baselines(ssl, pretrain, arch, downstream)
        pretrain = "imagenet32"
        for downstream in ["cifar10", "cifar100", "stl10"]:
            cache_baselines(ssl, pretrain, arch, downstream)
    