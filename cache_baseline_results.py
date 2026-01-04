
import torch
import numpy as np
from scipy.stats import kendalltau
import pandas as pd
from matplotlib import pyplot as plt
from datasets import load_dataset, load_from_disk

def cache_baselines(ssl, pretrain, arch, downstream, downstream_task):
    print(ssl, pretrain, arch, downstream, downstream_task)
    ret_brier = {
        model_num: {
            method: []
            for method in ["nc", "norm", "dist", "norm_ens", "dist_ens", "fv"]
        }
        for model_num in range(10)
    }
    ret_ent = {
        model_num: {
            method: []
            for method in ["nc", "norm", "dist", "norm_ens", "dist_ens", "fv"]
        }
        for model_num in range(10)
    }
    for seed in range(5):
        baseline_uncertainty = torch.load(f"baseline_uncertainty/{ssl}_{pretrain}_{arch}_{downstream}/{seed}.pth", weights_only=False, map_location="cpu")
        norm_unc = -baseline_uncertainty["norm"][0].clone()
        dist_unc = baseline_uncertainty["dist"]["cosine"][1][0].clone()
        for model_num in range(1, 10):
            norm_unc -= baseline_uncertainty["norm"][model_num].clone()
            dist_unc += baseline_uncertainty["dist"]["cosine"][1][model_num].clone()
        norm_unc /= 10
        dist_unc /= 10
        nc_baseline_uncertainty = -torch.load(f"nc_uncertainty/{ssl}_{pretrain}_{arch}_{downstream}/baseline_nc_{seed}.pth", weights_only=False, map_location="cpu")
        for model_num in range(10):
            downstream_performance = torch.load(f"downstream/{ssl}_{pretrain}_{arch}_{model_num}_{downstream}_{downstream_task}_test.pth")
            ret_brier[model_num]["nc"].append(kendalltau(downstream_performance["brier"], nc_baseline_uncertainty).statistic)
            ret_brier[model_num]["norm"].append(kendalltau(downstream_performance["brier"], -baseline_uncertainty["norm"][model_num]).statistic)
            ret_brier[model_num]["dist"].append(kendalltau(downstream_performance["brier"], baseline_uncertainty["dist"]["cosine"][1][model_num]).statistic)
            ret_brier[model_num]["norm_ens"].append(kendalltau(downstream_performance["brier"], norm_unc).statistic)
            ret_brier[model_num]["dist_ens"].append(kendalltau(downstream_performance["brier"], dist_unc).statistic)
            ret_brier[model_num]["fv"].append(kendalltau(downstream_performance["brier"], baseline_uncertainty["fv"]["cosine"]).statistic)
            ret_ent[model_num]["nc"].append(kendalltau(downstream_performance["pred_entropy"], nc_baseline_uncertainty).statistic)
            ret_ent[model_num]["norm"].append(kendalltau(downstream_performance["pred_entropy"], -baseline_uncertainty["norm"][model_num]).statistic)
            ret_ent[model_num]["dist"].append(kendalltau(downstream_performance["pred_entropy"], baseline_uncertainty["dist"]["cosine"][1][model_num]).statistic)
            ret_ent[model_num]["norm_ens"].append(kendalltau(downstream_performance["pred_entropy"], norm_unc).statistic)
            ret_ent[model_num]["dist_ens"].append(kendalltau(downstream_performance["pred_entropy"], dist_unc).statistic)
            ret_ent[model_num]["fv"].append(kendalltau(downstream_performance["pred_entropy"], baseline_uncertainty["fv"]["cosine"]).statistic)
    ret = {"brier": ret_brier, "pred_entropy": ret_ent}
    torch.save(ret, f"result_cache/{ssl}_{pretrain}_{arch}_{downstream}_{downstream_task}_baseline_corrs.pth")

for downstream_task in ["binary", "multi"]:
    for ssl in ["simclr", "byol", "moco"]:
        for arch in ["resnet18", "resnet50"]:
            for pretrain in ["cifar10", "cifar100"]:
                downstream = pretrain
                cache_baselines(ssl, pretrain, arch, downstream, downstream_task)
            pretrain = "imagenet32"
            for downstream in ["cifar10", "cifar100", "stl10"]:
                cache_baselines(ssl, pretrain, arch, downstream, downstream_task)
    