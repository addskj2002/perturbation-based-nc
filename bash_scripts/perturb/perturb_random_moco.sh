#!/bin/bash

ssl="moco"
archs=("resnet18" "resnet50")
pretrains=("cifar10" "cifar100" "imagenet32")
stddevs=("1e-05" "1.778e-05" "3.162e-05" "5.623e-05" "0.0001" "0.00017783" "0.00031623" "0.00056234" "0.001" "0.00177828" "0.00316228" "0.00562341" "0.01" "0.01778279" "0.03162278" "0.05623413" "0.1" "0.17782794" "0.31622777" "0.56234133" "1.0")

for arch in ${archs[@]};
do
    for pretrain in ${pretrains[@]};
    do
        for stddev in ${stddevs[@]};
        do
            for model_num in {0..9};
            do
                echo Perturbation $ssl $pretrain $arch $stddev $model_num
                python3 main_perturb.py \
                    --original-model models/$ssl\_$pretrain\_$arch/$model_num.pth \
                    --n-ens 9 \
                    --ssl $ssl \
                    --arch $arch \
                    --pretrain $pretrain \
                    --method random \
                    --seed 1234 \
                    --stddev $stddev \
                    --outdir perturbed_models/$ssl\_$pretrain\_$arch/ckpt_$model_num/random/$stddev
            done
        done
    done
done
