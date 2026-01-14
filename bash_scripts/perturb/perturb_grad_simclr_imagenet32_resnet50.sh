#!/bin/bash

ssl="simclr"
arch="resnet50"
pretrain="imagenet32"
stddevs=("1e-05" "1.778e-05" "3.162e-05" "5.623e-05" "0.0001" "0.00017783" "0.00031623" "0.00056234" "0.001" "0.00177828" "0.00316228" "0.00562341" "0.01" "0.01778279" "0.03162278" "0.05623413" "0.1" "0.17782794" "0.31622777" "0.56234133" "1.0")

for stddev in ${stddevs[@]};
do
    for model_num in {0..9};
    do
        echo Perturbation $ssl $pretrain $arch $stddev $model_num in_space
        python3 main_perturb.py \
            --original-model models/$ssl\_$pretrain\_$arch/$model_num.pth \
            --n-ens 9 \
            --ssl $ssl \
            --arch $arch \
            --pretrain $pretrain \
            --method in_space \
            --seed 1234 \
            --stddev $stddev \
            --inputs inputs/$ssl\_$pretrain\_$arch.pth \
            --cache caches/$ssl\_$pretrain\_$arch\_$model_num.pth \
            --outdir perturbed_models/$ssl\_$pretrain\_$arch/ckpt_$model_num/in_space/$stddev
            
        echo Perturbation $ssl $pretrain $arch $stddev $model_num ortho_space
        python3 main_perturb.py \
            --original-model models/$ssl\_$pretrain\_$arch/$model_num.pth \
            --n-ens 9 \
            --ssl $ssl \
            --arch $arch \
            --pretrain $pretrain \
            --method ortho_space \
            --seed 1234 \
            --stddev $stddev \
            --inputs inputs/$ssl\_$pretrain\_$arch.pth \
            --cache caches/$ssl\_$pretrain\_$arch\_$model_num.pth \
            --outdir perturbed_models/$ssl\_$pretrain\_$arch/ckpt_$model_num/ortho_space/$stddev
            
        echo Perturbation $ssl $pretrain $arch $stddev $model_num gradinv
        python3 main_perturb.py \
            --original-model models/$ssl\_$pretrain\_$arch/$model_num.pth \
            --n-ens 9 \
            --ssl $ssl \
            --arch $arch \
            --pretrain $pretrain \
            --method gradinv \
            --seed 1234 \
            --stddev $stddev \
            --inputs inputs/$ssl\_$pretrain\_$arch\_gradinv.pth \
            --cache caches/$ssl\_$pretrain\_$arch\_gradinv\_$model_num.pth \
            --outdir perturbed_models/$ssl\_$pretrain\_$arch/ckpt_$model_num/gradinv/$stddev
    done
done