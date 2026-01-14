#!/bin/bash

ssl="simclr"
archs=("resnet18" "resnet50")
pretrains=("cifar10" "cifar100" "imagenet32")
indist_pretrains=("cifar10" "cifar100" "imagenet32")

for arch in ${archs[@]};
do
    for pretrain in ${pretrains[@]};
    do
        echo $ssl $pretrain $arch
        python3 main_pert_dataset.py \
            --ssl $ssl \
            --pretrain $pretrain \
            --downstream $pretrain \
            --sample-size 200 \
            --seed 1234 \
            --outfile inputs/$ssl\_$pretrain\_$arch.pth
    done
done

for arch in ${archs[@]};
do
    for pretrain in ${indist_pretrains[@]};
    do
        echo $ssl $pretrain $arch
        python3 main_pert_dataset.py \
            --ssl $ssl \
            --pretrain $pretrain \
            --downstream $pretrain \
            --sample-size 50000 \
            --seed 1234 \
            --outfile inputs/$ssl\_$pretrain\_$arch\_gradinv.pth
    done
done

pretrain="imagenet32"
for arch in ${archs[@]};
do
    echo $ssl $pretrain $arch
    python3 main_pert_dataset.py \
        --ssl $ssl \
        --pretrain $pretrain \
        --downstream $pretrain \
        --sample-size 100000 \
        --seed 1234 \
        --outfile inputs/$ssl\_$pretrain\_$arch\_gradinv.pth
done
