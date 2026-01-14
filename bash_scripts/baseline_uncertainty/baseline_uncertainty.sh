#!/bin/bash

ssls=("simclr" "byol" "moco")
archs=("resnet18" "resnet50")
pretrains=("cifar10" "cifar100")

for ssl in ${ssls[@]};
do
    for arch in ${archs[@]};
    do
        for pretrain in ${pretrains[@]};
        do
            for seed in {0..4};
            do
                echo $ssl $pretrain $arch $seed
                python3 main_baseline_uncertainty.py \
                    --ensemble-dir models/$ssl\_$pretrain\_$arch \
                    --n-ens 10 \
                    --ssl $ssl \
                    --arch $arch \
                    --pretrain $pretrain \
                    --downstream $pretrain \
                    --seed $seed \
                    --outfile baseline_uncertainty/$ssl\_$pretrain\_$arch\_$pretrain/$seed.pth
            done
        done
    done
done
