#!/bin/bash

ssls=("simclr" "byol" "moco")
archs=("resnet18" "resnet50")
downstreams=("cifar10" "cifar100" "stl10")

for ssl in ${ssls[@]};
do
    for arch in ${archs[@]};
    do
        for downstream in ${downstreams[@]};
        do
            for seed in {0..4};
            do
                echo $ssl $downstream $arch $seed
                python3 main_baseline_uncertainty.py \
                    --ensemble-dir models/$ssl\_imagenet32_$arch \
                    --n-ens 10 \
                    --ssl $ssl \
                    --arch $arch \
                    --pretrain imagenet32 \
                    --downstream $downstream \
                    --seed $seed \
                    --outfile baseline_uncertainty/$ssl\_imagenet32_$arch\_$downstream/$seed.pth
            done
        done
    done
done
