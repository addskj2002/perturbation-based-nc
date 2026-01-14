#!/bin/bash

ssls=("simclr" "byol" "moco")
archs=("resnet18" "resnet50")
pretrains=("cifar10" "cifar100")
downstreams=("cifar10" "cifar100" "stl10")

for ssl in ${ssls[@]};
do
    for arch in ${archs[@]};
    do
        for pretrain in ${pretrains[@]};
        do
            for seed in {0..9};
            do
                echo In Distribution $ssl $pretrain $arch $seed
                python3 main_nc_uncertainty.py \
                    --original-model models/$ssl\_$pretrain\_$arch/9.pth \
                    --ensemble-dir models/$ssl\_$pretrain\_$arch \
                    --n-ens 9 \
                    --ssl $ssl \
                    --arch $arch \
                    --pretrain $pretrain \
                    --downstream $pretrain \
                    -k 100 \
                    --metric cosine \
                    --seed $seed \
                    --outfile nc_uncertainty/$ssl\_$pretrain\_$arch\_$pretrain/baseline_nc_$seed.pth
            done
        done
    done
done

for ssl in ${ssls[@]};
do
    for arch in ${archs[@]};
    do
        for downstream in ${downstreams[@]};
        do
            for seed in {0..9};
            do
                echo Transfer $ssl $downstream $arch $seed
                python3 main_nc_uncertainty.py \
                    --original-model models/$ssl\_imagenet32_$arch/9.pth \
                    --ensemble-dir models/$ssl\_imagenet32_$arch \
                    --n-ens 9 \
                    --ssl $ssl \
                    --arch $arch \
                    --pretrain imagenet32 \
                    --downstream $downstream \
                    -k 100 \
                    --metric cosine \
                    --seed $seed \
                    --outfile nc_uncertainty/$ssl\_imagenet32_$arch\_$downstream/baseline_nc_$seed.pth
            done
        done
    done
done
