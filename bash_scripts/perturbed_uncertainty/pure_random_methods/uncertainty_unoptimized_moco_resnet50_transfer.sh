#!/bin/bash

ssl="moco"
arch="resnet50"
downstreams=("cifar10" "cifar100" "stl10")
stddevs=("1e-05" "1.778e-05" "3.162e-05" "5.623e-05" "0.0001" "0.00017783" "0.00031623" "0.00056234" "0.001" "0.00177828" "0.00316228" "0.00562341" "0.01" "0.01778279" "0.03162278" "0.05623413" "0.1" "0.17782794" "0.31622777" "0.56234133" "1.0")

for downstream in ${downstreams[@]};
do
    for stddev in ${stddevs[@]};
    do
        for seed in {0..4};
        do
            for model_num in {0..9};
            do
                echo Transfer Learning $ssl $downstream $arch $stddev $seed $model_num
                python3 main_nc_uncertainty.py \
                    --original-model models/$ssl\_imagenet32_$arch/$model_num.pth \
                    --ensemble-dir perturbed_models/$ssl\_imagenet32_$arch/ckpt_$model_num/random/$stddev \
                    --n-ens 9 \
                    --ssl $ssl \
                    --arch $arch \
                    --pretrain imagenet32 \
                    --downstream $downstream \
                    -k 100 \
                    --metric cosine \
                    --seed $seed \
                    --outfile nc_uncertainty/$ssl\_imagenet32_$arch\_$downstream/test/ckpt_$model_num/random_$stddev\_nc_$seed.pth
            done
        done
    done
done

seed=1234
downstream="imagenet32"

for stddev in ${stddevs[@]};
do
    for model_num in {0..9};
    do
        echo Transfer Learning Train $ssl $downstream $arch $stddev $seed $model_num
        python3 main_nc_uncertainty.py \
            --original-model models/$ssl\_imagenet32_$arch/$model_num.pth \
            --ensemble-dir perturbed_models/$ssl\_imagenet32_$arch/ckpt_$model_num/random/$stddev \
            --n-ens 9 \
            --ssl $ssl \
            --arch $arch \
            --pretrain imagenet32 \
            --downstream $downstream \
            --train \
            -k 100 \
            --metric cosine \
            --seed $seed \
            --outfile nc_uncertainty/$ssl\_imagenet32_$arch\_$downstream/train/ckpt_$model_num/random_$stddev\_nc_$seed.pth
    done
done

