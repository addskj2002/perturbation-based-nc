#!/bin/bash

ssls=("simclr")
archs=("resnet18" "resnet50")
downstreams=("cifar10" "cifar100" "stl10" "imagenet32")
tasks=("binary" "multi")

for ssl in ${ssls[@]};
do
    for arch in ${archs[@]};
    do
        for downstream in ${downstreams[@]};
        do
            for task in ${tasks[@]};
            do
                for model_num in {0..9};
                do
                    echo $ssl $downstream $arch $task $model_num train
                    python3 main_downstream.py \
                        --filename models/$ssl\_imagenet32_$arch/$model_num.pth \
                        --ssl $ssl \
                        --arch $arch \
                        --pretrain imagenet32 \
                        --dataset $downstream \
                        --task $task \
                        --eval-train \
                        --outfile downstream/$ssl\_imagenet32_$arch\_$model_num\_$downstream\_$task\_train.pth
                        
                    echo $ssl $downstream $arch $task $model_num test
                    python3 main_downstream.py \
                        --filename models/$ssl\_imagenet32_$arch/$model_num.pth \
                        --ssl $ssl \
                        --arch $arch \
                        --pretrain imagenet32 \
                        --dataset $downstream \
                        --task $task \
                        --outfile downstream/$ssl\_imagenet32_$arch\_$model_num\_$downstream\_$task\_test.pth
                done
            done
        done
    done
done
