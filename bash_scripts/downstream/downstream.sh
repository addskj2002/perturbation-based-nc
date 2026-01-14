#!/bin/bash

ssls=("simclr" "byol" "moco")
archs=("resnet18" "resnet50")
pretrains=("cifar10" "cifar100")
tasks=("binary" "multi")

for ssl in ${ssls[@]};
do
    for arch in ${archs[@]};
    do
        for pretrain in ${pretrains[@]};
        do
            for task in ${tasks[@]};
            do
                for model_num in {0..9};
                do
                    echo $ssl $pretrain $arch $task $model_num train
                    python3 main_downstream.py \
                        --filename models/$ssl\_$pretrain\_$arch/$model_num.pth \
                        --ssl $ssl \
                        --arch $arch \
                        --pretrain $pretrain \
                        --dataset $pretrain \
                        --task $task \
                        --eval-train \
                        --outfile downstream/$ssl\_$pretrain\_$arch\_$model_num\_$pretrain\_$task\_train.pth
                        
                    echo $ssl $pretrain $arch $task $model_num test
                    python3 main_downstream.py \
                        --filename models/$ssl\_$pretrain\_$arch/$model_num.pth \
                        --ssl $ssl \
                        --arch $arch \
                        --pretrain $pretrain \
                        --dataset $pretrain \
                        --task $task \
                        --outfile downstream/$ssl\_$pretrain\_$arch\_$model_num\_$pretrain\_$task\_test.pth
                done
            done
        done
    done
done
