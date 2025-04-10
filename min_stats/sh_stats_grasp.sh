#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 stats_grasp.py --dataset mnist --batch_size 100 --arch ffn --pruning_method GraSP --sample_ratio 0.1


CUDA_VISIBLE_DEVICES=0 python3 stats_grasp.py --dataset cifar10 --batch_size 100 --arch vgg13_bn --pruning_method GraSP
CUDA_VISIBLE_DEVICES=0 python3 stats_grasp.py --dataset cifar100 --batch_size 100 --arch vgg13_bn --pruning_method GraSP


CUDA_VISIBLE_DEVICES=0 python3 stats_grasp.py --dataset cifar10 --batch_size 100 --arch resnet2x --depth 32 --pruning_method GraSP
CUDA_VISIBLE_DEVICES=0 python3 stats_grasp.py --dataset cifar100 --batch_size 100 --arch resnet2x --depth 32 --pruning_method GraSP

