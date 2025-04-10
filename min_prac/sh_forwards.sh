#!/bin/bash




#CUDA_VISIBLE_DEVICES=0 python3 forwards.py --dataset tinyimagenet --batch_size 100 --arch vgg19_bn --pruning_method SNIP --sample_ratio 0.05
CUDA_VISIBLE_DEVICES=0 python3 forwards.py --dataset tinyimagenet --batch_size 100 --arch vgg19_bn --pruning_method GraSP --sample_ratio 0.05
#CUDA_VISIBLE_DEVICES=0 python3 forwards.py --dataset tinyimagenet --batch_size 100 --arch vgg19_bn --pruning_method GraSP_abs --sample_ratio 0.05




#CUDA_VISIBLE_DEVICES=0 python3 forwards.py --dataset tinyimagenet --batch_size 100 --arch resnet2x --depth 32 --pruning_method SNIP --sample_ratio 0.05
CUDA_VISIBLE_DEVICES=0 python3 forwards.py --dataset tinyimagenet --batch_size 100 --arch resnet2x --depth 32 --pruning_method GraSP --sample_ratio 0.05
#CUDA_VISIBLE_DEVICES=0 python3 forwards.py --dataset tinyimagenet --batch_size 100 --arch resnet2x --depth 32 --pruning_method GraSP_abs --sample_ratio 0.05


