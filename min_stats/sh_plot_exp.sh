#!/bin/bash

python3 plot_exp.py --filename stats/bn_train/SNIP/cifar10_resnet2x_32.mat
python3 plot_exp.py --filename stats/bn_train/GraSP/cifar10_resnet2x_32.mat
python3 plot_exp.py --filename stats/bn_train/GraSP_abs/cifar10_resnet2x_32.mat


python3 plot_exp.py --filename stats/bn_train/SNIP/cifar10_vgg13_bn.mat
python3 plot_exp.py --filename stats/bn_train/GraSP/cifar10_vgg13_bn.mat
python3 plot_exp.py --filename stats/bn_train/GraSP_abs/cifar10_vgg13_bn.mat


python3 plot_exp.py --filename stats/bn_train/SNIP/cifar100_resnet2x_32.mat
python3 plot_exp.py --filename stats/bn_train/GraSP/cifar100_resnet2x_32.mat
python3 plot_exp.py --filename stats/bn_train/GraSP_abs/cifar100_resnet2x_32.mat


python3 plot_exp.py --filename stats/bn_train/SNIP/cifar100_vgg13_bn.mat
python3 plot_exp.py --filename stats/bn_train/GraSP/cifar100_vgg13_bn.mat
python3 plot_exp.py --filename stats/bn_train/GraSP_abs/cifar100_vgg13_bn.mat



python3 plot_exp.py --filename stats/bn_train/SNIP/tinyimagenet_resnet2x_32.mat
python3 plot_exp.py --filename stats/bn_train/GraSP/tinyimagenet_resnet2x_32.mat
python3 plot_exp.py --filename stats/bn_train/GraSP_abs/tinyimagenet_resnet2x_32.mat


python3 plot_exp.py --filename stats/bn_train/SNIP/tinyimagenet_vgg19_bn.mat
python3 plot_exp.py --filename stats/bn_train/GraSP/tinyimagenet_vgg19_bn.mat
python3 plot_exp.py --filename stats/bn_train/GraSP_abs/tinyimagenet_vgg19_bn.mat

