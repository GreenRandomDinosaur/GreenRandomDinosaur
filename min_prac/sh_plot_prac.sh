#!/bin/bash

python3 plot_prac.py --filename stats/avg/SNIP/mnist_ffn_0.1.mat
python3 plot_prac.py --filename stats/avg/GraSP/mnist_ffn_0.1.mat
python3 plot_prac.py --filename stats/avg/GraSP_abs/mnist_ffn_0.1.mat

python3 plot_prac.py --filename stats/avg/SNIP/cifar10_resnet2x_32_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP/cifar10_resnet2x_32_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP_abs/cifar10_resnet2x_32_0.05.mat

python3 plot_prac.py --filename stats/avg/SNIP/cifar10_vgg13_bn_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP/cifar10_vgg13_bn_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP_abs/cifar10_vgg13_bn_0.05.mat

python3 plot_prac.py --filename stats/avg/SNIP/cifar100_resnet2x_32_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP/cifar100_resnet2x_32_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP_abs/cifar100_resnet2x_32_0.05.mat

python3 plot_prac.py --filename stats/avg/SNIP/cifar100_vgg13_bn_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP/cifar100_vgg13_bn_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP_abs/cifar100_vgg13_bn_0.05.mat

python3 plot_prac.py --filename stats/avg/SNIP/tinyimagenet_resnet2x_32_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP/tinyimagenet_resnet2x_32_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP_abs/tinyimagenet_resnet2x_32_0.05.mat

python3 plot_prac.py --filename stats/avg/SNIP/tinyimagenet_vgg19_bn_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP/tinyimagenet_vgg19_bn_0.05.mat
python3 plot_prac.py --filename stats/avg/GraSP_abs/tinyimagenet_vgg19_bn_0.05.mat


