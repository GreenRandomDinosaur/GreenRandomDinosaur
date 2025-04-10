#!/bin/bash

python3 plot_fixed.py --filename stats/fixed/SNIP/mnist_ffn.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP/mnist_ffn.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP_abs/mnist_ffn.mat 

python3 plot_fixed.py --filename stats/fixed/SNIP/cifar10_resnet2x_32.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP/cifar10_resnet2x_32.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP_abs/cifar10_resnet2x_32.mat 

python3 plot_fixed.py --filename stats/fixed/SNIP/cifar10_vgg13_bn.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP/cifar10_vgg13_bn.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP_abs/cifar10_vgg13_bn.mat 

python3 plot_fixed.py --filename stats/fixed/SNIP/cifar100_resnet2x_32.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP/cifar100_resnet2x_32.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP_abs/cifar100_resnet2x_32.mat 

python3 plot_fixed.py --filename stats/fixed/SNIP/cifar100_vgg13_bn.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP/cifar100_vgg13_bn.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP_abs/cifar100_vgg13_bn.mat 

python3 plot_fixed.py --filename stats/fixed/SNIP/tinyimagenet_resnet2x_32.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP/tinyimagenet_resnet2x_32.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP_abs/tinyimagenet_resnet2x_32.mat 

python3 plot_fixed.py --filename stats/fixed/SNIP/tinyimagenet_vgg19_bn.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP/tinyimagenet_vgg19_bn.mat 
python3 plot_fixed.py --filename stats/fixed/GraSP_abs/tinyimagenet_vgg19_bn.mat 


