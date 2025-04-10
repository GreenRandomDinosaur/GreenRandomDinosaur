# ========== Thanks https://github.com/Eric-mingjie/rethinking-network-pruning ============
# ========== we adopt the code from the above link and did modifications ============
# ========== the comments as #=== === were added by us, while the comments as # were the original one ============

from __future__ import print_function

import argparse
import math
import os
import random
#import shutil
#import time
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import models as models

from data_piles import prepare_dataset
from pruner.SNIP import SNIP
from layer_stats import semi_stats
from draw_hist import draw_histogram


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', default='images/dists_log', type=str)
parser.add_argument('--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--dataset', default='mnist', type=str)
parser.add_argument('--batch_size', default=1, type=int, metavar='N')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ffn',
                    choices=model_names)
parser.add_argument('--depth', default=20, type=int, metavar='N',
                    help='resnet depth')
parser.add_argument('--rounds', default=500, type=int, metavar='N',
                    help='number of total rounds to run')
parser.add_argument('--pruning_method', default='SNIP', type=str)                    
args = parser.parse_args()
# state = {k: v for k, v in args._get_kwargs()}


def main():
    transform_train, transform_test, dataloader, num_classes = prepare_dataset(args)
    trainset = dataloader(root='../../data', train=True, download=True, transform=transform_train)
    if args.dataset == 'mnist':
        data_indices = torch.load('indices_mnist.pt')
    elif args.dataset == 'cifar10':
        data_indices = torch.load('indices_cifar10.pt')
    elif args.dataset == 'cifar100':
        data_indices = torch.load('indices_cifar100.pt')
    trainset = torch.utils.data.Subset(trainset, data_indices)
    testset = dataloader(root='../../data', train=False, download=False, transform=transform_test)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)   
   
    pruner1 = SNIP
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    trainiter = iter(trainloader)
    x, y = next(trainiter)


    stats = {}
    for r_idx in range(args.rounds):
        if r_idx%10==0: print('round: {}'.format(r_idx))
        model = models.__dict__[args.arch](num_classes=num_classes)
        model.cuda()
        cudnn.benchmark = True
        
        pruner1(model, x, y)
        one_stat = gather_stats(model)
        combine_stats(stats, one_stat)


    stats_semi = {}
    for r_idx in range(args.rounds):
        if r_idx%10==0: print('round: {}'.format(r_idx))
        model = models.__dict__[args.arch](num_classes=num_classes)
        model.cuda()
        cudnn.benchmark = True   
        
        one_stat = semi_stats(model, x, y)
        combine_stats(stats_semi, one_stat)
    
    
    cnt = 0
    num_layers = cnt_layers(model)
    for k, v in stats.items():
        if cnt >0 and cnt<num_layers-1: 
            draw_histogram(args, k, stats[k], stats_semi[k])
        cnt += 1
        
        # print(stats[k])
        # print(stats_semi[k])
        # print('')


def cnt_layers(model):
    num_layers = 0
    for m in model.modules(): 
        if isinstance(m, nn.Linear):
            num_layers += 1
    return num_layers
        


def combine_stats(stats, one_stat):
    for k, v in one_stat.items(): 
        if k in stats: 
            stats[k].append(v)
        else: 
            stats[k] = [v]



def gather_stats(model):
    layer_stats = {}
    
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            score = m.score.flatten()
            layer_stats[n] = score[0].item()
            
    return layer_stats



if __name__ == '__main__':
    main()
