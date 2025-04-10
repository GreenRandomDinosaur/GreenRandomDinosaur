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
from pruner.GraSP import GraSP
from rand_prune_stat import prune_randomly
from compute_average import average_mean
from fixed_compute_average import fixed_expected


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', default='stats/fixed/', type=str)
# parser.add_argument('--ckpt_dir', default='../min_stats/stats/baseline', type=str)
# parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--dataset', default='tinyimagenet', type=str)
parser.add_argument('--tinyimagenet_dir', default='/workspace/data/tiny-imagenet-200', type=str)
parser.add_argument('--batch_size', default=100, type=int, metavar='N')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ffn',
                    choices=model_names)
parser.add_argument('--depth', default=20, type=int, metavar='N',
                    help='resnet depth')
parser.add_argument('--rounds', default=100, type=int, metavar='N',
                    help='number of total rounds to run')
parser.add_argument('--sample_ratio', default=0.05, type=float, metavar='N',
                    help='1-prune_ratio')
parser.add_argument('--pruning_method', default='SNIP', type=str)                    
args = parser.parse_args()
# state = {k: v for k, v in args._get_kwargs()}


def main():
    if args.dataset=='tinyimagenet':
        dataset, num_classes = prepare_dataset(args, train_set=True, tiny_dir=args.tinyimagenet_dir)
    else: 
        dataset, num_classes = prepare_dataset(args, train_set=True)
    
    if args.arch=='resnet' or args.arch=='resnet2x':
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth
                )  
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
    model.cuda()
    cudnn.benchmark = True
    
    # sample_mode = nn.Conv2d
        # if args.arch == 'ffn' or args.arch == 'lenet':
            # sample_mode = (nn.Conv2d, nn.Linear)
    sample_mode = (nn.Conv2d, nn.Linear)
        
    if args.pruning_method == 'SNIP': pruner1 = SNIP
    else: pruner1 = GraSP
    
    abs_val = False
    if args.pruning_method == 'GraSP_abs':
        abs_val = True
    '''
    if args.pretrained:
        arch = args.arch
        if args.arch=='resnet' or args.arch=='resnet2x': arch = '{}_{}'.format(args.arch, args.depth)
        filename = '{}/{}_{}/ckpt_r0.pt'.format(args.ckpt_dir, args.dataset, arch)
        ckpt = torch.load(filename) 
        model.load_state_dict(ckpt)
    '''
    stats = {}
    stats['num_impt'], stats['num_impt_ratio']=[],[]
    stats['mean_score'], stats['mean_score_ratio']=[],[]
    stats['sample_num'], stats['sample_num_ratio']=[],[]   
    ms, ns = 0., 0.

    model.train()
    trainloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    num_dict, num_sample, threshold_val = pruner1(model, sample_mode, abs_val, args.sample_ratio, trainloader, args.batch_size, num_classes)
    e_num, e_num_ratio, e_mean, e_mean_ratio = fixed_expected(model, num_dict, args.sample_ratio, sample_mode)
    number_eq, mean_eq, _, _, _ = average_mean(model, sample_mode)

    stats['num_sample'] = num_sample
    stats['num_dict'] = num_dict
    stats['threshold_val'] = threshold_val
    stats['expected_num'], stats['expected_num_ratio'] = e_num, e_num_ratio
    stats['expected_mean'], stats['expected_mean_ratio'] = e_mean, e_mean_ratio
    
    for r_idx in range(args.rounds):
        print('round: {}'.format(r_idx))

        stats1 = prune_randomly(model, num_dict, args.sample_ratio, sample_mode)
        
        ms += stats1['mean_score']-stats1['mean_score_ratio']
        ns += stats1['num_impt']-stats1['num_impt_ratio']
        
        for k, v in stats1.items():
            stats[k].append(v)
        
    stats['ms']=ms/args.rounds # average difference between means
    stats['ms_expected']=mean_eq # expected difference between means
    stats['ns']=ns/args.rounds
    stats['ns_expected']=number_eq

    print('ms: {}, ms_expected: {}'.format(ms/args.rounds, mean_eq))
    print('ns: {}, ns_expected: {}'.format(ns/args.rounds, number_eq))
    
    sub_dir = '' #'pretrained' if args.pretrained else 'init'
    save_dir = '{}/{}/{}'.format(args.save_dir, sub_dir, args.pruning_method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    arch = args.arch
    if args.arch=='resnet' or args.arch=='resnet2x': arch = '{}_{}'.format(args.arch, args.depth)
    sio.savemat('{}/{}_{}.mat'.format(save_dir, args.dataset, arch), stats)

if __name__ == '__main__':
    main()
