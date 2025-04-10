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
from compute_average_grasp import average_mean


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', default='stats/bn_train', type=str)
parser.add_argument('--workers', default=1, type=int, metavar='N',
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
parser.add_argument('--pruning_method', default='GraSP', type=str)                    
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
    
    '''    
    if args.pruning_method == 'SNIP': pruner1 = SNIP
    else: pruner1 = GraSP
    
    abs_val = False
    if args.pruning_method == 'GraSP_abs':
        abs_val = True
    '''

    pruner1 = GraSP
    abs_val = False


    stats = {}
    stats['num_impt'], stats['num_impt_ratio']=[],[]
    stats['mean_score'], stats['mean_score_ratio']=[],[]
    stats['sample_num'], stats['sample_num_ratio']=[],[]
    stats['num_sample'], stats['num_dict'], stats['threshold_val']=[],[],[]    
    
    ms, ms_expected = 0., 0.
    ns, ns_expected = 0., 0.
    diff_m, diff_r, diff_p = [], [], []
    
    for r_idx in range(args.rounds):
        if r_idx%10==0: print('round: {}'.format(r_idx))
        model.init()
        model.eval()

        trainloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        
        num_dict, num_sample, threshold_val = pruner1(model, sample_mode, abs_val, args.sample_ratio, trainloader, args.batch_size, num_classes)
        stats1 = prune_randomly(model, num_dict, args.sample_ratio, sample_mode)
        number_eq, mean_eq, diff_means, diff_ratios, diff_prod, diff_cnt, n1, n2 = average_mean(model, sample_mode)
        
        ms += stats1['mean_score']-stats1['mean_score_ratio']
        ms_expected += mean_eq
        ns += stats1['num_impt']-stats1['num_impt_ratio']
        ns_expected += number_eq
        
        diff_m.append(diff_means.unsqueeze(0))
        diff_r.append(diff_ratios.unsqueeze(0))
        diff_p.append(diff_prod.unsqueeze(0))
        
        for k, v in stats1.items():
            stats[k].append(v)
        stats['num_sample'].append(num_sample)
        stats['num_dict'].append(num_dict)
        stats['threshold_val'].append(threshold_val)
        
    stats['ms']=ms/args.rounds
    stats['ms_expected']=ms_expected/args.rounds
    stats['ns']=ns/args.rounds
    stats['ns_expected']=ns_expected/args.rounds
    
    diff_m = torch.cat(diff_m, 0).mean(dim=0)
    diff_r = torch.cat(diff_r, 0).mean(dim=0)
    diff_p = torch.cat(diff_p, 0).mean(dim=0)
    diff_c = diff_p - diff_m*diff_r
    diff_d = diff_m*diff_r
    
    stats['diff_m'] = diff_m.tolist()
    stats['diff_r'] = diff_r.tolist()
    stats['diff_p'] = diff_p.tolist()
    stats['diff_c'] = diff_c.tolist()

    # print('ms: {}, ms_expected: {}'.format(ms/args.rounds, ms_expected/args.rounds))
    # print('ns: {}, ns_expected: {}'.format(ns/args.rounds, ns_expected/args.rounds))
    
    # print(diff_p)
    # print(diff_c)
    # print(diff_d)
    
    # print_ratio(diff_d, diff_p)
    # print_diff(diff_d, diff_p)
    
    '''
    min_max(diff_p)
    
    p_scale = diff_p.abs().sum()/(diff_p.numel()-diff_p.size(0))
    print('abs mean: {}'.format(p_scale.item()))
    
    diffs = (diff_p-diff_d).abs().sum()/(diff_p.numel()-diff_p.size(0))
    print(diffs.item())
    '''
    
    sum_p = 0.5*(diff_cnt * diff_p).sum()
    sum_c = 0.5*(diff_cnt * diff_c).sum()
    sum_d = 0.5*(diff_cnt * diff_d).sum()
    
    sum_p = sum_p.item()
    sum_c = sum_c.item()
    sum_d = sum_d.item()
    
    print('\n{}_{}_{}'.format(args.dataset, args.arch, args.pruning_method))
    print(sum_p/n1)#/n2)
    print(sum_c/n1)#/n2)
    print(sum_d/n1)#/n2)
  
    # print(diff_p.mean().item())
    # print(diff_c.mean().item())
    # print(diff_d.mean().item())
    
    
    # print(sum_d/sum_p)
    # print(sum_c/sum_p)
    print(abs(sum_c/sum_d))
    
    
    '''
    save_dir = '{}/{}'.format(args.save_dir, args.pruning_method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    arch = args.arch
    if args.arch=='resnet' or args.arch=='resnet2x': arch = '{}_{}'.format(args.arch, args.depth)
    sio.savemat('{}/{}_{}.mat'.format(save_dir, args.dataset, arch), stats)
    '''
    
    
    
    # for m in model.modules():
        # if isinstance(m, nn.BatchNorm2d):
            # print(m.running_mean)
            # print(m.running_var)
    

def print_ratio(nom, denom):
    denom2 = denom + (denom==0).float()
    ratio1 = (nom / denom2).abs().sum()
    ratio1 = ratio1/(denom2.numel()-denom2.size(0))
    ratio1 = ratio1.item()
    print(nom / denom2)
    print(ratio1)


def print_diff(nom, denom):
    denom2 = denom + (denom==0).float()
    ratio1 = ((nom-denom) / denom2).abs().sum()
    ratio1 = ratio1/(denom2.numel()-denom2.size(0))
    ratio1 = ratio1.item()
    print(nom / denom2)
    print(ratio1)

def min_max(x):
    x = x.abs()
    min_val = x.max()
    max_val = x[1,1]

    for i in range(x.size(0)):
        for j in range(x.size(1)):
            if i>j:
                curr = x[i,j]
                if curr < min_val: min_val = curr
                elif curr > max_val: max_val = curr
    
    print('max: {}'.format(max_val))
    print('min: {}'.format(min_val))


if __name__ == '__main__':
    main()
