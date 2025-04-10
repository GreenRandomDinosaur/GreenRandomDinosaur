# ========== Thanks https://github.com/Eric-mingjie/rethinking-network-pruning ============
# ========== we adopt the code from the above link and did modifications ============
# ========== the comments as #=== === were added by us, while the comments as # were the original one ============

from __future__ import print_function

import argparse
import math
import os
import random
#import shutil
import time
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import models as models

from data_piles import prepare_dataset
from pruner.SNIP import SNIP
from pruner.GraSP import GraSP, grasp_data, adjust_batch_size
from mask import set_prune_mask, mask_weights, mask_grads, rescale_weights
from utils import AverageMeter, ProgressMeter, accuracy
from measure_opt import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', default='stats/avg', type=str)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--dataset', default='tinyimagenet', type=str)
parser.add_argument('--tinyimagenet_dir', default='/workspace/data/tiny-imagenet-200', type=str)
parser.add_argument('--batch_size', default=100, type=int, metavar='N')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ffn',
                    choices=model_names)
parser.add_argument('--depth', default=20, type=int, metavar='N',
                    help='resnet depth')
parser.add_argument('--sample_ratio', default=0.05, type=float, metavar='N',
                    help='1-prune_ratio')
parser.add_argument('--pruning_method', default='SNIP', type=str) 

parser.add_argument('--rounds', default=30, type=int)
args = parser.parse_args()
# state = {k: v for k, v in args._get_kwargs()}


def main():
    if args.dataset=='tinyimagenet':
        dataset, num_classes = prepare_dataset(args, train_set=True, tiny_dir=args.tinyimagenet_dir)
    else: 
        dataset, num_classes = prepare_dataset(args, train_set=True)
    trainloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    g = torch.Generator()
    data_random_seed = g.seed()  
    print('seed num: {}'.format(data_random_seed))

    
    # sample_mode = nn.Conv2d
    # if args.arch == 'ffn' or args.arch == 'lenet':
        # sample_mode = (nn.Conv2d, nn.Linear)
    sample_mode = (nn.Conv2d, nn.Linear)    
        
    if args.pruning_method == 'SNIP': pruner1 = SNIP
    else: pruner1 = GraSP
    
    abs_val = False
    if args.pruning_method == 'GraSP_abs':
        abs_val = True
        
    model = renew_model(args, num_classes)
    ori_loss, ori_grad = forward_once(model, sample_mode, trainloader, data_random_seed, args, False) 
    num_dict, num_sample, threshold_val = pruner1(model, sample_mode, abs_val, args.sample_ratio, trainloader, args.batch_size, num_classes)
    ori_num, ori_score = count_ori(model, sample_mode)
    model_dict = model.state_dict()
    model_dict = clone_dict(model_dict)  
    print('ori grad: {}'.format(ori_grad[:10]))
    # print_param_ratio(model)

    # for m in model.modules():
        # if isinstance(m, nn.Conv2d):
            # print(m.weight.view(-1)[:3].tolist())
            # break

    model_dict_cloned = clone_dict(model_dict)
    model.load_state_dict(model_dict_cloned, strict=False)  
    
    stats1 = set_prune_mask(model, sample_mode, 'max', args.sample_ratio)
    mask_weights(model, sample_mode)
    max_loss, max_grad = forward_once(model, sample_mode, trainloader, data_random_seed, args)   
    max_num, max_score = stats1['num_impt'], stats1['mean_score'] 
    max_score_abs = stats1['mean_abs_score']     
    # print('{}, {}'.format(max_loss, max_grad))
    print('max grad: {}'.format(max_grad[:10]))
    print_param_ratio(model)


    nums1, nums2 = [], []
    scores1, scores2 = [], []
    abs_scores1, abs_scores2 = [], []
    losses1, losses2 = [], []
    losses_diff1, losses_diff2 = [], []
    grads1, grads2 = [], []
    grads_diff1, grads_diff2 = [], []
    

    for r_p in range(args.rounds):
        print('\nrand max {}'.format(r_p))
        model_dict_cloned = clone_dict(model_dict)
        model.load_state_dict(model_dict_cloned, strict=False)

        # for m in model.modules():
            # if isinstance(m, nn.Conv2d):
                # print(m.weight.view(-1)[:3].tolist())
                # break
        
        stats1 = set_prune_mask(model, sample_mode, 'rand_max', args.sample_ratio)
        mask_weights(model, sample_mode)
        loss_pruned, grad_pruned = forward_once(model, sample_mode, trainloader, data_random_seed, args)   
        # print('{}, {}'.format(loss_pruned, grad_pruned))

        nums1.append(stats1['num_impt'])
        scores1.append(stats1['mean_score'])
        abs_scores1.append(stats1['mean_abs_score'])
        losses1.append(sum(loss_pruned)/len(trainloader))        
        grads1.append(sum(grad_pruned)/len(trainloader))
                
        diff1 = [abs(x-y) for x,y in zip(loss_pruned, ori_loss)]
        losses_diff1.append(sum(diff1)/len(trainloader))
        diff1 = [abs(x-y) for x,y in zip(grad_pruned, ori_grad)]
        grads_diff1.append(sum(diff1)/len(trainloader))
        
        print('rand max grad {}: {}'.format(r_p, grad_pruned[:10]))
        print_param_ratio(model)

    for r_p in range(args.rounds):
        print('\nrand uni {}'.format(r_p))
        model_dict_cloned = clone_dict(model_dict)
        model.load_state_dict(model_dict_cloned, strict=False)

        # for m in model.modules():
            # if isinstance(m, nn.Conv2d):
                # print(m.weight.view(-1)[:3].tolist())
                # break
        
        stats1 = set_prune_mask(model, sample_mode, 'rand_uni', args.sample_ratio)
        mask_weights(model, sample_mode)
        loss_pruned, grad_pruned = forward_once(model, sample_mode, trainloader, data_random_seed, args)   
        # print('{}, {}'.format(loss_pruned, grad_pruned))

        nums2.append(stats1['num_impt'])
        scores2.append(stats1['mean_score'])
        abs_scores2.append(stats1['mean_abs_score'])
        losses2.append(sum(loss_pruned)/len(trainloader))
        grads2.append(sum(grad_pruned)/len(trainloader))
        
        diff1 = [abs(x-y) for x,y in zip(loss_pruned, ori_loss)]
        losses_diff2.append(sum(diff1)/len(trainloader))
        diff1 = [abs(x-y) for x,y in zip(grad_pruned, ori_grad)]
        grads_diff2.append(sum(diff1)/len(trainloader))        
        
        print('rand uni grad {}: {}'.format(r_p, grad_pruned[:10]))
        print_param_ratio(model)
     
  
    stats = {}  
    if args.pruning_method=='SNIP':
        max_opt = sum([abs(x-y) for x, y in zip(max_loss, ori_loss)])
        max_opt = max_opt / len(trainloader)
        rand_max_opt = sum(losses_diff1)/args.rounds
        rand_uni_opt = sum(losses_diff2)/args.rounds
        
    elif args.pruning_method=='GraSP_abs':
        max_opt = sum([abs(x-y) for x, y in zip(max_grad, ori_grad)])
        max_opt = max_opt/len(trainloader)
        rand_max_opt = sum(grads_diff1)/args.rounds
        rand_uni_opt = sum(grads_diff2)/args.rounds
    else: 
        max_opt = sum(max_grad)
        max_opt = max_opt/len(trainloader)
        rand_max_opt = sum(grads1)/args.rounds
        rand_uni_opt = sum(grads2)/args.rounds 
        
    print('max opt: {}'.format(max_opt))
    print('rand max opt: {}'.format(rand_max_opt))
    print('rand uni opt: {}'.format(rand_uni_opt))
    # stats['ori_opt'] = ori_grad
    stats['max_opt'] = max_opt
    stats['rand_max_opt'] = rand_max_opt
    stats['rand_uni_opt'] = rand_uni_opt
    
    if args.pruning_method=='GraSP':
        max_opt_delta = sum([abs(x-y) for x, y in zip(max_grad, ori_grad)])
        max_opt_delta = max_opt_delta/len(trainloader)
        rand_max_opt_delta = sum(grads_diff1)/args.rounds
        rand_uni_opt_delta = sum(grads_diff2)/args.rounds

        stats['max_opt_delta'] = max_opt_delta
        stats['rand_max_opt_delta'] = rand_max_opt_delta
        stats['rand_uni_opt_delta'] = rand_uni_opt_delta       
    
    
    
    # stats['ori_num'],stats['ori_score'] = ori_num, ori_score
    stats['max_num'],stats['max_score'] = max_num, max_score
    stats['max_score_abs'] = max_score_abs
    stats['rand_max_num'] = sum(nums1)/args.rounds
    stats['rand_max_score'] = sum(scores1)/args.rounds
    stats['rand_max_score_abs'] = sum(abs_scores1)/args.rounds
    stats['rand_uni_num'] = sum(nums2)/args.rounds
    stats['rand_uni_score'] = sum(scores2)/args.rounds
    stats['rand_uni_score_abs'] = sum(abs_scores2)/args.rounds

    print(stats)

    
    arch = args.arch
    if args.arch=='resnet' or args.arch=='resnet2x' or args.arch=='resnet2x_no_bn':
        arch = '{}_{}'.format(args.arch, args.depth) 
    save_dir = '{}/{}'.format(args.save_dir, args.pruning_method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)    

    filename = '{}_{}_{}'.format(args.dataset, arch, args.sample_ratio)
    sio.savemat('{}/{}.mat'.format(save_dir, filename),stats)
    print(stats)
    
    

def renew_model(args, num_classes):
    if args.arch=='resnet' or args.arch=='resnet2x' or args.arch=='resnet2x_no_bn':
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth
                )  
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
    model.cuda()
    cudnn.benchmark = True 
    return model


def compute_corr(x, y):
    x = torch.Tensor(x).cuda()
    y = torch.Tensor(y).cuda()
    # print(x)
    # print(y)

    # tensor2 = torch.Tensor([2]).cuda()
    # x = x.log()/tensor2.log()

    mean_x = x.mean()
    mean_y = y.mean()
    
    std_x = (x-mean_x).pow(2).sum().sqrt()
    std_y = (y-mean_y).pow(2).sum().sqrt()
    
    # print('var_x: {}, var_y: {}'.format(var_x, var_y))
    
    corr = ((x-mean_x)*(y-mean_y)).sum()
    corr = corr/(std_x*std_y)
    
    return corr.item()    


def compute_cnt(x, y, pruning_method):
    x = torch.Tensor(x).cuda()
    y = torch.Tensor(y).cuda()
    
    x_len = x.size(0)
    x1 = x.view(x_len, 1)
    x1 = x1.repeat(1, x_len)
    x2 = x1.transpose(0,1)

    y_len = y.size(0)
    y1 = y.view(y_len, 1)
    y1 = y1.repeat(1, y_len)
    y2 = y1.transpose(0,1)
    
    x_diff = x1-x2
    y_diff = y1-y2
    
    z = x_diff*y_diff
    if pruning_method == 'SNIP':
        z = z<0
    else: 
        z = z>0
    z = z.float()
    z_mean = z.sum()/(z.numel()-x_len)
    return z_mean.item()
    

def clone_dict(dict1):
    model_dict = {}
    for k, v in dict1.items(): 
        model_dict[k]=v.detach().clone()
    return model_dict
            
            
def update_stats_dict(stats, stats1, sampling_method):
    for k1, v1 in stats1.items():
        key1 = '{}_{}'.format(sampling_method, k1)
        if key1 in stats: 
            stats[key1].append(v1)
        else:
            stats[key1]=[v1]
       
       
def update_stats(stats, k1, v1):
    if k1 in stats: 
        stats[k1].append(v1)
    else:
        stats[k1]=[v1]


'''    
def forward_once(model, sample_mode, dataloader, data_random_seed, pruned=True):    
    # model.eval()
    model.train()
    model.zero_grad()
    torch.manual_seed(data_random_seed)
    temperature = 200.
    
    total_loss = 0.
    num_data = 0.
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx==0: print(inputs.view(-1)[:5])
        inputs, targets = inputs.cuda(), targets.cuda()        
        outputs = model(inputs)        
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets, reduction='sum') ###debug needed
        if args.pruning_method == 'GraSP': loss = loss / temperature 
        loss.backward()
        total_loss += loss.item()
        num_data += inputs.size(0)
    total_loss /= num_data
    # print(total_loss)

    g_vec = []
    for m in model.modules(): 
        if isinstance(m, sample_mode):
            if pruned:
                # print(m.prune_mask.mean().item())
                g_vec.append((m.weight.grad*m.prune_mask).detach().clone().view(-1))
            else: 
                g_vec.append(m.weight.grad.detach().clone().view(-1))
            # print(m.weight.grad.sum().item())
    g_vec = torch.cat(g_vec, 0)
    g_vec /= num_data
    grad_norm = g_vec.norm().item()
    
    model.zero_grad()
    return total_loss, grad_norm
'''

            
def forward_once(model, sample_mode, dataloader, data_random_seed, args, pruned=True):    
    model.train()
    torch.manual_seed(data_random_seed)
    temperature = 200.
    
    batch_loss = []
    batch_grad_norm = []
    num_data = 0.
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx==0: print(inputs.view(-1)[:5])
        model.zero_grad()
        inputs, targets = inputs.cuda(), targets.cuda()        
        outputs = model(inputs)
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets, reduction='sum') ###debug needed
        if args.pruning_method == 'GraSP': loss = loss / temperature        
        loss.backward()
        
        batch_loss.append(loss.unsqueeze(0))
        batch_grad_norm.append(get_grad_norm(model, sample_mode, pruned).unsqueeze(0))
        
        # if batch_idx==4: break
        
    batch_loss = torch.cat(batch_loss, 0)
    batch_grad_norm = torch.cat(batch_grad_norm, 0) 

    batch_loss = batch_loss.tolist()
    batch_grad_norm = batch_grad_norm.tolist()    
    
    return batch_loss, batch_grad_norm


def get_grad_norm(model, sample_mode, pruned):
    g_vec = []
    for m in model.modules(): 
        if isinstance(m, sample_mode):
            if pruned:
                g_vec.append((m.weight.grad*m.prune_mask).detach().clone().view(-1))
            else: 
                g_vec.append(m.weight.grad.detach().clone().view(-1))
    g_vec = torch.cat(g_vec, 0)
    # g_vec /= num_data
    grad_norm = g_vec.norm().pow(2)
    # print('ori: {}'.format(grad_norm.item()))
    return grad_norm
    

    

def print_mask_ratio(model,sample_mode):
    t_num = 0.
    m_num = 0.
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
    
        if cond1:
            t_num += m.weight.numel()
            m_num += m.prune_mask.float().sum()
    print('mask ratio: {}'.format(m_num/t_num))
    

def print_param_ratio(model):
    t_num = 0.
    p_num = 0.
    for n, p in model.named_parameters(): 
        if len(p.size())==4 or len(p.size())==2:
            t_num += p.numel()
            p_num += (p.abs()>0).sum().item()
            print('{}: {}'.format(n, (p.abs()>0).sum().item()/p.numel()))
            # print('{}: {}'.format(n, (p.abs()>0).sum().item()))
    print(p_num/t_num)
    
    
def count_ori(model, sample_mode):
    num = 0.
    score = 0.
    for n, m in model.named_modules():
        if isinstance(m, sample_mode):    
            num += m.weight.numel()
            score += m.score.sum().item()
    return num, score
    

if __name__ == '__main__':
    main()
