import numpy as np
import torch
import torch.nn as nn


def set_prune_mask(model, sample_mode, sampling_method, sample_ratio):
    stats1 = {}
    stats1['num_impt'], stats1['mean_score'], stats1['mean_abs_score']=0.,0.,0.
    num, total_num = 0., 0.
    prune_num = 0.
    dup_num = 0.

    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
    
        if not cond1: continue
        if sampling_method == 'max':
            prune_mask = m.score_mask.detach().clone()
            if buffer_check(m, 'prune_mask'):
                m.prune_mask = prune_mask
            else:
                m.register_buffer('prune_mask', prune_mask) 
            
            n_impt = m.score_mask.sum()            
            mean_val = (m.score_mask * m.score).sum()
            abs_mean_val = (m.score_mask * m.score).abs().sum()            
            stats1['num_impt'] += n_impt.item()            
            stats1['mean_score'] += mean_val.item()
            stats1['mean_abs_score'] += abs_mean_val.item()
            
            num += n_impt.item()
            total_num += m.weight.numel()
            
        elif sampling_method == 'min':
            prune_mask = m.score_mask_weak.detach().clone()
            if buffer_check(m, 'prune_mask'):
                m.prune_mask = prune_mask
            else:
                m.register_buffer('prune_mask', prune_mask) 
            
            n_impt = (m.score_mask_weak*m.score_mask).sum()            
            mean_val = (m.score_mask_weak * m.score).sum()
            abs_mean_val = (m.score_mask_weak * m.score).abs().sum()            
            stats1['num_impt'] += n_impt.item()            
            stats1['mean_score'] += mean_val.item()
            stats1['mean_abs_score'] += abs_mean_val.item()
            
            num += m.score_mask_weak.sum().item()
            total_num += m.weight.numel()
        elif sampling_method == 'max_uni' or sampling_method == 'min_uni':
            if sampling_method == 'max_uni':
                _, sorted_idx = m.score.view(-1).sort(descending=True)
            else: 
                _, sorted_idx = m.score.view(-1).sort(descending=False)
            num_weights = m.weight.numel()
            sample_num = int(np.floor(num_weights*sample_ratio))
            idx1 = torch.zeros_like(m.score.view(-1))
            idx1[sorted_idx[:sample_num]]=1
            idx1 = idx1.view(m.score.size())
            idx1 = idx1.detach().clone()
            if buffer_check(m, 'prune_mask'):
                m.prune_mask = idx1
            else:
                m.register_buffer('prune_mask', idx1) 

            n_impt = (idx1*m.score_mask).sum()            
            mean_val = (idx1 * m.score).sum()
            abs_mean_val = (idx1 * m.score).abs().sum()            
            stats1['num_impt'] += n_impt.item()            
            stats1['mean_score'] += mean_val.item()
            stats1['mean_abs_score'] += abs_mean_val.item()
            
            num += sample_num
            total_num += m.weight.numel()  

        elif sampling_method == 'min_max':
            _, sorted_idx = m.score.view(-1).sort(descending=False)
            num_weights = m.weight.numel()
            sample_num = m.score_mask.sum().item()
            idx1 = torch.zeros_like(m.score.view(-1))
            idx1[sorted_idx[:sample_num]]=1
            idx1 = idx1.view(m.score.size())
            idx1 = idx1.detach().clone()
            if buffer_check(m, 'prune_mask'):
                m.prune_mask = idx1
            else:
                m.register_buffer('prune_mask', idx1) 

            n_impt = (idx1*m.score_mask).sum()            
            mean_val = (idx1 * m.score).sum()
            abs_mean_val = (idx1 * m.score).abs().sum()            
            stats1['num_impt'] += n_impt.item()            
            stats1['mean_score'] += mean_val.item()
            stats1['mean_abs_score'] += abs_mean_val.item()
            
            num += sample_num
            total_num += m.weight.numel()             


        elif sampling_method == 'max_min':
            _, sorted_idx = m.score.view(-1).sort(descending=True)
            num_weights = m.weight.numel()
            sample_num = m.score_mask_weak.sum().item()
            idx1 = torch.zeros_like(m.score.view(-1))
            idx1[sorted_idx[:sample_num]]=1
            idx1 = idx1.view(m.score.size())
            idx1 = idx1.detach().clone()
            if buffer_check(m, 'prune_mask'):
                m.prune_mask = idx1
            else:
                m.register_buffer('prune_mask', idx1) 

            n_impt = (idx1*m.score_mask).sum()            
            mean_val = (idx1 * m.score).sum() 
            abs_mean_val = (idx1 * m.score).abs().sum()             
            stats1['num_impt'] += n_impt.item()            
            stats1['mean_score'] += mean_val.item()
            stats1['mean_abs_score'] += abs_mean_val.item()
            
            num += sample_num
            total_num += m.weight.numel()  
                
        else: 
            num_weights = m.weight.numel()
            if sampling_method == 'rand_max': 
                sample_num = m.score_mask.sum().item()
            elif sampling_method == 'rand_uni':
                sample_num = int(np.floor(num_weights*sample_ratio))
            elif sampling_method == 'rand_min':
                sample_num = m.score_mask_weak.sum().item()
        
            rand_idx1 = get_rand_idx(m.weight, sample_num)
            if buffer_check(m, 'prune_mask'):
                m.prune_mask = rand_idx1
            else:
                m.register_buffer('prune_mask', rand_idx1) 
            
            n_impt = (rand_idx1*m.score_mask).sum()            
            mean_val = (rand_idx1 * m.score).sum()
            abs_mean_val = (rand_idx1 * m.score).abs().sum()
            stats1['num_impt'] += n_impt.item()            
            stats1['mean_score'] += mean_val.item()
            stats1['mean_abs_score'] += abs_mean_val.item()
            
            num += sample_num
            total_num += m.weight.numel()
            
    # print(num, total_num)
    stats1['mean_score'] = stats1['mean_score']#/ num
    stats1['sample_num'] = num 

    # print(stats1)
    return stats1 
    
    
def buffer_check(m, buffer_name):
    buffer_dict = {k:v for k, v in m.named_buffers()}
    return buffer_name in buffer_dict


def get_rand_idx(weight, sample_num):
    num_weights = weight.numel()

    idx1 = np.random.permutation(num_weights)
    idx1 = idx1[:sample_num]
    
    rand_idx1 = torch.zeros_like(weight).view(-1)            
    rand_idx1[idx1] = 1
    
    rand_idx1 = rand_idx1.view(weight.size())
    return rand_idx1

 
def mask_weights(model, sample_mode):
    total_num, sample_num = 0., 0. 
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
    
        if cond1:
            m.weight.data.mul_(m.prune_mask)
                
            # print('{}: {}'.format(n, m.prune_mask.float().mean()))
            total_num += m.weight.numel()
            sample_num += m.prune_mask.sum().item()
    # print('ratio: {}'.format(sample_num/total_num))
    # print('')    
    

def mask_grads(model, sample_mode):
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
    
        if cond1:         
            m.weight.grad.data.mul_(m.prune_mask)
                
                
         
def rescale_weights(model, sample_mode):
    total_num, sample_num = 0., 0. 
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
    
        if cond1 and isinstance(m, nn.Conv2d):
            mult1 = compute_mult(m.weight, m.prune_mask)
            m.weight.data.mul_(mult1)
                
            # print('{}: {}'.format(n, m.prune_mask.float().mean()))
            # total_num += m.weight.numel()
            # sample_num += m.prune_mask.sum().item()
    # print('ratio: {}'.format(sample_num/total_num))
    # print('')    
                
                
def compute_mult(weight, prune_mask):
    weight = weight.detach().clone()
    sampled_weight = weight*prune_mask
    sampled_weight = sampled_weight.detach().clone()

    sum1 = weight.sum(dim=3,keepdim=True).sum(dim=2,keepdim=True).sum(dim=1,keepdim=True)
    sum2 = sampled_weight.sum(dim=3,keepdim=True).sum(dim=2,keepdim=True).sum(dim=1,keepdim=True)
    add_sum = sampled_weight==0
    mult1 = sum1/(sum2 + add_sum)
    
    repeat_size = [1, weight.size(1), weight.size(2), weight.size(3)]
    mult1 = mult1.repeat(repeat_size)
    return mult1
