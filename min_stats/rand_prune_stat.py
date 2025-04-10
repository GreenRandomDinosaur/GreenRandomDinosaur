import numpy as np
import torch
import torch.nn as nn


def prune_randomly(model, num_dict, sample_ratio, sample_mode):
    stats1 = {}
    stats1['num_impt'] = 0
    stats1['num_impt_ratio'] = 0

    stats1['mean_score'] = 0.
    stats1['mean_score_ratio'] = 0.
    
    num = 0.
    num_ratio = 0.
    total_num = 0.

    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
    
        if cond1:
            num_weights = m.weight.numel()
            sample_num1 = num_dict[n]
            sample_num2 = int(np.floor(num_weights*sample_ratio))            
            
            idx1 = np.random.permutation(num_weights)
            idx1 = idx1[:sample_num1]
            
            idx2 = np.random.permutation(num_weights)
            idx2 = idx2[:sample_num2]
            
            rand_idx1 = torch.zeros_like(m.weight).view(-1)
            rand_idx2 = torch.zeros_like(m.weight).view(-1)
            
            rand_idx1[idx1] = 1
            rand_idx2[idx2] = 1
            
            rand_idx1 = rand_idx1.view(m.weight.size())
            rand_idx2 = rand_idx2.view(m.weight.size())
            
            n_impt = (rand_idx1*m.score_mask).sum()
            n_impt_ratio = (rand_idx2*m.score_mask).sum()
            
            mean_val = (rand_idx1 * m.score).sum()
            mean_val_ratio = (rand_idx2 * m.score).sum()
            
            stats1['num_impt'] += n_impt.item()
            stats1['num_impt_ratio'] += n_impt_ratio.item()
            
            stats1['mean_score'] += mean_val.item()
            stats1['mean_score_ratio'] += mean_val_ratio.item()
            
            num += sample_num1
            num_ratio += sample_num2
            total_num += m.weight.numel()
           
    # print(num, num_ratio, total_num)        
    stats1['mean_score'] = stats1['mean_score']#/ num
    stats1['mean_score_ratio'] = stats1['mean_score_ratio']#/ num_ratio
    stats1['sample_num'] = num
    stats1['sample_num_ratio'] = num_ratio
    
    return stats1 
    
    
    
