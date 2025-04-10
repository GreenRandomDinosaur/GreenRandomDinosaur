import numpy as np
import torch
import torch.nn as nn


def fixed_expected(model, num_dict, sample_ratio, sample_mode):
    expected_mean = 0.
    expected_num = 0.
    expected_mean_ratio =0.
    expected_num_ratio = 0.
    num_sample = 0.
    num_sample_ratio = 0.


    tot_weights = 0.
    tot_impt_weights = 0.
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)       
        
        if cond1:
            tot_weights += m.weight.numel()
            tot_impt_weights += num_dict[n]
    sample_ratio = tot_impt_weights/tot_weights

    
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
    
        if cond1:
            sum1 = m.score.sum().item()
            # num1 = m.score_mask.float().sum()
            num_weight1 = m.weight.numel()
            
            num = num_dict[n]
            num_ratio = int(sample_ratio * m.weight.numel())
            
            proportion = 1.* num / num_weight1
            proportion_ratio = 1. * num_ratio / num_weight1
            
            expected_num += num * proportion
            expected_num_ratio += num * proportion_ratio
            
            expected_mean += sum1 * proportion
            expected_mean_ratio += sum1 * proportion_ratio
    
            num_sample += num
            num_sample_ratio += num_ratio
    
    # expected_mean /= num_sample
    # expected_mean_ratio /= num_sample_ratio
    
    return expected_num, expected_num_ratio, expected_mean, expected_mean_ratio
    

