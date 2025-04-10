import numpy as np
import torch
import torch.nn as nn


def average_mean(model, sample_mode):
    means = []
    ratios = []
    counts = []
    counts_impt = 0.
    
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
    
        if cond1:
            means.append(m.score.mean().view(-1,1))
            ratios.append(m.score_mask.float().mean().view(-1,1))
            num_weights = m.score.new_full((1,), m.weight.numel())
            counts.append(num_weights.view(-1,1))
            counts_impt += m.score_mask.sum().item()
    
    means = torch.cat(means, 0)
    ratios = torch.cat(ratios, 0)
    counts = torch.cat(counts, 0)
    counts_all = counts.sum()
    
    m1 = means.repeat(1,len(means))
    m2 = m1.transpose(0,1)
    r1 = ratios.repeat(1,len(means))
    r2 = r1.transpose(0,1)
    c1 = counts.repeat(1,len(counts))
    c2 = c1.transpose(0,1)
    
    number_eq = ((c1*c2)*(r1-r2).pow(2)).sum()
    mean_eq = ((c1*c2)*(m1-m2)*(r1-r2)).sum()
    
    number_eq = number_eq/counts_all/2
    # mean_eq = mean_eq/counts_all/counts_impt/2
    mean_eq = mean_eq/counts_all/2

    diff_means = m1-m2
    diff_ratios = r1-r2
    diff_prod = diff_means * diff_ratios
    diff_counts = c1*c2
    
    return number_eq.item(), mean_eq.item(), diff_means, diff_ratios, diff_prod, diff_counts, counts_all.item(), counts_impt
    

