import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from .pruner_utils import *



def one_hot(targets):
    num_class = targets.max()-targets.min()+1
    targets = F.one_hot(targets%num_class)
    return targets.float()


def SNIP(model, sample_mode, abs_val, sample_ratio, dataloader, batch_size, num_classes, num_class_samples=10):
    inputs, targets = grasp_data(dataloader, num_classes, num_class_samples)
    batch_size = adjust_batch_size(inputs.size(0), batch_size)
    num_iters = inputs.size(0)//batch_size
    # update_batchnorm_stats(model, inputs, batch_size)
    # update_batchnorm_stats(model, dataloader)
    # model.eval()
    model.train()
    # print('end of gathering data')

    
    grads = {}
    for i in range(num_iters):
        model.zero_grad()
        inputs1 = inputs[i*batch_size:(i+1)*batch_size,:,:,:]
        targets1 = targets[i*batch_size:(i+1)*batch_size]
        inputs1 = inputs1.cuda()
        targets1 = targets1.cuda()
            
        outputs = model(inputs1)
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets1)
        loss.backward()
        grads1 = gather_grads(model, sample_mode)
        update_grads(grads, grads1)
    model.zero_grad()  
    # reset_batchnorm_stats(model)
    '''

    grads = {}
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()  

        model.zero_grad()
        outputs = model(inputs)        
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets, reduction='sum') ###debug needed
        loss.backward()
        grads1 = gather_grads(model, sample_mode)
        update_grads(grads, grads1)
    model.zero_grad()      
    '''
    
    
    all_params = []
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
        
        if cond1:
            # print('trouble shooting: {}'.format(n))
            scores = (grads[m] * m.weight).abs()
            scores = copy.deepcopy(scores.detach())
            
            buffer_keys = [key for key, buffer in m.named_buffers()]
            if 'score' in buffer_keys: 
                m.score = scores
            else: 
                m.register_buffer('score', scores)
            all_params.append(scores.view(-1))
    
    all_params = torch.cat(all_params, 0)
    all_params = torch.sort(all_params, descending=True)[0]
    num_sample = int(np.floor(all_params.numel()*sample_ratio))
    if sample_ratio==1: num_sample = all_params.numel()-1
    threshold_val = all_params[num_sample]
    # print('threshold_val: {}'.format(threshold_val))
    threshold_val_weak = all_params[all_params.numel()-num_sample]
    
    num_dict = {}
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
    
        if cond1:
            score_mask = (m.score >= threshold_val).long()
            if 'score_mask' in buffer_keys:
                m.score_mask = score_mask
            else: 
                m.register_buffer('score_mask', score_mask) 
            num_dict[n] = m.score_mask.sum().item()
            
            score_mask_weak = (m.score <= threshold_val_weak).long()
            if 'score_mask_weak' in buffer_keys:
                m.score_mask_weak = score_mask_weak
            else: 
                m.register_buffer('score_mask_weak', score_mask_weak) 
            # num_dict_weak[n] = m.score_mask_weak.sum().item()            
            
    return num_dict, num_sample, threshold_val.item()
    
    
    