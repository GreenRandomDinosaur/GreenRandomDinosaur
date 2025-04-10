import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
import time 
from .pruner_utils import *


def GraSP(model, sample_mode, abs_val, sample_ratio, dataloader, batch_size, num_classes, num_class_samples=10):
    if abs_val: temperature = 1.
    else: temperature = 200.

    inputs, targets = grasp_data(dataloader, num_classes, num_class_samples)
    batch_size = adjust_batch_size(inputs.size(0), batch_size)
    num_iters = inputs.size(0)//batch_size
    # update_batchnorm_stats(model, inputs, batch_size)
    # update_batchnorm_stats(model, dataloader)
    # model.eval()
    model.train()
    
    weights = gather_weights(model, sample_mode)
    grads = [torch.zeros_like(w) for w in weights]
    for i in range(num_iters):
        model.zero_grad()
        inputs1 = inputs[i*batch_size:(i+1)*batch_size,:,:,:]
        targets1 = targets[i*batch_size:(i+1)*batch_size]
        inputs1 = inputs1.cuda()
        targets1 = targets1.cuda()   
        
        outputs = model(inputs1)
        outputs /= temperature
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets1, reduction='sum')
        grads1 = autograd.grad(loss, weights, create_graph=False)
        grads = [g+g1.detach().clone() for g,g1 in zip(grads, grads1)]


    # for batch_idx, (inputs1, targets1) in enumerate(dataloader):
        # inputs1, targets1 = inputs1.cuda(), targets1.cuda()        
            
        # model.zero_grad()
        # outputs = model(inputs1)
        # outputs /= temperature
        # loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets1, reduction='sum')
        # grads1 = autograd.grad(loss, weights, create_graph=False)
        # grads = [g+g1.detach().clone() for g,g1 in zip(grads, grads1)]

    Hg = {}
    for i in range(num_iters):
        model.zero_grad()
        inputs1 = inputs[i*batch_size:(i+1)*batch_size,:,:,:]
        targets1 = targets[i*batch_size:(i+1)*batch_size]
        inputs1 = inputs1.cuda()
        targets1 = targets1.cuda()     

        model.zero_grad()    
        outputs = model(inputs1)
        outputs /= temperature
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets1, reduction='sum')        
        grads1 = autograd.grad(loss, weights, create_graph=True)


    # for batch_idx, (inputs1, targets1) in enumerate(dataloader):
        # inputs1, targets1 = inputs1.cuda(), targets1.cuda() 
        
        # model.zero_grad()    
        # outputs = model(inputs1)
        # outputs /= temperature
        # loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets1, reduction='sum')        
        # grads1 = autograd.grad(loss, weights, create_graph=True)
        
        loss = 0.
        for g, g1 in zip(grads, grads1): 
            loss += (g*g1).sum()
        loss.backward()
        
        hg1 = gather_grads(model, sample_mode)
        update_grads(Hg, hg1)
    model.zero_grad()
    # reset_batchnorm_stats(model)
        
    all_params = []
    for m in model.modules():
        cond1 = isinstance(m, sample_mode)        
        if cond1:
            if abs_val: 
                scores = (Hg[m] * m.weight).abs()
            else: 
                scores = Hg[m] * m.weight
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
    threshold_val = all_params[num_sample]
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
    