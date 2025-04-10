import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
# from .pruner_utils import *



def one_hot(targets):
    num_class = targets.max()-targets.min()+1
    targets = F.one_hot(targets%num_class)
    return targets.float()


def SNIP(model, x, y):
    model.train()
    
    grads = {}
    model.zero_grad()
    inputs1 = x
    targets1 = y
    inputs1 = inputs1.cuda()
    targets1 = targets1.cuda()
        
    outputs = model(inputs1)
    loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets1)
    loss.backward()
    grads1 = gather_grads(model)
    update_grads(grads, grads1)
    model.zero_grad()  
    
    
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear): continue
        scores = (grads[m] * m.weight).abs()
        scores = copy.deepcopy(scores.detach())
        
        buffer_keys = [key for key, buffer in m.named_buffers()]
        if 'score' in buffer_keys: 
            m.score = scores
        else: 
            m.register_buffer('score', scores)
            
    
    
def gather_grads(model, sample_mode=nn.Linear):
    grads = {}
    for m in model.modules():
        if isinstance(m, sample_mode):
            grads[m] = m.weight.grad.detach().clone()
    return grads     
    
    
def update_grads(grads, grads1):
    for k, v in grads1.items():
        if k in grads:
            grads[k] += v    
        else: 
            grads[k] = v    