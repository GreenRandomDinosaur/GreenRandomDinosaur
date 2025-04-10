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


def SNIP(model, sample_mode, x, y):
    # inputs, targets = grasp_data(dataloader, num_classes, num_class_samples)
    # batch_size = adjust_batch_size(inputs.size(0), batch_size)
    # num_iters = inputs.size(0)//batch_size
    # update_batchnorm_stats(model, inputs, batch_size)
    # update_batchnorm_stats(model, dataloader)
    # model.eval()
    model.train()
    
    model.zero_grad()
    inputs1 = inputs[i*batch_size:(i+1)*batch_size,:,:,:]
    targets1 = targets[i*batch_size:(i+1)*batch_size]
    inputs1 = inputs1.cuda()
    targets1 = targets1.cuda()
        
    outputs = model(inputs1)
    loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets1)
    grads = autograd.grad(loss, weights, create_graph=False)
            
    return grads
    
    
    