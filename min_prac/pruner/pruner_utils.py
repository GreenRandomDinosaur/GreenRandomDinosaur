import torch
import torch.nn as nn
import torch.nn.functional as F
# from pruner_utils import *

def adjust_batch_size(num_data, batch_size):
    bs = batch_size
    if num_data < batch_size: 
        bs = num_data
    elif num_data > batch_size:
        if num_data % batch_size > 0:
            bs = batch_size
            while num_data % bs > 0:
                bs = bs -1
    return bs


def grasp_data(dataloader, n_classes, n_samples=10):
    datas = [[] for _ in range(n_classes)]
    labels = [[] for _ in range(n_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == n_samples:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == n_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y
    
    
def gather_weights(model, mode):
    weights = []
    for m in model.modules():
        cond1 = isinstance(m, mode)
        
        if cond1:
            m.weight.requires_grad_(True)
            weights.append(m.weight)
    return weights
    
    
def gather_grads(model, mode):
    grads = {}
    for m in model.modules():
        cond1 = isinstance(m, mode)
        
        if cond1:
            # grads.append(m.weight.grad.detach().clone())
            grads[m] = m.weight.grad.detach().clone()
    return grads 


def update_grads(grads, grads1):
    for k, v in grads1.items():
        if k in grads:
            grads[k] += v    
        else: 
            grads[k] = v    
            
# def update_batchnorm_stats(model, inputs, batch_size):
def update_batchnorm_stats(model, dataloader):
    model.train()
    '''
    batch_size = adjust_batch_size(inputs.size(0), batch_size)
    num_iters = inputs.size(0)//batch_size#+1
    for i in range(num_iters):        
        if i==num_iters-1:
            inputs1 = inputs[i*batch_size:,:,:,:]
        else: 
            inputs1 = inputs[i*batch_size:(i+1)*batch_size,:,:,:]
        inputs1 = inputs1.cuda()
        outputs = model(inputs1)
    '''
    # for m in model.modules():
        # if isinstance(m, nn.BatchNorm2d):
            # print(m.running_mean[:10].tolist())
            # print(m.running_var[:10].tolist())
    
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()        
        outputs = model(inputs)
        

    # for m in model.modules():
        # if isinstance(m, nn.BatchNorm2d):
            # print(m.running_mean[:10].tolist())
            # print(m.running_var[:10].tolist())


def reset_batchnorm_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.fill_(0.)
            m.running_var.fill_(1.)