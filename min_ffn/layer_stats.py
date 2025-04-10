import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


def semi_stats(model, x, y):
    grad_norm = get_grad(model, x, y)
    prod_sigma = cal_sigma(model)
    prod_norms, norm_vec = cal_norm(model)
    dims = dim_info(model)
    common_var = cal_var()

    stats = {}
    cnt = 0
    for n, m in model.named_modules(): 
        if not isinstance(m, nn.Linear): continue
        val1 = x.norm() * grad_norm * prod_sigma
        val1 *= common_var
        val1 *= prod_norms
        
        if cnt == 0: # Only the intermediate layers are considered. This is a dummy result.
            norm_curr = norm_vec[cnt]
            norm_prev = 1 
        else:
            norm_curr = norm_vec[cnt]
            norm_prev = norm_vec[cnt-1]            
        cnt += 1
        
        val1 /= norm_curr
        val1 /= norm_prev
        
        stats[n]= val1.item()
    return stats
    
    
def cal_var():
    b = torch.bernoulli(torch.ones(1)*0.5)
    u1 = torch.normal(0., 1., size=[1]).abs()
    v = torch.normal(0., 1., size=[1]).abs()
    u2 = torch.normal(0., 1., size=[1])
    if u2 < 0: u2 = torch.zeros(1)
    prod1 = b*u1*v*u2
    
    return prod1.item()
    
    
    
def dim_info(model):
    dim1 = 28*28
    dim_dict = {}
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear): continue
        dim_dict[n] = [dim1, m.weight.size(0)]
        dim1 = m.weight.size(0)
    return dim_dict
    
    
def cal_norm(model):
    norm_vec = []
    for m in model.modules(): 
        if not isinstance(m, nn.Linear): continue
        dim1 = m.weight.size(0)
        norm1 = layer_norm(dim1)
        norm_vec.append(norm1)
    
    prod_norms = 1.
    for i in range(len(norm_vec)-1):
        prod_norms *= norm_vec[i]
        
    return prod_norms, norm_vec

    
def layer_norm(dim1):
    normal_vec = torch.normal(0., 1., size=[dim1])
    
    ber_vec = torch.ones(dim1)*0.5
    ber_vec = torch.bernoulli(ber_vec)
    
    vec_norm = (normal_vec*ber_vec).norm().item()
    return vec_norm

    


def cal_sigma(model):
    sigma1 = 1.
    for m in model.modules(): 
        if not isinstance(m, nn.Linear): continue
        dim1 = m.weight.size(0)
        sigma1 *= np.sqrt(2/dim1)
    return sigma1
    
    
def get_grad(model, x, y):
    model.train()
    
    model.zero_grad()
    inputs1 = x.cuda()
    targets1 = y.cuda()
        
    outputs = model(inputs1)
    loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets1)
    grads = autograd.grad(loss, outputs, create_graph=False)

    return grads[0].norm().item()