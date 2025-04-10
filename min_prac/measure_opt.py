import torch

def get_param_norm(model, sample_mode):
    all_params = []
    for m in model.modules(): 
        if isinstance(m, sample_mode): #Conv2d, Linear
            buffer_dict = {k:v for k,v in m.named_buffers()}
            if 'prune_mask' in buffer_dict:
                all_params.append((m.weight*m.prune_mask).view(-1))
            else: 
                all_params.append(m.weight.view(-1))
            all_params.append(m.bias.view(-1))
        else: 
            for p in m.parameters(): #BatchNorm2d
                all_params.append(p.view(-1))
        
    all_params = torch.cat(all_params,0)
    param_norm = all_params.norm().detach().clone()
    return param_norm
    
    
def get_grad_norm(model, sample_mode):
    all_params = []
    for m in model.modules(): 
        if isinstance(m, sample_mode): #Conv2d, Linear
            buffer_dict = {k:v for k,v in m.named_buffers()}
            if 'prune_mask' in buffer_dict:
                all_params.append((m.weight.grad*m.prune_mask).view(-1))
            else: 
                all_params.append(m.weight.grad.view(-1))
            all_params.append(m.bias.grad.view(-1))
        else: 
            for p in m.parameters(): #BatchNorm2d
                all_params.append(p.grad.view(-1))
        
    all_params = torch.cat(all_params,0)
    param_norm = all_params.norm().detach().clone()
    return param_norm    


def cal_diff(x):
    x_n = x[1:]
    x = x[:-1]
    return x_n-x
    
    

