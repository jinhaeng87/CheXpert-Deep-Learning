import random
import numpy as np
import torch

def seed_everything(seed=404):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def recurse_children(mdl, max_depth=3, weighted_only=True, _depthidx=0):
    _depthidx+=1
    for n,c in mdl.named_children():
        wparam = [p for p in c.parameters()]
        nparam = len(wparam)
        
        trncnt = sum(p.requires_grad for p in wparam)
        if not weighted_only or any(hasattr(m,'weight') for m in c.modules()):
            print('  '*_depthidx,'-', n, f'({trncnt if trncnt>0 else "-"}/{nparam})')

        if _depthidx < max_depth:
            recurse_children(getattr(mdl,n),max_depth,weighted_only,_depthidx)


__all__ = ['seed_everything', 'to_device', 'recurse_children']