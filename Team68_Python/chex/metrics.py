import torch
from sklearn.metrics import roc_auc_score
from . import config as C

@torch.no_grad()
def compute_auc(out,target):
    """Computes multiclass roc_auc_score with micro averaging for all labels and 5 core labels.

    Args:
        out (Tensor): Batch of probabilities output from model.
        target (Tensor): Batch of ground truth labels.

    Returns:
        Tuple(float,float): roc_auc_scores for all labels and core 5 labels, respectively.
    """    
    targ = target.round().detach().to('cpu')
    out = torch.sigmoid(out).detach().to('cpu')
    score = roc_auc_score(targ, out, average='micro',multi_class='ovo')
    score5 = roc_auc_score(targ[:,C.TARGET5_INDEX], out[:,C.TARGET5_INDEX], average='micro',multi_class='ovo')

    return score,score5

@torch.no_grad()
def _compute_auc_dbg(out,target,ninv=0):
    try:
        targ = target.round().detach().to('cpu')
        out = torch.sigmoid(out).detach().to('cpu')
        score = roc_auc_score(targ, out, average='micro',multi_class='ovo')
    except ValueError as e:
        score = 0.5
        ninv+=1
    return score,ninv



class AverageMeter(object):
    """Computes and stores the average and current value
    
    Ref: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        #fmtstr = '{name}  ({avg' + self.fmt + '})'
        fmtstr ='{avg' + self.fmt +'}'
        return fmtstr.format(**self.__dict__)