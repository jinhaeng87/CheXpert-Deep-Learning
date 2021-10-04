import time
import pandas as pd
# from tqdm.notebook import tqdm
from tqdm.autonotebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from .modeling import CheXModel
from .saving import save_trained
from .metrics import compute_auc, AverageMeter
from . import config as C


class TrainerBase:
    """Base class for Trainer class"""

    def __init__(self):
        self.history = {}

    def freeze(self, param_names=None, invert_match=True, unfreeze=False):
        """Freezes/Unfreezes layers by name in self.model 

        Args:
            param_names (List(str), optional): Iterable containing layers of model architecture to mutate. 
                If none provided, prints current Freeze status of layers. Defaults to None.
            invert_match (bool, optional): If True, matches ALL layers that do NOT match `params_names`.
                If False, matches `params_names` items exactly. Defaults to True.
            unfreeze (bool, optional): If True, makes matching layers trainable, if False, untrainable. Defaults to False.

        Returns:
            [type]: [description]
        """
        candidate_modules = [(n, m) for n, m in self.model.named_modules() if hasattr(m, 'weight')]
        n_params = len(candidate_modules)

        child_names, children = zip(*[*self.model.named_children()])
        #n_child = len(children)

        if param_names is None:
            child_train = [[x.requires_grad for x in child.parameters()] for child in self.model.children()]
            lay_df = pd.DataFrame({'Name': child_names, 'Trainable': [f'{sum(c)}/{len(c)}' for c in child_train]})
            print(lay_df)

            # print('Frozen Parameters: ({} / {})'.format((~init_trainables).sum(),n_params))
            # print('Trainable Parameters: ({} / {})'.format(init_trainables.sum(),n_params))
            return

        for name, module in candidate_modules:
            if any(pn in name for pn in param_names):
                if not invert_match:
                    module.requires_grad_(unfreeze)
            elif invert_match:
                module.requires_grad_(unfreeze)

        params_status = {'trainable': [], 'frozen': []}
        for n, m in candidate_modules:
            params_status['trainable' if any(p.requires_grad for p in m.parameters()) else 'frozen'].append(n)

        print(f'Trainable: {len(params_status["trainable"])}, Frozen: {len(params_status["frozen"])}')
        return params_status

    def update_history(self, **kwargs):
        for k, v in kwargs.items():
            self.history.setdefault(k, []).append(v)

    def to_device(self, data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    def save_improved(self, score, best_score, save_name=None, save_path='models/'):
        if score > best_score:
            print(f'Score improved: {score:.5f} > {best_score:.5f}')
            best_score = score
            if save_name is not None:
                save_trained(self.model, save_name, save_path=save_path)

        return best_score

    def train_batch_end(self, i, log_freq, **kwargs):
        if log_freq is not None and i % log_freq == 0:
            self.update_history(**kwargs)

    def train_epoch_end(self, **kwargs):
        self.update_history(**kwargs)  # {'train_loss':loss,'train_auc':auc})

    def validation_epoch_end(self, **kwargs):
        self.update_history(**kwargs)  # {'valid_loss':loss, 'valid_auc':auc}

    def epoch_end(self, epoch, exclude_keys=None):
        if exclude_keys is None:
            exclude_keys = ()
        hist_str = f'Epoch [{epoch}] ' + ', '.join(
            [f'{k}: {v[-1]:.4f}' for k, v in self.history.items() if k not in exclude_keys])
        print(hist_str)

class Trainer(TrainerBase):
    """Trainer for all things training and evaluatings models

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optim): Optimizer linked to `model` 
        criterion (nn.Module): Loss function for model training
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. 
            Defaults to None.
        thaw_schedule (dict, optional): Dict containing epochs and layer names 
            (e.g. {1:('denseblock1',),}. When provided, layers are unfrozen once the 
            training epoch reaches one of the keys. Defaults to None.
        device (torch.device, optional): device on which to load and train models and data. 
            Defaults to config option `DEVICE`.
    """
    def __init__(self, model, optimizer, criterion, scheduler=None, thaw_schedule=None, device=None):

        super(Trainer, self).__init__()
        self.device = device if device is not None else C.DEVICE
        self.model = model.to(self.device)#.half()
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)#.half()
        self.scheduler = scheduler
        self.thaw_schedule = thaw_schedule if thaw_schedule is not None else {}

    def train(self, train_loader, valid_loader, n_epochs=1, log_freq=10, save_name=None, accum_steps=None):
        """Trains self.model, printing progress statistics

        Args:
            train_loader (DataLoader): DataLoader containing training Dataset
            valid_loader (DataLoader):  DataLoader containing validation Dataset
            n_epochs (int, optional): Number of epochs of training to conduct. Defaults to 1.
            log_freq (int, optional): Number of batch iterations before adding training statistics to history. Defaults to 10.
            save_name (str, optional): filename for saving model. If provided, whenever the validation AUC improves, the trained
                layers will be saved under this name using the `save_trained` function. Defaults to None.

        Returns:
            dict: training history, also available as a class attribute
        """
        best_val_auc = max(self.history.get('valid_auc',[0.0]))
        train_func = self.train_one if accum_steps is None else self.train_one_accum
        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            if epoch in self.thaw_schedule:
                self.freeze(self.thaw_schedule[epoch], invert_match=False, unfreeze=True)
            
            t0 = time.perf_counter()
            train_loss, train_auc, train_auc5 = train_func(train_loader, pbar, log_freq, accum_steps)
            self.train_epoch_end(train_loss=train_loss, train_auc=train_auc, train_auc5=train_auc5, train_time=time.perf_counter()-t0)
            
            v0 = time.perf_counter()
            valid_loss, valid_auc, valid_auc5 = self.evaluate(valid_loader, pbar)
            self.validation_epoch_end(valid_loss=valid_loss, valid_auc=valid_auc, valid_auc5=valid_auc5, valid_time=time.perf_counter()-v0)

            best_val_auc = self.save_improved(valid_auc, best_val_auc, save_name=save_name)

            if self.scheduler is not None:
                self.scheduler.step()

            self.epoch_end(epoch, exclude_keys=['intraepoch_tloss', 'intraepoch_tauc'])

        return self.history

    def train_one(self, data_loader, pbar, log_freq=None, accum_steps=None):
        self.model.train()
        #tloss, tauc, tauc5 = 0, 0, 0
        tloss = AverageMeter('TLoss',':0.4f')
        tauc = AverageMeter('TAUC',':0.4f')
        tauc5 = AverageMeter('TAUC5',':0.4f')
        nbat = max(len(data_loader),1)
        for i, batch in enumerate(tqdm(data_loader, leave=False)):
            data, target = self.to_device(batch, self.device)

            # Side effects: https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer.zero_grad
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(data)#.half()
            loss = self.criterion(output, target.float())#.float())  # Remove float after testing

            loss.backward()
            self.optimizer.step()

            #tloss += loss.detach().item()
            tloss.update(loss.detach().item())
            ta, ta5 = compute_auc(output, target)
            tauc.update(ta)
            tauc5.update(ta5)
            #tauc += ta
            #tauc5 += ta5

            #itloss, itauc, itauc5 = tloss / (i + 1), tauc / (i + 1), tauc5 / (i + 1)
            #pbar.set_postfix({'TLoss': f'{itloss:.4f}', 'TAUC': f'{itauc:.4f} ({itauc5:.4f})'})
            pbar.set_postfix({'TLoss': tloss, 'TAUC': f'{tauc} ({tauc5})'})
            self.train_batch_end(i, log_freq, intraepoch_tloss=tloss.avg, intraepoch_tauc=tauc.avg)
            # tloss.avg / nbat, tauc.avg / nbat, tauc5.avg / nbat
        return tloss.avg, tauc.avg, tauc5.avg

    def train_one_accum(self, data_loader, pbar, log_freq=None, accum_steps=None):
        self.model.train()
        #tloss, tauc, tauc5 = 0, 0, 0
        tloss = AverageMeter('TLoss',':0.4f')
        tauc = AverageMeter('TAUC',':0.4f')
        tauc5 = AverageMeter('TAUC5',':0.4f')
        nbat = max(len(data_loader),1)
        self.optimizer.zero_grad()
        for i, batch in enumerate(tqdm(data_loader, leave=False)):
            data, target = self.to_device(batch, self.device)

            output = self.model(data)
            loss = self.criterion(output, target.float())/accum_steps
            loss.backward()
            tloss.update(loss.detach().item())
            if (i+1) % accum_steps ==0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            #tloss += loss.detach().item()
            
            ta, ta5 = compute_auc(output, target)
            tauc.update(ta)
            tauc5.update(ta5)

            pbar.set_postfix({'TLoss': tloss, 'TAUC': f'{tauc} ({tauc5})'})
            self.train_batch_end(i, log_freq, intraepoch_tloss=tloss.avg, intraepoch_tauc=tauc.avg)

            # itloss, itauc, itauc5 = tloss / (i + 1), tauc / (i + 1), tauc5 / (i + 1)
            # pbar.set_postfix({'TLoss': f'{itloss:.4f}', 'TAUC': f'{itauc:.4f} ({itauc5:.4f})'})
            # self.train_batch_end(i, log_freq, intraepoch_tloss=itloss, intraepoch_tauc=itauc)

        return tloss.avg, tauc.avg, tauc5.avg

    @torch.no_grad()
    def evaluate(self, data_loader, pbar=None):
        self.model.eval()
        nbat = max(len(data_loader),1)
        #vloss, vauc, vauc5 = 0, 0, 0
        vloss = AverageMeter('VLoss',':0.4f')
        vauc = AverageMeter('VAUC',':0.4f')
        vauc5 = AverageMeter('VAUC5',':0.4f')
        for i, batch in enumerate(tqdm(data_loader, leave=False)):
            data, target = self.to_device(batch, self.device)

            output = self.model(data)
            loss = self.criterion(output, target.float())
            
            vloss.update(loss.detach().item())
            va, va5 = compute_auc(output, target)
            vauc.update(va)
            vauc5.update(va5)
            
            #vloss += loss.detach().item()
            #vauc += va
            #vauc5 += va5
            if pbar is not None:
                #ivloss, ivauc, ivauc5 = vloss/(i+1), vauc/(i+1), vauc5/(i+1)
                pbar.set_postfix({'VLoss': vloss, 'VAUC': f'{vauc} ({vauc5})'})

        return vloss.avg, vauc.avg, vauc5.avg #vloss/nbat, vauc/nbat, vauc5/nbat



def make_trainer(arch='densenet121', lr=1e-3, scheduler=None, thaw_schedule=None):
    """Create a Trainer object with senseable defaults.

    Internal Defaults:
        optimizer: Adam
        criterion: BCEWithLogitsLoss
        layer freezing applied to all but final fully connected layer.

    Args:
        arch (str, optional): Base archtecture for pretrained model. Defaults to 'densenet121'.
        lr (float, optional): Learning rate for Adam optimizer. Defaults to 1e-3.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        thaw_schedule (dict, optional): Dict containing epochs and layer names (e.g. {1:'denseblock1',}.
            When provided, layers are unfrozen once the training epoch reaches one of the keys. Defaults to None.

    Returns:
        Trainer: Initialized Trainer object
    """
    criterion = nn.BCEWithLogitsLoss()
    model = CheXModel(arch)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduler == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1, verbose=True)
    trainer = Trainer(model, optimizer, criterion, scheduler, thaw_schedule)
    pstat = trainer.freeze(['_fc', 'fc', 'network.classifier', 'classifier'], invert_match=True)

    return trainer