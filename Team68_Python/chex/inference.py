import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
#from tqdm.notebook import tqdm
from tqdm.autonotebook import tqdm

from . import config as C
from .dataload import subset_dloader
from .plotting import show_timg
from .saving import load_trained
from .training import make_trainer
from .utils import to_device


@torch.no_grad()
def predict_one(model, img_tensor, label):
    was_training = model.training
    model.eval()
    logits = model(img_tensor.unsqueeze(0).to(C.DEVICE))
    pred = torch.sigmoid(logits.squeeze()).detach().to('cpu').numpy()

    label = label.to('cpu').numpy()
    show_timg(img_tensor, label, denorm=True)
    thresh_pred = (pred > 0.5).astype(float)
    print(thresh_pred)
    print(pred.round(5))

    micro_auc = roc_auc_score(label, pred, average='micro', multi_class='ovo')
    macro_auc = roc_auc_score(label, pred, average='macro', multi_class='ovo')
    micro_auc_thresh = roc_auc_score(label, thresh_pred, average='micro', multi_class='ovo')
    macro_auc_thresh = roc_auc_score(label, thresh_pred, average='macro', multi_class='ovo')
    print(f'micro_auc: {micro_auc:.5f}, macro_auc: {macro_auc:.5f}')
    print(f'micro_auc_thresh: {micro_auc_thresh:.5f}, macro_auc_thresh: {macro_auc_thresh:.5f}')
    model.train(was_training)


@torch.no_grad()
def predict_loader(model, data_loader, sigmoid=False, device=None):
    """Generate predictions for all items in a DataLoader.

    Args:
        model (nn.Module): Trained model used to make predictions.
        data_loader (DataLoader): DataLoader containing data and (ignored) labels.
        sigmoid (bool, optional): If True, model outputs are passed through a sigmoid function
            else raw output is used. Defaults to False.
        device (torch.device, optional): device to perform computations on. Defaults to config option `DEVICE`.

    Returns:
        np.array: Stacked predictions for all data items shape (n_sample,n_classes)
    """
    device = device if device is not None else C.DEVICE
    was_training = model.training
    model.eval()
    preds = []
    for batch in tqdm(data_loader, leave=False):
        data, target = to_device(batch, device)
        logits = model(data)
        pred = torch.sigmoid(logits) if sigmoid else logits
        preds.append(pred.detach())
    model.train(was_training)
    return torch.vstack(preds).to('cpu').numpy()

def load_predict(base_model, checkpoint, loader, sigmoid=False, device=None):
    """Load a saved model checkpoint and predict on a dataloader

    Args:
        base_model (nn.Module): Shell model of the same archtecture as saved checkpoint models
        checkpoints (List(str)): List of checkpoint filenames to be loaded into the ensemble
        loader (DataLoader): DataLoader to generate ensembled predictions.
        sigmoid (bool, optional): If True, model outputs are passed through a sigmoid function
            else raw output is used. Defaults to False.
        device (torch.device, optional): device to perform computations on. Defaults to config option `DEVICE`.

    Returns:
        pd.DataFrame: DataFrame containing averaged prediction labels.
    """
    device = device if device is not None else C.DEVICE
    model = load_trained(base_model, checkpoint, device=device)
    preds = predict_loader(base_model, loader, sigmoid, device=device)

    lab_preds = [*zip(loader.dataset.df.index, preds)]
    df_ens = pd.DataFrame(
        [{'CheX_Image_ID': cid, **{k: v for k, v in zip(C.TARGET_LABELS, preds)}} for cid, preds in lab_preds]).set_index('CheX_Image_ID')
    
    return df_ens

def _full_freeze(model):
    for child in model.children():
        _full_freeze(child)
    if hasattr(model, 'requires_grad'):
        model.requires_grad_(False)


def ensemble(base_model, checkpoints, loaders, sigmoid=False, device=None):
    """Creates a weighted average prediction ensemble from saved model checkpoints

    Args:
        base_model (nn.Module): Shell model of the same archtecture as saved checkpoint models
        checkpoints (List(str)): List of checkpoint filenames to be loaded into the ensemble
        loaders (DataLoader): DataLoaders to generate ensembled predictions.
        sigmoid (bool, optional): If True, model outputs are passed through a sigmoid function
            else raw output is used. Defaults to False.
        device (torch.device, optional): device to perform computations on. Defaults to config option `DEVICE`.

    Returns:
        pd.DataFrame: DataFrame containing averaged prediction labels.
    """
    device = device if device is not None else C.DEVICE
    ens_preds = []
    for mdlchk, loader in zip(checkpoints, loaders):
        model = load_trained(base_model, mdlchk, device=device)
        preds = predict_loader(model, loader, sigmoid, device)
        ens_preds.append(preds)

    lab_preds = [*zip(loader.dataset.df.index, np.mean(ens_preds, 0))]
    df_ens = pd.DataFrame(
        [{'CheX_Image_ID': cid, **{k: v for k, v in zip(C.TARGET_LABELS, preds)}} for cid, preds in lab_preds]).set_index('CheX_Image_ID')
    
    return df_ens,loader.dataset.df[C.TARGET_LABELS]


def pseudo_ensemble(base_model, checkpoints, loaders, sigmoid=False, device=None):
    """Creates a prediction ensemble in which models trained on a specific subgroup of
    data are used to predict that subgroup of the validation set.

    Args:
        base_model (nn.Module): Shell model of the same archtecture as saved checkpoint models
        checkpoints (List(str)): List of checkpoint filenames to be loaded into the ensemble
        loaders (DataLoader): DataLoaders to generate, partioned into groups fit for each models' use.
        sigmoid (bool, optional): If True, model outputs are passed through a sigmoid function
            else raw output is used. Defaults to False.
        device (torch.device, optional): device to perform computations on. Defaults to config option `DEVICE`.

    Returns:
        pd.DataFrame: DataFrame containing averaged prediction labels.
    """
    device = device if device is not None else C.DEVICE
    psens_preds = []
    loadr_labs = []
    for mdlchk, loader in zip(checkpoints, loaders):
        model = load_trained(base_model, mdlchk, device=device)
        preds = predict_loader(model, loader, sigmoid, device)
        psens_preds.extend([*zip(loader.dataset.df.index, preds)])
        loadr_labs.extend([*zip(loader.dataset.df.index, loader.dataset.labels)])
    ps_df = pd.DataFrame(
        [{'CheX_Image_ID': cid, **{k: v for k, v in zip(C.TARGET_LABELS, pred)}} for cid, pred in psens_preds]).set_index('CheX_Image_ID')
    lbs_df = pd.DataFrame(
        [{'CheX_Image_ID': cid, **{k: v for k, v in zip(C.TARGET_LABELS, lbl)}} for cid, lbl in loadr_labs]).set_index('CheX_Image_ID')
    
    return ps_df,lbs_df


def validation_ensemble(model, df_pvalid, batch_size=256, n_ens=10, sigmoid=False, color=True, device=None):
    """Generate a weighted average ensemble of the validation set with TTA enabled

    Args:
        model (nn.Module): Trained model to make predictions
        df_pvalid (pd.DataFrame): Processed validation DataFrame
        batch_size (int, optional): Size of validation batches. Defaults to 256.
        n_ens (int, optional): Number of predictions to generate with the full validation dataset. Defaults to 10.
        sigmoid (bool, optional): If True, model outputs are passed through a sigmoid function
            else raw output is used. Defaults to False.
        device (torch.device, optional): device to perform computations on. Defaults to config option `DEVICE`.

    Returns:
        pd.DataFrame: Averaged ensemble prediction, shape (n_sample, n_class)
    """
    device = device if device is not None else C.DEVICE
    val_loader = subset_dloader(df_pvalid, is_train=False, batch_size=batch_size, color=color, print_tfms=False)
    preds = [predict_loader(model, val_loader, sigmoid=False, device=device) for _ in range(n_ens)]
    pred_ens = np.mean(preds, 0)
    df_ens = pd.DataFrame(pred_ens, index=df_pvalid.index, columns=C.TARGET_LABELS)
    return df_ens