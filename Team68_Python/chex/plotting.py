import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import roc_curve, auc, roc_auc_score

from . import config as C


def plot_learning_curves(history, save_name=None, savedir='imgs/'):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(history['train_loss'], label='Training')
    axes[0].plot(history['valid_loss'], label='Validation')
    axes[0].set_title('Loss')

    axes[1].plot(history['train_auc'], label='Training')
    axes[1].plot(history['valid_auc'], label='Validation')
    axes[1].set_title('AUC')
    for ax in axes:
        ax.legend()

    plt.tight_layout()
    plt.show()
    if save_name is not None:
        os.makedirs(savedir, exist_ok=True)
        save_path = os.path.join(savedir, save_name)
        plt.savefig(save_path)

def denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(img, torch.Tensor):
        img_c = img.clone()
        for c, m, s in zip(img_c, mean, std):
            c.mul_(s).add_(m)
    else:
        img_c = img.copy()
        for c, m, s in zip(img_c, mean, std):
            c = (c * s) + m
    return img_c

def show_timg(tensor_img, label=None, denorm=True):
    img = denormalize(tensor_img) if denorm else tensor_img
    img = img.permute(1, 2, 0).detach().to('cpu')

    if label is not None:
        print(label)
    plt.imshow(img)

def show_sample(batch, idx=0, invert=False, denorm=False):
    imgs, targets = batch
    img, target = imgs[idx], targets[idx]
    img = denormalize(img) if denorm else img
    pimg = img.permute(1, 2, 0) if not invert else 1 - img.permute(1, 2, 0)
    print(img.shape)

    plt.imshow(pimg)
    print('Labels:', target)
    return pimg

def show_batch(data_loader, invert=False, denorm=False):
    for images, labels in data_loader:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]);
        ax.set_yticks([])
        images = torch.stack([denormalize(im) for im in images]) if denorm else images
        data = 1 - images if invert else images
        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
        break
    plt.show()


def best_mthresh(y_true, y_pred, classidx=None):
    btx = []
    clsrange = classidx if classidx is not None else range(y_true.shape[1])
    for i in clsrange:
        fpr, tpr, thsh = roc_curve(y_true[:, i], y_pred[:, i])
        bt = thsh[Jstat(tpr, fpr).argmax()]
        btx.append(bt)
        print(i, bt)

    btx = np.array(btx)
    if classidx is not None:
        sco = roc_auc_score(y_true[:, classidx], y_pred[:, classidx] >= btx, multi_class='ovr')
    else:
        sco = roc_auc_score(y_true, y_pred >= btx, average='micro', multi_class='ovr')

    print(sco)
    return btx

def Jstat(tpr, fpr):
    '''https://en.wikipedia.org/wiki/Youden%27s_J_statistic'''
    return tpr + (1 - fpr) - 1

def get_roc_dat(y_test, y_score, class_idx=None, thresh=None):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc, bthrsh = dict(), dict(), dict(), dict()
    n_classes = y_test.shape[1] if class_idx is None else len(class_idx)
    classrange = range(n_classes) if class_idx is None else class_idx
    if isinstance(thresh, float):
        thresh = np.array([thresh] * n_classes)
    for i in classrange:
        fpr[i], tpr[i], thsh = roc_curve(y_test[:, i],
                                         (y_score[:, i] if thresh is None else y_score[:, i] >= thresh[i]))
        ji = Jstat(tpr[i], fpr[i]).argmax()
        bthrsh[i] = (fpr[i][ji], tpr[i][ji], thsh[ji])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in classrange]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in classrange:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc, bthrsh

def plot_roc(y_test, y_score, class_idx=None, thresh=None, title_end=''):
    fpr, tpr, roc_auc, bthrsh = get_roc_dat(y_test, y_score, class_idx, thresh)
    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr["micro"], tpr["micro"], label=f'micro-ROC ({roc_auc["micro"]:0.3f})',
            color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
            label=f'macro-ROC ({roc_auc["macro"]:0.3f})',
            color='navy', linestyle=':', linewidth=4)

    for i in [*fpr.keys()][:-2]:
        ax.plot(fpr[i], tpr[i], lw=2, label=f'{C.TARGET_LABELS[i]} ({roc_auc[i]:0.3f})')
        ax.plot(bthrsh[i][0], bthrsh[i][1], 'or')
        ax.annotate(f'{bthrsh[i][-1]:0.4f}', 
                    (bthrsh[i][0], bthrsh[i][1]), 
                    xytext=(bthrsh[i][0]+0.07, bthrsh[i][1]-0.25),
                    arrowprops=dict(arrowstyle='->',connectionstyle="angle3"))


    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{len(fpr) - 2} Pathologies ROC Curves '+title_end)
    plt.legend()
    plt.show()