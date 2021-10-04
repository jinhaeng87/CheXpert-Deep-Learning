import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm

from chex import config as C
from chex.dataaug import get_transforms
from chex.dataload import CheXDataset, subset_dloader, get_dloaders
from chex.etl import proc_df, make_df_merged
from chex.inference import validation_ensemble, ensemble, predict_loader
from chex.modeling import CheXModel
from chex.plotting import show_batch, plot_learning_curves
from chex.saving import save_history, save_trained, load_trained, make_desc
from chex.training import Trainer, make_trainer
from chex.utils import seed_everything, recurse_children


def create_parser():
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    parser = argparse.ArgumentParser(description='Train CheXpert DenseNet121 with best defaults')
    parser.add_argument('msavename', help='model savename postfix for saving improvement checkpoints')
    parser.add_argument('psavename', help='prediction savename postfix for saving post-train predictions')
    parser.add_argument('-c','--cuda', action='store_true', help='use cuda for training', dest='USE_CUDA')
    parser.add_argument('-a','--aws', action='store_true', help='training is on AWS', dest='ON_AWS')
    parser.add_argument('-g','--grayscale', action='store_false', help='use grayscale images for training rather than converting to color', dest='color')
    parser.add_argument('-s','--sample', action='store_true', help='use the scala processed sampled dataset for training')
    parser.add_argument('-e','--epochs', default=5, type=int, help='num training epochs')
    parser.add_argument('-w','--workers', default=4, type=int, help='num workers to use for dataloaders', dest='NUM_WORKERS')
    parser.add_argument('-b','--batch-size', default=256, type=int, help='batch size for training and validation')

    return parser
    

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.USE_CUDA:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    DEVICE = torch.device('cuda' if torch.cuda.is_available() and args.USE_CUDA else 'cpu')
    C.set_config(USE_CUDA=args.USE_CUDA, ON_AWS=args.ON_AWS, NUM_WORKERS=args.NUM_WORKERS, DEVICE=DEVICE)

    if args.sample:
        df_master = pd.read_csv(C.DATA_PATH/'Team68_CheXpert_Full.csv').set_index('CheX_Image_ID')
        df_psample = pd.read_csv(C.DATA_PATH/'Team68_Sample_001.csv').rename({'chex_image_id':'CheX_Image_ID'},axis=1).set_index('CheX_Image_ID')
        df_hotest = df_master.loc[df_master['Dataset_ID'] == 2].copy()
        df_hotest = df_hotest.assign(Pop_ID=3, Posit_ID=df_hotest['ap_pa_ll'].map({'AP':1,'PA':2,'LL':3}))
        df_trnvaltst = proc_df(df_psample.join(df_master).append(df_hotest)[[*df_psample.columns]+['Path']+C.TARGET_LABELS])
        df_train, df_valid, df_test = (df_trnvaltst[df_trnvaltst['Pop_ID'].eq(i)] for i in range(1,4))
        #df_train = df_master.loc[df_psample.index,:]
        #df_valid = df_master[df_master.Dataset_ID == 2]
    else:
        df_train = pd.read_csv(C.CHX_PATH/'train.csv')
        df_valid = pd.read_csv(C.CHX_PATH/'valid.csv')

    train_loader, valid_loader = get_dloaders(df_train, df_valid, args.batch_size, None, color=args.color)
    nsamp = len(train_loader.dataset)//1000#str(len(train_loader.dataset)/1000).split('.')[0]
    savename=f'dn121_TSlr{args.batch_size}bSig_{nsamp}k_script_'+args.msavename

    criterion = nn.BCELoss()
    model = CheXModel('densenet121', use_sig=True, color=args.color)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    thaw_schd = {2:('norm5','denseblock4'), 3:('transition3','denseblock3'),}
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1,5], gamma=0.1, verbose=True)

    trainer = Trainer(model, optimizer, criterion, scheduler, thaw_schd, device=DEVICE)
    pstat = trainer.freeze(['_fc','fc','network.classifier','classifier'], invert_match=True)

    hist = trainer.train(train_loader, valid_loader, args.epochs, log_freq=10, save_name=f'{savename}_chkpnt')
    save_desc = make_desc(train_loader, trainer, f"{args.epochs}e, run from script file")
    save_history(trainer.history, savename, save_desc)

    df_preds = validation_ensemble(trainer.model, proc_df(df_valid), color=args.color, device=DEVICE)
    df_preds.to_csv(C.SAVE_PATH/f'preds/script_preds_{"sample" if args.sample else "full"}_{args.psavename}.csv')

if __name__ == '__main__':
    main()

