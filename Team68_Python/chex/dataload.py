import random

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .dataaug import get_transforms
from .etl import proc_df
from .utils import seed_everything


# (blank for unmentioned, 0 for negative, -1 for uncertain, and 1 for positive)
class CheXDataset(Dataset):
    """Custom dataset for CheXpert dataset.

    Args:
        df (pandas.DataFrame): preprocessed dataframe that has at least a "Path" column and 14 coulmns containing target labels.
        use_albu (bool, optional): If true, use albumentations for image augmentations else torchvision. Defaults to True.
        tfms (albumentations.Compose,torchvision.Compose, optional): Series of image augmentations to apply. Defaults to None.
        seed (int, optional): random seed used for deterministic augmentations. Defaults to None.
    """

    def __init__(self, df, use_albu=True, tfms=None, color=True, seed=None):
        self.df = df
        self.paths = self.df['Path'].values
        self.labels = self.df[C.TARGET_LABELS].values.astype(float)#.drop(columns=['Path'])
        self.tfms = tfms

        self.use_albu = use_albu
        self.color = color
        self._seedcntr = seed
        self._readflag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED 
        if self.tfms is not None:
            self.tfm_list = self.tfms.transforms.transforms if self.use_albu else self.tfms.transforms

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        if self._seedcntr is not None:
            random.seed(self._seedcntr)
            self._seedcntr += 1
        labels = self.labels[idx]

        imgpath = str(C.DATA_PATH / self.paths[idx])

        is_lateral = 'lateral' in imgpath
        if self.use_albu:
            img = cv2.imread(imgpath, self._readflag)
            # img = template_match(img, template)
            # img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
            # img = np.expand_dims(cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE),2)
            aug = self.tfms(image=img, is_lateral=is_lateral)
            img = aug['image'] if self.color else torch.from_numpy(aug['image'][:,:,None].transpose(2,0,1))
            #img = 
        else:
            img = Image.open(imgpath)
            img = self.tfms(img)

        return img, labels


def get_dloaders(df_train, df_valid, batch_size=32, sampsz=None, tfmlib='albu', color=True, seed=None, print_tfms=True,
                 proc_kwargs=None):
    """Creates DataLoader objects from training and validation DataFrames.

    Args:
        df_train (DataFrame): Training DataFrame containing Path and Label columns
        df_valid (DataFrame): Validation DataFrame containing Path and Label columns
        batch_size (int, optional): Training/Validation batch size passed to DataLoaders. Defaults to 32.
        sampsz (int, optional): If provided, uses a sample of `df_train` else uses entire dataset. Defaults to None.
        tfmlib (str, optional): Image augmentation library to use. Options: ('albu','torch'). Defaults to 'albu'.
        seed ([type], optional): random seed for sampling, shuffling and augmentations. Defaults to None.
        print_tfms (bool, optional): If True, prints list of training image augmentations. Defaults to True.
        proc_kwargs (dict, optional): Keyword arguments passed to `proc_df` for preprocessing `df_train`. Defaults to None.

    Returns:
        tuple(DataLoader,DataLoader): training and validation DataLoaders
    """
    if seed is not None:
        seed_everything(seed)

    if proc_kwargs is None:
        proc_kwargs = dict(method='uones', smooth=True, nafill_val=0)

    df_trn = proc_df(df_train, **proc_kwargs)
    df_val = proc_df(df_valid)
    if sampsz is not None:
        df_trn = df_trn.sample(sampsz)  # 17->65k, 13->8k

    ualbu = (tfmlib == 'albu')
    train_tfm = get_transforms('train', tfmlib, (244, 244), color=color)
    valid_tfm = get_transforms('test', tfmlib, (244, 244), color=color)

    train_dataset = CheXDataset(df=df_trn, use_albu=ualbu, tfms=train_tfm, color=color, seed=seed)  # smooth_bounds=(0.55,0.8501)
    valid_dataset = CheXDataset(df=df_val, use_albu=ualbu, tfms=valid_tfm, color=color)

    train_loader = DataLoader(train_dataset, batch_size, pin_memory=C.USE_CUDA, num_workers=C.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size, pin_memory=C.USE_CUDA, num_workers=C.NUM_WORKERS,
                              shuffle=False)

    if print_tfms:
        print(f'{"=" * 20} Augmentations {"=" * 20}', '\n'.join([*map(str, train_loader.dataset.tfm_list)]), '-' * 55,
              sep='\n')

    return train_loader, valid_loader


def get_ctdloaders(df_ptrain, df_pvalid, batch_size=32, tfmlib='albu', seed=None, print_tfms=True):
    ualbu = (tfmlib == 'albu')
    train_tfm = get_transforms('train', tfmlib, (244, 244))
    valid_tfm = get_transforms('test', tfmlib, (244, 244))

    train_dataset = CheXDataset(df=df_ptrain, use_albu=ualbu, tfms=train_tfm, seed=seed)  # smooth_bounds=(0.55,0.8501)
    valid_dataset = CheXDataset(df=df_pvalid, use_albu=ualbu, tfms=valid_tfm)

    train_loader = DataLoader(train_dataset, batch_size, pin_memory=C.USE_CUDA, num_workers=C.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size, pin_memory=C.USE_CUDA, num_workers=C.NUM_WORKERS,
                              shuffle=False)
    if print_tfms:
        print(f'{"=" * 20} Augmentations {"=" * 20}', '\n'.join([*map(str, train_loader.dataset.tfm_list)]), '-' * 55,
              sep='\n')

    return train_loader, valid_loader


def subset_dloader(df, grp_id=None, is_train=False, grp_name='ap_pa_ll', batch_size=32, seed=None, tfmlib='albu', color=True,
                   print_tfms=True):
    if seed is not None:
        seed_everything(seed)

    set_tfms = get_transforms('train' if is_train else 'test', tfmlib, (244, 244), color=color)

    df_subset = df[df[grp_name].eq(grp_id)] if grp_id is not None else df

    sub_dataset = CheXDataset(df=df_subset, use_albu=(tfmlib == 'albu'), tfms=set_tfms, color=color, seed=seed)
    sub_loader = DataLoader(sub_dataset, batch_size, pin_memory=C.USE_CUDA, num_workers=C.NUM_WORKERS, shuffle=is_train)

    if print_tfms:
        print(f'{"=" * 20} Augmentations {"=" * 20}', '\n'.join([*map(str, sub_loader.dataset.tfm_list)]), '-' * 55,
              sep='\n')

    return sub_loader


def get_pdloaders(train_df, valid_df, grp_id, grp_name='ap_pa_ll', batch_size=32, seed=None, tfmlib='albu',
                  print_tfms=True, ):
    if seed is not None:
        seed_everything(seed)

    ualbu = (tfmlib == 'albu')

    train_tfm = get_transforms('train', tfmlib, (244, 244))
    valid_tfm = get_transforms('test', tfmlib, (244, 244))

    df_trn = train_df[train_df[grp_name].eq(grp_id)]
    df_val = valid_df[valid_df[grp_name].eq(grp_id)]
    train_dataset = CheXDataset(df=df_trn, use_albu=ualbu, tfms=train_tfm, seed=seed)
    valid_dataset = CheXDataset(df=df_val, use_albu=ualbu, tfms=valid_tfm, seed=seed)

    train_loader = DataLoader(train_dataset, batch_size, pin_memory=C.USE_CUDA, num_workers=C.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size, pin_memory=C.USE_CUDA, num_workers=C.NUM_WORKERS,
                              shuffle=False)

    if print_tfms:
        print(f'{"=" * 20} Augmentations {"=" * 20}', '\n'.join([*map(str, train_loader.dataset.tfm_list)]), '-' * 55,
              sep='\n')

    return train_loader, valid_loader