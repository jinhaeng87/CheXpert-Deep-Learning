from pathlib import Path
from collections import OrderedDict
import pickle
import torch

from . import config as C

def make_desc(train_loader,trainer,desc):
    """Append basic training statistics to description string

    Args:
        train_loader (DataLoader): DataLoader for training set
        trainer (Trainer): Trainer object containing trained model and history statistics
        desc (str): Description prefix, text added before basic summary stats in the text file

    Returns:
        str: Concatenated description string, ready for file write
    """
    save_desc = desc+", samples {}, batch {}, lr {}, top val AUC {:0.5f}".format(
    len(train_loader.dataset), train_loader.batch_size, 
    trainer.optimizer.param_groups[0]['lr'], max(trainer.history['valid_auc']))
    return save_desc

def save_history(history, save_name, description, save_path='histories/'):
    """Saves model training history to a pickle file. 
    
    In determining full save location, the global config option `SAVE_PATH` 
    will be prepended to `save_path` param, making for full destination path
    `SAVE_PATH`/`save_path`/`save_name`.pkl 

    Args:
        history (dict): model training history object, provided by Trainer interface
        save_name (str): file name for pickle object, if not ending in .pkl, .pkl will be added
        description (str): Description of training procedure/hyperparameters, 
            will be written to `SAVE_PATH`/`save_path`/description.txt along with filename
        save_path (str, optional): Directory for files to be saved. Defaults to 'histories/'.
    """
    full_path = Path(C.SAVE_PATH/save_path)
    full_path.mkdir(parents=True, exist_ok=True)
    desc = f'{save_name} - {description} \n'
    with full_path.joinpath('description.txt').open('a+') as f:
        f.write(desc)
    dump_path = full_path.joinpath(save_name).with_suffix('.pkl')    
    pickle.dump(history,dump_path.open('wb'))
    print('File saved to:',str(dump_path))

def save_trained(model, save_name, description=None, module='network', save_path='models/'):
    """Save a trained model statedict, excluding all frozen layers.

    In determining full save location, the global config option `SAVE_PATH` 
    will be prepended to `save_path` param, making for full destination path
    `SAVE_PATH`/`save_path`/`save_name`.pt 

    Args:
        model (nn.Module): Trained pytorch model object
        save_name (str): save filename, if not ending in .pt, .pt will be added
        description (str, optional): Description of model training procedure/hyperparamters. 
            If provided, will be written to `SAVE_PATH`/`save_path`/description.txt along with filename. Defaults to None.
        module (str, optional): submodule of network, by default CheXmodel assigns to 'network'. Defaults to 'network'.
        save_path (str, optional): Directory for files to be saved. Defaults to 'models/'.
    """
    full_path = Path(C.SAVE_PATH/save_path)
    full_path.mkdir(parents=True, exist_ok=True)
    modelmod = getattr(model,module,model)
    states = OrderedDict(
        {n: c.state_dict() for n,c in modelmod.named_children() if any(p.requires_grad for p in c.parameters())}
    )
    if description is not None:
        desc = f"{save_name} - ({', '.join(states.keys())}) : {description} \n"
        with full_path.joinpath('description.txt').open('a+') as f:
            f.write(desc)
    
    out_path = full_path.joinpath(save_name).with_suffix('.pt')
    torch.save(states,out_path)
    
    print('state dict saved to:',out_path.as_posix())
    

def load_trained(model, file_name, module='network', load_path='models/', device=None):
    """Load the trained statedict of a model saved by `save_trained` function.

    In determining full load location, the global config option `SAVE_PATH` 
    will be prepended to `load_path` param, making for full destination path
    `SAVE_PATH`/`load_path`/`file_name`.pt 

    Args:
        model (nn.Module): Untrained shell model that shares the same state_dict structure with the loading object. 
        file_name (str): file name of saved state_dict object, if not ending in .pt, .pt will be added
        module (str, optional): submodule of network, by default CheXmodel assigns to 'network'. Defaults to 'network'.
        load_path (str, optional): Directory from which saved file will be loaded. Defaults to 'models/'.
        device (torch.device, optional): Device to load model on. Defaults to None.

    Example:
        model = CheXModel('densenet121')
        model = load_trained(model,'densenet121.pt')

    Returns:
        nn.Module: Model of the same type of `model` with trained state_dict parameters inserted
    """
    load_path = C.SAVE_PATH/Path(load_path).joinpath(file_name).with_suffix('.pt')
    saved_odict = torch.load(load_path)
    modelmod = getattr(model,module,model)
    
    for k,v in saved_odict.items():
        getattr(modelmod,k).load_state_dict(v, strict=False)
    if device is not None:    
        model.to(device)

    return model