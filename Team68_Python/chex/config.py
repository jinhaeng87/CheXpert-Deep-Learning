from pathlib import Path

_global_config = dict(
	SEED = 404,
	USE_CUDA = False,
	ON_AWS = False,
	NUM_WORKERS = 0,
	DEVICE = None,
)

_orig_config = _global_config.copy()

SEED,USE_CUDA,ON_AWS,NUM_WORKERS,DEVICE = (
	_global_config.get('SEED'),
	_global_config.get('USE_CUDA'),
	_global_config.get('ON_AWS'),
	_global_config.get('NUM_WORKERS'),
	_global_config.get('DEVICE')
)

# Path Options
DATA_PATH = Path('data/')
SAVE_PATH = Path('save/')
TEMPLATE_PATH = Path('data/templates')

CHX_PATH =  DATA_PATH/'CheXpert-v1.0-small/'
TRAIN_PATH = CHX_PATH/'train'
VALID_PATH = CHX_PATH/'valid'

# Template Images
FRONTAL_TEMPLATE_PATH = str(TEMPLATE_PATH / 'fnt_ext_template244.jpg')
LATERAL_TEMPLATE_PATH = str(TEMPLATE_PATH / 'lat_ext_template244.jpg')

# Label Options
TARGET_LABELS = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly',
       'Lung_Opacity', 'Lung_Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other',
       'Fracture', 'Support_Devices']
# Core 5 Labels
TARGET5_LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural_Effusion']
TARGET5_INDEX = [TARGET_LABELS.index(t5) for t5 in TARGET5_LABELS]

# All labels excluding Fracture, which has no entries in validation set
TARGET13_LABELS = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly',
       'Lung_Opacity', 'Lung_Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other', 'Support_Devices']
TARGET13_IDX = [TARGET_LABELS.index(t) for t in TARGET13_LABELS]




def get_config(*config):
	if len(config) == 0:
		return _global_config.copy()
	return [_global_config.get(k) for k in config] if len(config) > 1 else _global_config[config[0]]


def set_config(**config):
	global _global_config
	gbls = globals()
	#update_globals = False
	for k,v in config.items():
		if k in _global_config:
			gbls[k] = v	
		_global_config[k] = v


def reset_config():
	global _global_config, _orig_config
	_global_config = _orig_config
	
	global SEED, USE_CUDA, ON_AWS, NUM_WORKERS, DEVICE
	SEED,USE_CUDA,ON_AWS,NUM_WORKERS,DEVICE = get_config('SEED','USE_CUDA','ON_AWS','NUM_WORKERS','DEVICE')







