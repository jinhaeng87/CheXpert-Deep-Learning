import numpy as np
import pandas as pd

from . import config as C


def label_smooth(df, method='uones', smooth_bounds=None):
	df_sub = df.copy()
	if smooth_bounds is None:
		eps = 1e-5
		if method == 'uones':
			smooth_bounds = (0.55, 0.85 + eps)
		elif method == 'uzeros':
			smooth_bounds = (0, 0.30 + eps)
		else:
			smooth_bounds = (0, 0.85 + eps)

	if method in ['uones', 'uzeros']:
		smooth_distrb = np.random.uniform(*smooth_bounds, df_sub[C.TARGET_LABELS].shape)
		df_sub.loc[:, C.TARGET_LABELS] = np.where(df_sub[C.TARGET_LABELS] == -1, smooth_distrb, df_sub[C.TARGET_LABELS])

	return df_sub


def proc_df(df, method='uones', smooth=True, nafill_val=0, ufill_val=1, **kwargs):
	"""
	Preprocess dataframe for model consumption

	Args:
		df (pandas.DataFrame): dataframe containing img paths, metadata, and labels.
		method (str), ('uzeros','uones','constant'): method for replacing uncertainty labels (default: 'uones')
		smooth (bool): use Label Smoothing Regression (LSR) only applies when `method`=('uzeros','uones') (default: True)
		nafill_val (int,float): value used to fill nan values (default: 0)
		ufill_val (int,float): value used to fill -1 (uncertainty) labels

	kwargs:
		smooth_bounds (tuple(float,float)): replace -1 labels uniform random values between the given bounds
		(default: `method`='uzeros': (0,0.3001), `method`='uones': (0.55,0.8501) when `Smooth`=True,

	Returns:
		processed_df: pandas.Dataframe
	"""

	is_val = df['Path'].str.contains('valid').all()
	df_sub = df.rename(lambda x: x.replace(' ', '_'), axis=1).drop(columns=['Sex', 'Age', 'Frontal/Lateral', 'AP/PA'], errors='ignore')

	if is_val:
		return df_sub  # val set has no nans, no -1s

	df_targets = df_sub[C.TARGET_LABELS]

	if isinstance(nafill_val, tuple):
		nan_smooth_distrb = np.random.uniform(*nafill_val, df_targets.shape)
		df_sub.loc[:, C.TARGET_LABELS] = np.where(df_targets.isna(), nan_smooth_distrb, df_targets)
	else:
		df_sub = df_sub.fillna(nafill_val)

	if smooth:
		df_sub = label_smooth(df_sub, method, kwargs.get('smooth_bounds'))
	elif method == 'constant':
		df_sub = df_sub.replace(-1, ufill_val)
	elif method in ['uzeros', 'uones'] and ufill_val not in [0.0, 1.0]:
		print(f'WARNING: Overwritting `ufill_val` to match method "{method}"')
		ufill_val = 1.0 if method == 'uones' else 0.0
		df_sub = df_sub.replace(-1, ufill_val)

	df_sub.loc[:, C.TARGET_LABELS] = df_sub.loc[:, C.TARGET_LABELS].astype(float)

	return df_sub


def make_df_merged(df_master):
	# Filter out undesirable data entries
	data_tot = df_master[  # (df_master.Dataset_ID ==1) &
		(df_master.gender != 'U')
		& df_master.ap_pa_ll.isin(['AP', 'LL', 'PA'])
		& df_master[C.TARGET_LABELS].notna().any(axis=1)].reset_index(drop=True)

	# Count each quanity of nan, -1, 0, and 1 labels per sample
	count_mat = np.vstack((np.isnan(data_tot[C.TARGET_LABELS].values).sum(1),
						   (data_tot[C.TARGET_LABELS].values == -1).sum(1),
						   (data_tot[C.TARGET_LABELS].values == 0).sum(1),
						   (data_tot[C.TARGET_LABELS].values == 1).sum(1))).T

	# Create an equal-index dataframe
	df_counts = pd.DataFrame(count_mat, index=data_tot.index, columns=['n_nan', 'n_unk', 'n_neg', 'n_pos'])
	df_totcnts = pd.concat([data_tot[['patient_id', 'study', 'CheX_Image_ID']], df_counts], axis=1)

	# Assign patient-study group id (Option 1 BROKEN)
	# psid = df_totcnts['patient_id'].astype(str).str.pad(7,'right','0').str.cat(df_totcnts['study'].astype(str)).astype(int)
	psid = df_totcnts.groupby(['patient_id', 'study']).ngroup()  # (Option 2)
	df_totcnts.insert(2, 'psid', psid)

	df_merged_all = data_tot.merge(df_totcnts)

	return df_merged_all.rename({'File_Path': 'Path'}, axis=1)


def quick_sample(df_merge, test_size=0.1, data_size=150000, use_allcols=False, seed=404):
	test_df = df_merge[df_merge['Dataset_ID'] == 2]
	df_merge = df_merge.drop(index=test_df.index)
	valid_df = df_merge[df_merge.n_unk.eq(0) & df_merge.n_nan.lt(14)].sample(frac=.1, random_state=seed)
	train_df = df_merge.drop(index=valid_df.index).sample(data_size - len(valid_df))

	tt_df = train_df.append(valid_df).append(test_df)
	tt_df.loc[train_df.index, 'data_id'] = 0
	tt_df.loc[valid_df.index, 'data_id'] = 1
	tt_df.loc[test_df.index, 'data_id'] = 2
	# if tt_df.index.isin(train_df.index) else (1 if tt_df.index.isin(valid_df.index) else 2)
	if use_allcols:
		return tt_df

	subcols = ['CheX_Image_ID', 'ap_pa_ll', 'data_id']
	return tt_df[subcols].assign(ap_pa_ll=tt_df['ap_pa_ll'].map({'AP': 1, 'PA': 2, 'LL': 3})).astype(int)
