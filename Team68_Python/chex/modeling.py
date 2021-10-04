from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision as _tv


def get_pretrained(arch):
	"""Get a pretrained model from torchvision.models

	Args:
		arch (str): One of avaible torchvision.models names 

	Returns:
		nn.Module: pretrained model

	Resources:
		https://gluon-cv.mxnet.io/model_zoo/classification.html
		https://sotabench.com/benchmarks/image-classification-on-imagenet
	"""

	if arch == 'resnet34':
		return _tv.models.resnet34(pretrained=True)
	elif arch == 'resnet50':
		return _tv.models.resnet50(pretrained=True)
	elif arch == 'resnet101':
		return _tv.models.resnet101(pretrained=True)
	elif arch == 'wresnet50':
		return _tv.models.wide_resnet50_2(pretrained=True)
	elif arch == 'densenet121':
		return _tv.models.densenet121(pretrained=True)
	else:
		return _tv.models.__dict__[arch](pretrained=True)
		#return getattr(_tv.models, arch)(pretrained=True)


class AdaptiveConcatPool2d(nn.Module):
	def __init__(self, out_size=1):
		super(AdaptiveConcatPool2d, self).__init__()
		self.ap = nn.AdaptiveAvgPool2d(output_size=out_size)
		self.mp = nn.AdaptiveMaxPool2d(output_size=out_size)

	def forward(self, x):
		x = x.unsqueeze(-1)
		x_avg = self.ap(x)
		x_max = self.mp(x)
		return torch.cat([x_max, x_avg], 1)


class CheXModel(nn.Module):
	"""Pretrained model with the final fully connected layer replaced.

	Args:
		arch (str, optional): Base architecture for pretrained model, should be one of torchvision.models. Defaults to 'densenet121'.
		n_class (int, optional): Number of model output labels . Defaults to 14.
		use_sig (bool, optional): If True, a sigmoid activation function will be added after final Linear layer. Defaults to False.
		color (bool, optional): If False, the leading convolutional layer will be summed across channel dim to allow for grayscale
			images to be passed to the network. Defaults to True.
	"""

	def __init__(self, arch='densenet121', n_class=14, use_sig=False, color=True):

		super(CheXModel, self).__init__()
		self.arch = arch
		self.n_class = n_class
		self.use_sig = use_sig
		self.color = color
		self.network = get_pretrained(arch)
		self._set_head(arch)

	def _set_head(self, arch):
		if (not self.color) and arch=='densenet121':
			# WARNING: Hardcoded for dn121
			c1w = self.network.features.conv0.weight
			self.network.features.conv0.weight = nn.Parameter(c1w.sum(1,keepdim=True), c1w.requires_grad)
			self.network.features.conv0.in_channels = 1
		fcnames = ['fc', 'classifier', '_fc']
		fc_name = [x for x in fcnames if hasattr(self.network, x)][0]
		fc_module = getattr(self.network,fc_name) 
		in_feats = fc_module.in_features  # getattr(self.network,fc_label).in_features
		head = nn.Sequential(nn.Linear(in_feats, self.n_class), nn.Sigmoid()) if self.use_sig else nn.Linear(in_feats,
																											 self.n_class)
		setattr(self.network, fc_name, head)

	def forward(self, x):
		x = self.network(x)

		return x


class ChexResnet(nn.Module):
	def __init__(self, arch='34', n_out=14):
		super(ChexResnet, self).__init__()
		# Use a pretrained model
		self.upcast = nn.Sequential(OrderedDict(
			conv0=nn.Conv2d(1, 3, 3, bias=False),
			bn0=nn.BatchNorm2d(3),
			relu0=nn.ReLU(True)
		))

		self.network = getattr(_tv.models, f'resnet{arch}')(pretrained=True)
		# Replace last layer
		num_ftrs = self.network.fc.in_features
		self.network.fc = nn.Linear(num_ftrs, n_out)

	def forward(self, x):
		x = self.upcast(x)
		x = self.network(x)

		return x


class SimpleCNN(nn.Module):
	def __init__(self, n_out=14):
		super(SimpleCNN, self).__init__()
		# Use a pretrained model
		self.conv1 = nn.Conv2d(1, 3, 3)
		self.bn1 = nn.BatchNorm2d(3)
		self.conv2 = nn.Conv2d(3, 6, 3)
		self.bn2 = nn.BatchNorm2d(6)
		self.relu = nn.ReLU()
		self.pool = nn.MaxPool2d(2)

		self.apool = nn.AdaptiveAvgPool2d((16, 16))

		self.fc1 = nn.Linear(6 * 16 * 16, 256)
		self.fc2 = nn.Linear(256, n_out)

	def forward(self, x):
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.pool(x)
		x = self.relu(self.bn2(self.conv2(x)))
		x = self.apool(x)
		x = x.flatten(1)
		x = self.relu(self.fc1(x))
		x = self.fc2(x)
		return x

