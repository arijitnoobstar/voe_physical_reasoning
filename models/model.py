import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Model(nn.Module):
	def __init__(self, model_type, use_pretrained = False, device = 'cpu', resnet_depth = 34, freeze_pretrained_weights = True):
		super(Model, self).__init__()
		self.model_type = model_type
		self.use_pretrained = use_pretrained
		self.freeze_pretrained_weights = freeze_pretrained_weights

		if self.model_type == 'resnet3d_direct':
			self.resnet3d_direct = generate_resnet_model(resnet_depth, mode = 'direct')
			# add pretrained weights of model trained on Kinetics-700 (r3d34_K_200ep)
			if self.use_pretrained:
				pretrained_state_dict = torch.load('pretrained_models/r3d34_K_200ep.pth', map_location=device)['state_dict']
				self.resnet3d_direct.load_state_dict(pretrained_state_dict, strict=False)
				if self.freeze_pretrained_weights:
					freeze_weights(self.resnet3d_direct, unfreezed_layers = ['fc2', 'fc3', 'fc4', 'fc5'])

		if self.model_type == 'Ablation':
			self.Ablation = generate_resnet_model(resnet_depth, mode = 'Ablation')
			# add pretrained weights of model trained on Kinetics-700 (r3d34_K_200ep)
			if self.use_pretrained:
				pretrained_state_dict = torch.load('pretrained_models/r3d34_K_200ep.pth', map_location=device)['state_dict']
				self.Ablation.load_state_dict(pretrained_state_dict, strict=False)
				if self.freeze_pretrained_weights:
					freeze_weights(self.Ablation, unfreezed_layers = ['posterior1','posterior2','posterior3','posterior4',
						'posterior5','posterior6','posterior7','posterior8','posterior9'])

		elif self.model_type == 'OF_PR':
			self.OF_PR = generate_resnet_model(resnet_depth, mode = 'extractor')
			# add pretrained weights of model trained on Kinetics-700 (r3d34_K_200ep)
			if self.use_pretrained:
				pretrained_state_dict = torch.load('pretrained_models/r3d34_K_200ep.pth', map_location=device)['state_dict']
				self.OF_PR.load_state_dict(pretrained_state_dict, strict=False)
				if self.freeze_pretrained_weights:
					freeze_weights(self.OF_PR, unfreezed_layers = [
							'object_height' ,'object_width' ,'left_size' ,'right_size' ,'left_prior_speed' ,'right_prior_speed' ,'left_posterior_speed',
							'right_posterior_speed' ,'left_prior_direction','right_prior_direction','left_posterior_direction',
							'right_posterior_direction','container_height' ,'container_width' ,'wall_opening','wall_opening','opening_width',
							'opening_height' ,'contact_point' ,'middle_segment_height' ,'object_shape','left_shape','right_shape','container_shape',
							'posterior1','posterior2','posterior3','posterior4','posterior5','posterior6','posterior7','posterior8','posterior9'
						])
				# freeze_weights(self.OF_PR, unfreezed_layers = [
				# 		'object_height' ,'object_width','middle_segment_height','object_shape','posterior2'
				# 	])

	def forward(self, x):
		if self.model_type == 'random':
			x = torch.rand((x.size(0), 1))
		elif self.model_type == 'resnet3d_direct':
			x = self.resnet3d_direct(x)
		elif self.model_type == 'OF_PR':
			x = self.OF_PR(x)
		elif self.model_type == 'Ablation':
			x = self.Ablation(x)
		return x

def freeze_weights(model, unfreezed_layers = []):
	for n, p in model.named_parameters():
		if n.split('.')[0] not in unfreezed_layers:
			p.requires_grad = False



class regression_block(nn.Module):
	"""
	simple feed-forward block for feature extraction regression
	for features like width, height, contact point etc.
	"""

	def __init__(self, n_in, activation):

		super().__init__()
		self.activation = activation
		self.rb1 = nn.Linear(n_in, n_in // 2)
		self.rb2 = nn.Linear(n_in // 2, n_in // 4)
		self.rb3 = nn.Linear(n_in // 4, n_in // 8)
		self.rb4 = nn.Linear(n_in // 8, 1)

	def forward(self, x):

		x = F.relu(self.rb1(x))
		x = F.relu(self.rb2(x))
		x = F.relu(self.rb3(x))
		if self.activation == 'relu':
			x = F.relu(self.rb4(x))
		else:
			x = self.rb4(x)

		return x

class classification_block(nn.Module):
	"""
	simple feed-forward block for feature extraction classification
	like object shape, container shape, right/left object shape etc.
	"""

	def __init__(self, n_in, num_categories):

		super().__init__()
		self.num_categories = num_categories

		self.cb1 = nn.Linear(n_in, n_in // 2)
		self.cb2 = nn.Linear(n_in // 2, n_in // 4)
		self.cb3 = nn.Linear(n_in // 4, n_in // 8)
		
		if self.num_categories > 2:
			self.cb4 = nn.Linear(n_in // 8, self.num_categories)
			# self.softmax = nn.Softmax(dim = 1)
		else:
			self.cb4 = nn.Linear(n_in // 8, 1)
			self.sigmoid = nn.Sigmoid()

	def forward(self, x):

		x = F.relu(self.cb1(x))
		x = F.relu(self.cb2(x))
		x = F.relu(self.cb3(x))
		if self.num_categories <= 2:
			x = self.sigmoid(self.cb4(x))
		else:
			# x = self.softmax(self.cb4(x))
			x = self.cb4(x)

		return x

# code below taken from <https://github.com/kenshohara/3D-ResNets-PyTorch> and modified
class ResNet(nn.Module):

	def __init__(self,
				 block,
				 layers,
				 block_inplanes,
				 mode = None,
				 n_input_channels=3,
				 conv1_t_size=7,
				 conv1_t_stride=1,
				 no_max_pool=False,
				 shortcut_type='B',
				 widen_factor=1.0,
				 n_classes=700):
		super().__init__()

		self.mode = mode

		block_inplanes = [int(x * widen_factor) for x in block_inplanes]

		self.in_planes = block_inplanes[0]
		self.no_max_pool = no_max_pool

		self.conv1 = nn.Conv3d(n_input_channels,
							   self.in_planes,
							   kernel_size=(conv1_t_size, 7, 7),
							   stride=(conv1_t_stride, 2, 2),
							   padding=(conv1_t_size // 2, 3, 3),
							   bias=False)
		self.bn1 = nn.BatchNorm3d(self.in_planes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
									   shortcut_type)
		self.layer2 = self._make_layer(block,
									   block_inplanes[1],
									   layers[1],
									   shortcut_type,
									   stride=2)
		self.layer3 = self._make_layer(block,
									   block_inplanes[2],
									   layers[2],
									   shortcut_type,
									   stride=2)
		self.layer4 = self._make_layer(block,
									   block_inplanes[3],
									   layers[3],
									   shortcut_type,
									   stride=2)

		self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
		self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
		# new addition
		if self.mode == 'direct':
			self.fc2 = nn.Linear(n_classes, n_classes // 2)
			self.fc3 = nn.Linear(n_classes // 2, n_classes // 4)
			self.fc4 = nn.Linear(n_classes // 4, n_classes // 8)
			self.fc5 = nn.Linear(n_classes // 8, 1)
			self.sigmoid = nn.Sigmoid()
		elif self.mode == 'extractor':
			# add the feature branches
			self.object_height = regression_block(n_classes, 'linear')
			self.object_width = regression_block(n_classes, 'linear')
			self.left_size = regression_block(n_classes, 'linear')
			self.right_size = regression_block(n_classes, 'linear')
			self.left_prior_speed = regression_block(n_classes, 'linear')
			self.right_prior_speed = regression_block(n_classes, 'linear')
			self.left_posterior_speed = regression_block(n_classes, 'linear')
			self.right_posterior_speed = regression_block(n_classes, 'linear')
			self.left_prior_direction = classification_block(n_classes, 3)
			self.right_prior_direction = classification_block(n_classes, 3)
			self.left_posterior_direction = classification_block(n_classes, 3)
			self.right_posterior_direction = classification_block(n_classes, 3)
			self.container_height = regression_block(n_classes, 'linear')
			self.container_width = regression_block(n_classes, 'linear')
			self.wall_opening = classification_block(n_classes, 3)
			self.wall_opening = classification_block(n_classes, 3)
			self.opening_width = regression_block(n_classes, 'linear')
			self.opening_height = regression_block(n_classes, 'linear')
			self.contact_point = regression_block(n_classes, 'linear')
			self.middle_segment_height = regression_block(n_classes, 'linear')
			self.object_shape = classification_block(n_classes, 8)
			self.left_shape = classification_block(n_classes, 8)
			self.right_shape = classification_block(n_classes, 8)
			self.container_shape = classification_block(n_classes, 3)
			# add the posterior rule branches (the outcome)
			self.posterior1 = classification_block(n_classes, 3)
			self.posterior2 = classification_block(n_classes, 3)
			self.posterior3 = classification_block(n_classes, 3)
			self.posterior4 = classification_block(n_classes, 3)
			self.posterior5 = classification_block(n_classes, 3)
			self.posterior6 = classification_block(n_classes, 3)
			self.posterior7 = classification_block(n_classes, 3)
			self.posterior8 = classification_block(n_classes, 3)
			self.posterior9 = classification_block(n_classes, 3)
		# ablation model
		elif self.mode == 'Ablation': # purely posterior expected outcome
			self.posterior1 = classification_block(n_classes, 3)
			self.posterior2 = classification_block(n_classes, 3)
			self.posterior3 = classification_block(n_classes, 3)
			self.posterior4 = classification_block(n_classes, 3)
			self.posterior5 = classification_block(n_classes, 3)
			self.posterior6 = classification_block(n_classes, 3)
			self.posterior7 = classification_block(n_classes, 3)
			self.posterior8 = classification_block(n_classes, 3)
			self.posterior9 = classification_block(n_classes, 3)

		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_normal_(m.weight,
										mode='fan_out',
										nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _downsample_basic_block(self, x, planes, stride):
		out = F.avg_pool3d(x, kernel_size=1, stride=stride)
		zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
								out.size(3), out.size(4))
		if isinstance(out.data, torch.cuda.FloatTensor):
			zero_pads = zero_pads.cuda()

		out = torch.cat([out.data, zero_pads], dim=1)

		return out

	def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
		downsample = None
		if stride != 1 or self.in_planes != planes * block.expansion:
			if shortcut_type == 'A':
				downsample = partial(self._downsample_basic_block,
									 planes=planes * block.expansion,
									 stride=stride)
			else:
				downsample = nn.Sequential(
					conv1x1x1(self.in_planes, planes * block.expansion, stride),
					nn.BatchNorm3d(planes * block.expansion))

		layers = []
		layers.append(
			block(in_planes=self.in_planes,
				  planes=planes,
				  stride=stride,
				  downsample=downsample))
		self.in_planes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.in_planes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		if not self.no_max_pool:
			x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)

		x = x.view(x.size(0), -1)
		x = self.fc(x)
		# new addition
		if self.mode == 'direct':
			x = F.relu(self.fc2(x))
			x = F.relu(self.fc3(x))
			x = F.relu(self.fc4(x))
			x = F.relu(self.fc5(x))
			x = self.sigmoid(x)

			return x
			
		elif self.mode == 'extractor':
			# feature branches
			f1 = self.object_height(x)
			f2 = self.object_width(x)
			f3 = self.left_size(x)
			f4 = self.right_size(x)
			f5 = self.left_prior_speed(x)
			f6 = self.right_prior_speed(x)
			f7 = self.left_posterior_speed(x)
			f8 = self.right_posterior_speed(x)
			f9 = self.left_prior_direction(x)
			f10 = self.right_prior_direction(x)
			f11 = self.left_posterior_direction(x)
			f12 = self.right_posterior_direction(x)
			f13 = self.container_height(x)
			f14 = self.container_width(x)
			f15 = self.wall_opening(x)
			f16 = self.wall_opening(x)
			f17 = self.opening_width(x)
			f18 = self.opening_height(x)
			f19 = self.contact_point(x)
			f20 = self.middle_segment_height(x)
			f21 = self.object_shape(x)
			f22 = self.left_shape(x)
			f23 = self.right_shape(x)
			f24 = self.container_shape(x)
			# posterior branches
			p1 = self.posterior1(x)
			p2 = self.posterior2(x)
			p3 = self.posterior3(x)
			p4 = self.posterior4(x)
			p5 = self.posterior5(x)
			p6 = self.posterior6(x)
			p7 = self.posterior7(x)
			p8 = self.posterior8(x)
			p9 = self.posterior9(x)

			return f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15, \
			f16,f17,f18,f19,f20,f21,f22,f23,f24,p1,p2,p3,p4,p5,p6,p7,p8,p9

		elif self.mode == 'Ablation':
			p1 = self.posterior1(x)
			p2 = self.posterior2(x)
			p3 = self.posterior3(x)
			p4 = self.posterior4(x)
			p5 = self.posterior5(x)
			p6 = self.posterior6(x)
			p7 = self.posterior7(x)
			p8 = self.posterior8(x)
			p9 = self.posterior9(x)

			return p1,p2,p3,p4,p5,p6,p7,p8,p9

def get_inplanes():
	return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes,
					 out_planes,
					 kernel_size=3,
					 stride=stride,
					 padding=1,
					 bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
	return nn.Conv3d(in_planes,
					 out_planes,
					 kernel_size=1,
					 stride=stride,
					 bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, downsample=None):
		super().__init__()

		self.conv1 = conv3x3x3(in_planes, planes, stride)
		self.bn1 = nn.BatchNorm3d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3x3(planes, planes)
		self.bn2 = nn.BatchNorm3d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1, downsample=None):
		super().__init__()

		self.conv1 = conv1x1x1(in_planes, planes)
		self.bn1 = nn.BatchNorm3d(planes)
		self.conv2 = conv3x3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm3d(planes)
		self.conv3 = conv1x1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm3d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

def generate_resnet_model(model_depth, mode = None, **kwargs):
	assert model_depth in [10, 18, 34, 50, 101, 152, 200]

	if model_depth == 10:
		model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), mode = mode, **kwargs)
	elif model_depth == 18:
		model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), mode = mode, **kwargs)
	elif model_depth == 34:
		model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), mode = mode, **kwargs)
	elif model_depth == 50:
		model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), mode = mode, **kwargs)
	elif model_depth == 101:
		model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), mode = mode, **kwargs)
	elif model_depth == 152:
		model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), mode = mode, **kwargs)
	elif model_depth == 200:
		model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), mode = mode, **kwargs)

	return model