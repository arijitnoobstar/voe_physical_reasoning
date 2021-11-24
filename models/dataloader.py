import os
import numpy as np
import json
import torch
import random
from torch.utils.data import Dataset
from torchvision.io import read_video
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.video_dataset import VideoFrameDataset

shape_mapping = {
	'Cube': 0,
	'Sphere' : 1,
	'Torus' : 2,
	'Cylinder' : 3,
	'Side_Cylinder' : 4,
	'Cone' : 5,
	'Inverted_Cone' : 6
} # NOTE invalid shape is considered as 7

container_mapping = {
	'box': 0,
	'mug' : 1,
} # NOTE invalid shape is considered as 2

feature_list = [
	'object_height', #0
	'object_width', #1
	'left_size', #2
	'right_size', #3
	'left_prior_speed', #4
	'right_prior_speed', #5
	'left_posterior_speed', #6
	'right_posterior_speed', #7
	'left_prior_direction', #8
	'right_prior_direction', #9
	'left_posterior_direction', #10
	'right_posterior_direction', #11
	'container_height', #12
	'container_width', #13
	'wall_opening',#14
	'wall_soft',#15
	'opening_width', #16
	'opening_height', #17
	'contact_point', #18
	'middle_segment_height', #19
	'object_shape',#20
	'left_shape',#21
	'right_shape',#22
	'container_shape' #23
]

prior_rule_list = [
	'does_object_have_majority_contact_proportion?',
	'does_object_have_majority_volume_proportion?',
	'is_object_taller_than_middle?',
	'is_object_taller_than_container?',
	'is_object_thinner_than_container?',
	'is_right_object_larger?',
	'are_both_objects_same_size?',
	'is_right_object_faster?',
	'are_both_objects_same_speed?',
	'is_there_an_opening?',
	'is_the_blocker_soft?',
	'is_the_object_thinner_than_the_opening?',
	'is_the_object_shorter_than_the_opening?'
]

posterior_rule_list = [
	'does_support_hold_object?',#24
	'see_object_in_middle?',#25
	'did_the_object_fit?',#26
	'did_the_object_protude?',#27
	'did_right_object_change_direction?',#28
	'did_left_object_change_direction?',#29
	'did_right_object_increase_speed_magnitude?',#30
	'did_left_object_increase_speed_magnitude?',#31
	'did_the_object_pass_through_the_wall?'#32
]

class memoryEfficient_AVoEDataset(torch.utils.data.Dataset):

	def __init__(self, annotation_path, annotation_name, root_path, dataset_type, model_type, instance = False, depth = False, alpha = 0):
		super(memoryEfficient_AVoEDataset, self).__init__()

		self.annotation_path = annotation_path
		self.annotation_name = annotation_name
		self.root_path = root_path
		self.dataset_type = dataset_type
		self.model_type = model_type

		if self.dataset_type != 'train':
			# self.videos = []
			self.metadata = []
		else:
			# self.expected_videos = []
			# self.surprising_videos = []
			self.expected_metadata = []
			self.surprising_metadata = []

		self.alpha = alpha
		self.do_instance = instance
		self.do_depth = depth

		f = open(self.annotation_path)
		annotations = f.readlines()
		f.close()

		annotations_len = len(annotations)
		
		for video_count, line in enumerate(annotations[:]):
			# remove newline
			line = line[:-1]
			# get label
			label = int(line.split(' ')[1])
			# get trial identifier
			trial_name = line.split(' ')[0]

			# extract all the feature and rules info
			features = []
			prior_rules = []
			posterior_rules = []
			with open(self.root_path + trial_name + '/physical_data.json') as jsonfile:
				metadata = json.load(jsonfile)

			# insert features in fixed order, with -1 indicating invalid feature
			for feature in feature_list:
				if feature == 'container_shape':
					try:
						features.append(container_mapping[metadata['features']['container_shape']])
					except:
						features.append(2) # invalid container has a label of 2
				elif feature == 'left_shape':
					try:
						features.append(shape_mapping[metadata['features']['left_shape']])
					except:
						features.append(7) # invalid shape has a label of 7
				elif feature == 'right_shape':	
					try:
						features.append(shape_mapping[metadata['features']['right_shape']])
					except:
						features.append(7) # invalid shape has a label of 7
				elif feature == 'object_shape':
					try:
						features.append(shape_mapping[metadata['features']['object_shape']])
					except:
						features.append(7) # invalid shape has a label of 7
				elif feature == 'wall_opening':
					try:
						features.append(1 * metadata['features']['wall_opening'])
					except:
						features.append(2) # invalid binary option has a label of 2
				elif feature == 'wall_soft':
					try:
						features.append(1 * metadata['features']['wall_soft'])
					except:
						features.append(2) # invalid binary option has a label of 2
				elif feature == 'left_prior_speed': # magnitude of velocity
					try:
						features.append(abs(metadata['features']['left_prior_velocity']))
					except:
						features.append(-1) # invalid regression option has a label of -1
				elif feature == 'right_prior_speed': # magnitude of velocity
					try:
						features.append(abs(metadata['features']['right_prior_velocity']))
					except:
						features.append(-1) # invalid regression option has a label of -1
				elif feature == 'left_posterior_speed': # magnitude of velocity
					try:
						features.append(abs(metadata['features']['left_posterior_velocity']))
					except:
						features.append(-1) # invalid regression option has a label of -1
				elif feature == 'right_posterior_speed': # magnitude of velocity
					try:
						features.append(abs(metadata['features']['right_posterior_velocity']))
					except:
						features.append(-1) # invalid regression option has a label of -1
				elif feature == 'left_prior_direction': # direction of velocity (-ve is 0, non -ve is 1)
					try:
						if metadata['features']['left_prior_velocity'] < 0:
							features.append(0)
						else:
							features.append(1)
					except:
						features.append(2) # invalid binary option has a label of 2
				elif feature == 'right_prior_direction': # direction of velocity (-ve is 0, non -ve is 1)
					try:
						if metadata['features']['right_prior_velocity'] < 0:
							features.append(0)
						else:
							features.append(1)
					except:
						features.append(2) # invalid binary option has a label of 2
				elif feature == 'left_posterior_direction': # direction of velocity (-ve is 0, non -ve is 1)
					try:
						if metadata['features']['left_posterior_velocity'] < 0:
							features.append(0)
						else:
							features.append(1)
					except:
						features.append(2) # invalid binary option has a label of 2
				elif feature == 'right_posterior_direction': # direction of velocity (-ve is 0, non -ve is 1)
					try:
						if metadata['features']['right_posterior_velocity'] < 0:
							features.append(0)
						else:
							features.append(1)
					except:
						features.append(2) # invalid binary option has a label of 2
				# all other regression features
				else:
					try:
						features.append(metadata['features'][feature])
					except:
						features.append(-1) # invalid regression option has a label of -1

			# insert prior rules in fixed order, with -1 indicating invalid rule
			for prior_rule in prior_rule_list:				
				try:
					prior_rules.append(1 * metadata['prior_rules'][prior_rule])
				except:
					prior_rules.append(2) # invalid binary option has a label of 2
			# insert posterior rules in fixed order, with -1 indicating invalid rule
			for posterior_rule in posterior_rule_list:				
				try:
					posterior_rules.append(1 * metadata['posterior_rules'][posterior_rule])
				except:
					posterior_rules.append(2) # invalid binary option has a label of 2

			if self.dataset_type != 'train':
				self.metadata.append((label, trial_name, features, prior_rules, posterior_rules))
			elif 'expected' in trial_name:
				self.expected_metadata.append((label, trial_name, features, prior_rules, posterior_rules))
			elif 'surprising' in trial_name:
				self.surprising_metadata.append((label, trial_name, features, prior_rules, posterior_rules))
			else:
				raise ValueError("invalid data label")

		if self.dataset_type != 'train':
			self.len = len(self.metadata)
		else:
			self.len = len(self.expected_metadata)
		

	def _get_video(self, trial_name, video_type):
		
		# read tensor from avi file, unsqueeze, resize (interpolate),scale then add to video data
		video_tensor = read_video(self.root_path + trial_name + '/{}.avi'.format(video_type))[0] # extract video into pytorch tensor (T,H,W,C)
		video_tensor = torch.unsqueeze(video_tensor, dim=0) # (1,T,H,W,C)
		if 'resnet3d' in self.model_type: # OF_PR and Ablation and resnet3d_direct
			# resnet3d has a specifc type of input
			video_tensor = F.interpolate(video_tensor, size=(112, 112, 3)) # (1,T,112,112,C)
			video_tensor = torch.squeeze(video_tensor) # (T,112,112,C)
			video_tensor = torch.permute(video_tensor, (3,0,1,2)) # (C,T,112,112)
			video_tensor = video_tensor / 255.0 # scale to 0~1
		else:
			video_tensor = F.interpolate(video_tensor, size=(32, 32, 3)) # (1,T,H_resized,W_resized,C)
			video_tensor = torch.squeeze(video_tensor) # (T,H_resized,W_resized,C)
			video_tensor = torch.permute(video_tensor, (0,3,1,2)) # (T,C,H_resized,W_resized)
			video_tensor = torch.reshape(video_tensor, (-1, 32, 32)) # (T*C,H_resized,W_resized)
			video_tensor = video_tensor / 255.0 # scale to 0~1

		return video_tensor

	def __getitem__(self, idx):

		# for training, save surprising and expected videos separately
		if self.dataset_type != 'train':
			label, trial_name, features, prior_rules, posterior_rules = self.metadata[idx]
			# gather rgb, instance, depth information into torch tensors
			rgb_video_tensor = self._get_video(trial_name, 'rgb')
			data_item = [rgb_video_tensor, label, trial_name, features, prior_rules, posterior_rules]
			if self.do_instance:
				inst_video_tensor = self._get_video(trial_name, 'inst_raw')
				data_item.append(inst_video_tensor)
			if self.do_depth:
				depth_video_tensor = self._get_video(trial_name, 'depth_raw')
				data_item.append(depth_video_tensor)
		# otherwise put them all in the same list
		else:
			if random.uniform(0,1) >= self.alpha:
				label, trial_name, features, prior_rules, posterior_rules = self.expected_metadata[idx]
				# gather rgb, instance, depth information into torch tensors
				rgb_video_tensor = self._get_video(trial_name, 'rgb')
				data_item = [rgb_video_tensor, label, trial_name, features, prior_rules, posterior_rules]
				if self.do_instance:
					inst_video_tensor = self._get_video(trial_name, 'inst_raw')
					data_item.append(inst_video_tensor)
				if self.do_depth:
					depth_video_tensor = self._get_video(trial_name, 'depth_raw')
					data_item.append(depth_video_tensor)
			else:
				label, trial_name, features, prior_rules, posterior_rules = self.surprising_metadata[idx]
				# gather rgb, instance, depth information into torch tensors
				rgb_video_tensor = self._get_video(trial_name, 'rgb')
				data_item = [rgb_video_tensor, label, trial_name, features, prior_rules, posterior_rules]
				if self.do_instance:
					inst_video_tensor = self._get_video(trial_name, 'inst_raw')
					data_item.append(inst_video_tensor)
				if self.do_depth:
					depth_video_tensor = self._get_video(trial_name, 'depth_raw')
					data_item.append(depth_video_tensor)

		return data_item

	def set_alpha(self, value):
		""" set alpha (ratio of surprising to expected videos) """
		self.alpha = value

	def __len__(self):
		return self.len

class timeEfficient_AVoEDataset(torch.utils.data.Dataset):

	def __init__(self, annotation_path, annotation_name, root_path, dataset_type, model_type, instance = False, depth = False, alpha = 0):
		super(timeEfficient_AVoEDataset, self).__init__()

		self.annotation_path = annotation_path
		self.annotation_name = annotation_name
		self.root_path = root_path
		self.dataset_type = dataset_type
		self.model_type = model_type
		if self.dataset_type != 'train':
			self.videos = []
		else:
			self.expected_videos = []
			self.surprising_videos = []
		self.alpha = alpha

		f = open(self.annotation_path)
		annotations = f.readlines()
		f.close()

		annotations_len = len(annotations)
		
		for video_count, line in enumerate(annotations[:]):
			# remove newline
			line = line[:-1]
			# get label
			label = int(line.split(' ')[1])
			# get trial identifier
			trial_name = line.split(' ')[0]

			# extract all the feature and rules info
			features = []
			prior_rules = []
			posterior_rules = []
			with open(self.root_path + trial_name + '/physical_data.json') as jsonfile:
				metadata = json.load(jsonfile)

			# insert features in fixed order, with -1 indicating invalid feature
			for feature in feature_list:
				if feature == 'container_shape':
					try:
						features.append(container_mapping[metadata['features']['container_shape']])
					except:
						features.append(2) # invalid container has a label of 2
				elif feature == 'left_shape':
					try:
						features.append(shape_mapping[metadata['features']['left_shape']])
					except:
						features.append(7) # invalid shape has a label of 7
				elif feature == 'right_shape':	
					try:
						features.append(shape_mapping[metadata['features']['right_shape']])
					except:
						features.append(7) # invalid shape has a label of 7
				elif feature == 'object_shape':
					try:
						features.append(shape_mapping[metadata['features']['object_shape']])
					except:
						features.append(7) # invalid shape has a label of 7
				elif feature == 'wall_opening':
					try:
						features.append(1 * metadata['features']['wall_opening'])
					except:
						features.append(2) # invalid binary option has a label of 2
				elif feature == 'wall_soft':
					try:
						features.append(1 * metadata['features']['wall_soft'])
					except:
						features.append(2) # invalid binary option has a label of 2
				elif feature == 'left_prior_speed': # magnitude of velocity
					try:
						features.append(abs(metadata['features']['left_prior_velocity']))
					except:
						features.append(-1) # invalid regression option has a label of -1
				elif feature == 'right_prior_speed': # magnitude of velocity
					try:
						features.append(abs(metadata['features']['right_prior_velocity']))
					except:
						features.append(-1) # invalid regression option has a label of -1
				elif feature == 'left_posterior_speed': # magnitude of velocity
					try:
						features.append(abs(metadata['features']['left_posterior_velocity']))
					except:
						features.append(-1) # invalid regression option has a label of -1
				elif feature == 'right_posterior_speed': # magnitude of velocity
					try:
						features.append(abs(metadata['features']['right_posterior_velocity']))
					except:
						features.append(-1) # invalid regression option has a label of -1
				elif feature == 'left_prior_direction': # direction of velocity (-ve is 0, non -ve is 1)
					try:
						if metadata['features']['left_prior_velocity'] < 0:
							features.append(0)
						else:
							features.append(1)
					except:
						features.append(2) # invalid binary option has a label of 2
				elif feature == 'right_prior_direction': # direction of velocity (-ve is 0, non -ve is 1)
					try:
						if metadata['features']['right_prior_velocity'] < 0:
							features.append(0)
						else:
							features.append(1)
					except:
						features.append(2) # invalid binary option has a label of 2
				elif feature == 'left_posterior_direction': # direction of velocity (-ve is 0, non -ve is 1)
					try:
						if metadata['features']['left_posterior_velocity'] < 0:
							features.append(0)
						else:
							features.append(1)
					except:
						features.append(2) # invalid binary option has a label of 2
				elif feature == 'right_posterior_direction': # direction of velocity (-ve is 0, non -ve is 1)
					try:
						if metadata['features']['right_posterior_velocity'] < 0:
							features.append(0)
						else:
							features.append(1)
					except:
						features.append(2) # invalid binary option has a label of 2
				# all other regression features
				else:
					try:
						features.append(metadata['features'][feature])
					except:
						features.append(-1) # invalid regression option has a label of -1

			# insert prior rules in fixed order, with -1 indicating invalid rule
			for prior_rule in prior_rule_list:				
				try:
					prior_rules.append(1 * metadata['prior_rules'][prior_rule])
				except:
					prior_rules.append(2) # invalid binary option has a label of 2
			# insert posterior rules in fixed order, with -1 indicating invalid rule
			for posterior_rule in posterior_rule_list:				
				try:
					posterior_rules.append(1 * metadata['posterior_rules'][posterior_rule])
				except:
					posterior_rules.append(2) # invalid binary option has a label of 2

			# for training, save surprising and expected videos separately
			if self.dataset_type != 'train':
				# gather rgb, instance, depth information into torch tensors
				rgb_video_tensor = self._get_video(trial_name, 'rgb')
				self.videos.append([rgb_video_tensor, label, trial_name, features, prior_rules, posterior_rules])
				if instance:
					inst_video_tensor = self._get_video(trial_name, 'inst_raw')
					self.videos[video_count].append(inst_video_tensor)
				if depth:
					depth_video_tensor = self._get_video(trial_name, 'depth_raw')
					self.videos[video_count].append(depth_video_tensor)
			# otherwise put them all in the same list
			else:
				if 'expected' in trial_name:
					# gather rgb, instance, depth information into torch tensors
					rgb_video_tensor = self._get_video(trial_name, 'rgb')
					self.expected_videos.append([rgb_video_tensor, label, trial_name, features, prior_rules, posterior_rules])
					exp_idx = len(self.expected_videos) - 1
					if instance:
						inst_video_tensor = self._get_video(trial_name, 'inst_raw')
						self.expected_videos[exp_idx].append(inst_video_tensor)
					if depth:
						depth_video_tensor = self._get_video(trial_name, 'depth_raw')
						self.expected_videos[exp_idx].append(depth_video_tensor)
				else:
					# gather rgb, instance, depth information into torch tensors
					rgb_video_tensor = self._get_video(trial_name, 'rgb')
					self.surprising_videos.append([rgb_video_tensor, label, trial_name, features, prior_rules, posterior_rules])
					surp_idx = len(self.surprising_videos) - 1
					if instance:
						inst_video_tensor = self._get_video(trial_name, 'inst_raw')
						self.surprising_videos[surp_idx].append(inst_video_tensor)
					if depth:
						depth_video_tensor = self._get_video(trial_name, 'depth_raw')
						self.surprising_videos[surp_idx].append(depth_video_tensor)

			print("Loading {} dataset: {}/{}".format(self.dataset_type, video_count+1, annotations_len), end = '\r')

		if self.dataset_type != 'train':
			self.len = len(self.videos)
		else:
			self.len = len(self.expected_videos)
		

	def _get_video(self, trial_name, video_type):
		
		# read tensor from avi file, unsqueeze, resize (interpolate),scale then add to video data
		video_tensor = read_video(self.root_path + trial_name + '/{}.avi'.format(video_type))[0] # extract video into pytorch tensor (T,H,W,C)
		video_tensor = torch.unsqueeze(video_tensor, dim=0) # (1,T,H,W,C)
		if 'resnet3d' in self.model_type: # OF_PR and Ablation and resnet3d_direct
			# resnet3d has a specifc type of input
			video_tensor = F.interpolate(video_tensor, size=(112, 112, 3)) # (1,T,112,112,C)
			video_tensor = torch.squeeze(video_tensor) # (T,112,112,C)
			video_tensor = torch.permute(video_tensor, (3,0,1,2)) # (C,T,112,112)
			video_tensor = video_tensor / 255.0 # scale to 0~1
		else:
			video_tensor = F.interpolate(video_tensor, size=(32, 32, 3)) # (1,T,H_resized,W_resized,C)
			video_tensor = torch.squeeze(video_tensor) # (T,H_resized,W_resized,C)
			video_tensor = torch.permute(video_tensor, (0,3,1,2)) # (T,C,H_resized,W_resized)
			video_tensor = torch.reshape(video_tensor, (-1, 32, 32)) # (T*C,H_resized,W_resized)
			video_tensor = video_tensor / 255.0 # scale to 0~1

		return video_tensor

	def __getitem__(self, idx):
		if self.dataset_type != 'train':
			return self.videos[idx]
		if random.uniform(0,1) < self.alpha:
			return self.surprising_videos[idx]
		else:
			return self.expected_videos[idx]

	def set_alpha(self, value):
		""" set alpha (ratio of surprising to expected videos) """
		self.alpha = value

	def __len__(self):
		return self.len

def load_avoe_data(root = None, model_type = None, annotation_name = 'combined' ,test = False, val = False, train = False, 
	instance = False, depth = False, alpha = 0, dataset_efficiency = None):
	"""
	load the AVoE dataset into train, val and test. Dataset mix is based on annotation_name
	"""
	# annotations file path for TRAIN
	train_annotations_path = os.path.join(os.getcwd(), 'utils/annotations/annotations_{}_train.txt'.format(annotation_name)) 
	# annotations file path for VAL
	val_annotations_path = os.path.join(os.getcwd(), 'utils/annotations/annotations_{}_val.txt'.format(annotation_name)) 
	# annotations file path for TEST
	test_annotations_path = os.path.join(os.getcwd(), 'utils/annotations/annotations_{}_test.txt'.format(annotation_name)) 
	# load 

	datasets = []

	if dataset_efficiency == 'time':

		if train:

			datasets.append(timeEfficient_AVoEDataset(
				root_path=root,
				annotation_path=train_annotations_path,
				annotation_name= annotation_name,
				dataset_type = 'train',
				model_type = model_type,
				instance = instance,
				depth = depth,
				alpha = alpha
			))

		if val: 

			datasets.append(timeEfficient_AVoEDataset(
				root_path=root,
				annotation_path=val_annotations_path,
				annotation_name= annotation_name,
				dataset_type = 'val',
				model_type = model_type,
				instance = instance,
				depth = depth,
				alpha = alpha
			))

		if test:

			datasets.append(timeEfficient_AVoEDataset(
				root_path=root,
				annotation_path=test_annotations_path,
				annotation_name= annotation_name,
				dataset_type = 'test',
				model_type = model_type,
				instance = instance,
				depth = depth,
				alpha = alpha
			))

	elif dataset_efficiency == 'memory':

		if train:

			datasets.append(memoryEfficient_AVoEDataset(
				root_path=root,
				annotation_path=train_annotations_path,
				annotation_name= annotation_name,
				dataset_type = 'train',
				model_type = model_type,
				instance = instance,
				depth = depth,
				alpha = alpha
			))

		if val: 

			datasets.append(memoryEfficient_AVoEDataset(
				root_path=root,
				annotation_path=val_annotations_path,
				annotation_name= annotation_name,
				dataset_type = 'val',
				model_type = model_type,
				instance = instance,
				depth = depth,
				alpha = alpha
			))

		if test:

			datasets.append(memoryEfficient_AVoEDataset(
				root_path=root,
				annotation_path=test_annotations_path,
				annotation_name= annotation_name,
				dataset_type = 'test',
				model_type = model_type,
				instance = instance,
				depth = depth,
				alpha = alpha
			))

	return tuple(datasets)