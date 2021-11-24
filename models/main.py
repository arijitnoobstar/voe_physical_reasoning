import torch.nn as nn
import torch
from dataloader import load_avoe_data
from model import Model, generate_resnet_model
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import random
import yaml
import argparse
import datetime
from sklearn import metrics, tree
from sklearn.multioutput import MultiOutputClassifier
import os
import sys
import math
import json

train_branches_ofpr = {
	'A' : [0,1,18,20,24],
	'B' : [0,1,19,20,25],
	'C' : [0,1,12,13,20,23,26,27],
	'D' : [2,3,4,5,6,7,8,9,10,11,21,22,28,29,30,31],
	'E' : [0,1,14,15,16,17,20,32],
	'combined' : list(range(33))
}

train_branches_ablation = {
	'A' : [0],
	'B' : [1],
	'C' : [2,3],
	'D' : [4,5,6,7],
	'E' : [8],
	'combined' : list(range(9))
}

def calculate_hit_score(pred_labels):
	"""
	NOTE: val dataset is always in the order of exp - surprising for the same trial
	for a batch of videos, find the number of hits (expected score < surprise score for the same stimuli)
	if expected score is equal to surprise score, the hit score is given a value of 0.5 
	""" 
	hit_score = 0
	for idx in range(0,len(pred_labels), 2):
		if pred_labels[idx] < pred_labels[idx + 1]:
			hit_score += 1
		elif pred_labels[idx] == pred_labels[idx + 1]:
			hit_score += 0.5
	return hit_score / (len(pred_labels) // 2)

def calculate_auc(pred_labels, labels):
	"""
	calculate AUC of surprising video score vs expected video score
	"""

	return metrics.roc_auc_score(labels.squeeze().tolist(), pred_labels.squeeze().tolist())

def train_avoe_decision_tree(dataset, save_dts = True, experiment_path = None):
	"""
	train a decision tree model for the specfied AVoE dataset (require features, prior and posterior rule data)
	"""

	feature_matrix = []
	prior_matrix = []
	posterior_matrix = []
	feature_prior_combined_matrix = []

	# gather metadata into matrices
	for idx in range(len(dataset)):
		_, _, _, features, prior_rules, posterior_rules = dataset[idx]
		feature_matrix.append(features)
		prior_matrix.append(prior_rules)
		posterior_matrix.append(posterior_rules)
		feature_prior_combined_matrix.append(features + prior_rules)

	# convert each matrix into numpy array 
	feature_matrix = np.array(feature_matrix)
	prior_matrix = np.array(prior_matrix)
	posterior_matrix = np.array(posterior_matrix)
	feature_prior_combined_matrix = np.array(feature_prior_combined_matrix)

	# train multi-target decision tree 1: features --> prior rules
	dt_1 = MultiOutputClassifier(tree.DecisionTreeClassifier())
	dt_1.fit(feature_matrix, prior_matrix)

	# train multi-target decision tree 2: prior rules --> posterior rules
	dt_2 = MultiOutputClassifier(tree.DecisionTreeClassifier())
	dt_2.fit(prior_matrix, posterior_matrix)

	# train multi-target decision tree 3: features --> posterior rules
	dt_3 = MultiOutputClassifier(tree.DecisionTreeClassifier())
	dt_3.fit(feature_matrix, posterior_matrix)

	# train multi-target decision tree 4: features + prior rules --> posterior rules
	dt_4 = MultiOutputClassifier(tree.DecisionTreeClassifier())
	dt_4.fit(feature_prior_combined_matrix, posterior_matrix)

	# save the dt models if necessary
	if save_dts:
		torch.save(dt_1, experiment_path + 'dt_1.pth')
		torch.save(dt_2, experiment_path + 'dt_2.pth')
		torch.save(dt_3, experiment_path + 'dt_3.pth')
		torch.save(dt_4, experiment_path + 'dt_4.pth')

	dt_1_score = dt_1.score(feature_matrix, prior_matrix)
	dt_2_score = dt_2.score(prior_matrix, posterior_matrix)
	dt_3_score = dt_3.score(feature_matrix, posterior_matrix)
	dt_4_score = dt_4.score(feature_prior_combined_matrix, posterior_matrix)

	print("TRAIN dt_1 score: {:.2f}, dt_2 score: {:.2f}, dt_3 score: {:.2f}, dt_4 score: {:.2f}".format(dt_1_score,dt_2_score,dt_3_score,dt_4_score))

	# sys.exit()

	return dt_1, dt_2, dt_3, dt_4

def eval_avoe_decision_tree(dt_1, dt_2, dt_3, dt_4, eval_dataset):
	"""
	evaluate performance of decision tree
	"""
	feature_matrix = []
	prior_matrix = []
	posterior_matrix = []
	feature_prior_combined_matrix = []

	# gather metadata into matrices
	for idx in range(len(eval_dataset)):
		_, _, _, features, prior_rules, posterior_rules = eval_dataset[idx]
		feature_matrix.append(features)
		prior_matrix.append(prior_rules)
		posterior_matrix.append(posterior_rules)
		feature_prior_combined_matrix.append(features + prior_rules)

	# convert each matrix into numpy array 
	feature_matrix = np.array(feature_matrix)
	prior_matrix = np.array(prior_matrix)
	posterior_matrix = np.array(posterior_matrix)
	feature_prior_combined_matrix = np.array(feature_prior_combined_matrix)

	dt_1_score = dt_1.score(feature_matrix, prior_matrix)
	dt_2_score = dt_2.score(prior_matrix, posterior_matrix)
	dt_3_score = dt_3.score(feature_matrix, posterior_matrix)
	dt_4_score = dt_4.score(feature_prior_combined_matrix, posterior_matrix)

	print("VAL dt_1 score: {:.2f}, dt_2 score: {:.2f}, dt_3 score: {:.2f}, dt_4 score: {:.2f}".format(dt_1_score,dt_2_score,dt_3_score,dt_4_score))

	return dt_1_score, dt_2_score, dt_3_score, dt_4_score

def determine_surprise_scores(dt_1, dt_2, dt_3, dt_4, pred_labels, gt_posterior_rules, no_prior = False, 
	combine_feature_prior = False, semi_oracle = False, model_type = None):
	
	num_samples = len(pred_labels[0])
	num_posterior_rules = len(gt_posterior_rules)
	if model_type == 'OF_PR':
		# extract all the outputs in a form compatible with the multi-target decision tree
		nn_features = []
		nn_posterior_rules = []
		for sample_num in range(num_samples):
			sample_feature_output = []
			# only loop through predicted features
			for nn_branch_num in range(len(pred_labels) - num_posterior_rules): 
				value = pred_labels[nn_branch_num][sample_num]
				if len(value) > 1:
					# convert pre-softmax values to class label
					value = torch.argmax(value).item()
				else:
					value = value.item()
				sample_feature_output.append(value)
			nn_features.append(sample_feature_output)
			# obtain nn posterior rules
			sample_posterior_output = []
			for nn_branch_num in range(len(pred_labels) - num_posterior_rules, len(pred_labels)): 
				value = pred_labels[nn_branch_num][sample_num]
				if len(value) > 1:
					# convert pre-softmax values to class label
					value = torch.argmax(value).item()
				else:
					value = value.item()
				sample_posterior_output.append(value)
			nn_posterior_rules.append(sample_posterior_output)

		# extract decision tree outcome
		if no_prior:
			dt_posterior_rules = dt_3.predict(nn_features)
		elif combine_feature_prior:
			dt_prior_rules = dt_1.predict(nn_features)
			dt_feature_prior_combine = [np.array(list(nn_features[i]) + list(dt_prior_rules[i])) for i in range(num_samples)] # get combined
			dt_posterior_rules = dt_4.predict(dt_feature_prior_combine)
		else:
			dt_prior_rules = dt_1.predict(nn_features)
			dt_posterior_rules = dt_2.predict(dt_prior_rules)

	elif model_type == 'Ablation':
		nn_posterior_rules = []
		for sample_num in range(num_samples):
			# obtain nn posterior rules
			sample_posterior_output = []
			for nn_branch_num in range(len(pred_labels)): 
				value = pred_labels[nn_branch_num][sample_num]
				if len(value) > 1:
					# convert pre-softmax values to class label
					value = torch.argmax(value).item()
				else:
					value = value.item()
				sample_posterior_output.append(value)
			nn_posterior_rules.append(sample_posterior_output)
		dt_posterior_rules = [x[:] for x in nn_posterior_rules]

	surprise_scores = []

	for sample_num in range(num_samples):
		# may use ground truth observed outcome
		if semi_oracle:
			# set default expected to True
			expected = True
			for post_rule_num in range(num_posterior_rules):
				# note that the swapped index ordering is expected and correct
				if dt_posterior_rules[sample_num][post_rule_num] != gt_posterior_rules[post_rule_num][sample_num]:
					expected = False

			if expected:
				surprise_scores.append([0])
			else:
				surprise_scores.append([1])
		# otherwise, compare directly with neural network
		else:
			if list(dt_posterior_rules[sample_num]) == list(nn_posterior_rules[sample_num]):
				surprise_scores.append([0])
			else:
				surprise_scores.append([1])

	return torch.Tensor(surprise_scores)


def train(experiment_id, event_category, model_type, relative_save_path, relative_dataset_path, absolute_data_path, relative_load_model_path = None, use_gpu = True, learning_rate = 0.001,
	num_epochs = 1, batch_size = 2, use_pretrained = False, no_prior = False, combine_feature_prior = False, semi_oracle = False, freeze_pretrained_weights = True,
	optimizer_type = 'adam', dataset_efficiency = 'time'):
	
	if experiment_id == None:
		experiment_id = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + event_category + '_train_' + model_type
	# create directories
	if not os.path.exists(relative_save_path):
		os.makedirs(relative_save_path, exist_ok=True)
	experiment_path = os.path.join(relative_save_path, experiment_id + '/')
	os.makedirs(experiment_path, exist_ok=True)

	# extract data model type made for resnet3d
	if model_type in ['Ablation', 'OF_PR', 'resnet3d_direct']:
		data_model_type = 'resnet3d_extractor'
	else:
		data_model_type = model_type

	# load AVoE data (train and val)
	# memory efficient method reduces the memory stored in the CPU during the training process, however this may slow
	# training if takes a long time to load each video even after the cache stores some of it
	if dataset_efficiency == 'memory':
		(train_dataset,) = load_avoe_data(root = absolute_data_path, model_type = data_model_type, annotation_name = event_category, train = True,
		instance = False, depth = False, dataset_efficiency = 'memory')
		print('\nloading train dataset')
		(val_dataset,) = load_avoe_data(root = absolute_data_path, model_type = data_model_type, annotation_name = event_category, val = True,
		instance = False, depth = False, dataset_efficiency = 'memory')
		print('\nloading val dataset')
	# time efficient method stores the full dataset in the CPU during the training process, this speeds up training as there is no
	# need to load a video from the raw file everytime it is used. However, the dataset may be too big for the CPU to handle
	elif dataset_efficiency == 'time':
		if not os.path.exists(relative_dataset_path + '{}_{}_train_dataset.pt'.format(event_category, data_model_type)):
			(train_dataset,) = load_avoe_data(root = absolute_data_path, model_type = data_model_type, annotation_name = event_category, train = True,
				instance = False, depth = False, dataset_efficiency = 'time')
			print('\nsaving train dataset to disk')
			torch.save(train_dataset, relative_dataset_path + '{}_{}_train_dataset.pt'.format(event_category, data_model_type))
		else:
			print('loading train dataset')
			train_dataset = torch.load(relative_dataset_path + '{}_{}_train_dataset.pt'.format(event_category, data_model_type))

		if not os.path.exists(relative_dataset_path + '{}_{}_val_dataset.pt'.format(event_category, data_model_type)):
			(val_dataset,) = load_avoe_data(root = absolute_data_path, model_type = data_model_type, annotation_name = event_category, val = True,
				instance = False, depth = False, dataset_efficiency = 'time')
			print('\nsaving val dataset to disk')
			torch.save(val_dataset, relative_dataset_path + '{}_{}_val_dataset.pt'.format(event_category, data_model_type))
		else:
			print('loading val dataset')
			val_dataset= torch.load(relative_dataset_path + '{}_{}_val_dataset.pt'.format(event_category, data_model_type))
	
	# set the alpha value. Alpha of 0 --> purely expected videos
	train_dataset.set_alpha(0)

	# number of videos in train and val datasets 
	train_sample_size = len(train_dataset)
	val_sample_size = len(val_dataset)

	# create dataloaders (note that val dataloader batch size is equal to the full dataset size)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	# train_eval_dataloader = DataLoader(train_dataset, batch_size=train_sample_size, shuffle=True, num_workers=0)
	val_dataloader = DataLoader(val_dataset, batch_size=val_sample_size, shuffle=False, num_workers=0) # SHUFFLE MUST BE FALSE

	# get device and load model
	device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
	if relative_load_model_path is not None:
		model = Model(model_type = model_type, device = device, use_pretrained = use_pretrained, 
			freeze_pretrained_weights = freeze_pretrained_weights).to(device)
		model.load_state_dict(torch.load(relative_load_model_path, map_location=device))
		model.device = device
	else:
		model = Model(model_type = model_type, device = device, use_pretrained = use_pretrained,
			freeze_pretrained_weights = freeze_pretrained_weights).to(device)
	# send model to GPU if available
	if model_type == 'resnet3d_direct':
		model.resnet3d_direct.device = device
	elif model_type == 'Ablation':
		model.Ablation.device = device
	elif model_type == 'OF_PR':
		model.OF_PR.device = device
		# train the decision tree classifiers with purely the training dataset
		dt_1, dt_2, dt_3, dt_4 = train_avoe_decision_tree(train_dataset, save_dts = True, experiment_path = experiment_path)
		dt_1_score, dt_2_score, dt_3_score, dt_4_score = eval_avoe_decision_tree(dt_1, dt_2, dt_3, dt_4, val_dataset)

	# # use multiple gpus if available
	# if torch.cuda.device_count() > 1:
	# 	model = nn.DataParallel(model)

	# optimizer and loss functions
	if model_type != 'random':
		if optimizer_type == 'adam':
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		elif optimizer_type == 'sgd':
			optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

	if model_type in ['random', 'resnet3d_direct']:
		bce_loss = nn.BCELoss().to(device) 
	elif model_type == 'OF_PR':
		loss_list = [nn.MSELoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),
				nn.MSELoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),
				nn.MSELoss().to(device),nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device), nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device)]
	elif model_type == 'Ablation':
		loss_list = [nn.CrossEntropyLoss().to(device), nn.CrossEntropyLoss().to(device), nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device), nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device), nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device)]

	# misc variables and initialise logger and add basic info
	max_hit_score = 0
	max_auc_score = 0
	lowest_val_loss = math.inf
	batches_per_epoch = math.ceil(train_sample_size / batch_size)

	experiment_logger = {'train': {}, 'val': {}}
	experiment_logger['event_category'] = event_category
	experiment_logger['model_type'] = model_type
	experiment_logger['experiment_id'] = experiment_id
	experiment_logger['device'] = 'cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu'
	experiment_logger['train_sample_size'] = train_sample_size
	experiment_logger['val_sample_size'] = val_sample_size
	experiment_logger['learning_rate'] = learning_rate
	experiment_logger['num_epochs'] = num_epochs
	experiment_logger['batch_size'] = batch_size
	experiment_logger['use_pretrained'] = use_pretrained
	if model_type == 'OF_PR':
		experiment_logger['val']['dt_1_score'] = dt_1_score
		experiment_logger['val']['dt_2_score'] = dt_2_score
		experiment_logger['val']['dt_3_score'] = dt_3_score
		experiment_logger['val']['dt_4_score'] = dt_4_score

	# save the log once before training
	with open(experiment_path + 'train_val_logdata.json', 'w') as jsonfile:
		json.dump(experiment_logger, jsonfile, indent=4)
	

	for epoch in range(num_epochs):

		experiment_logger['train']['epoch_{}'.format(epoch+1)] = {}
		experiment_logger['train']['epoch_{}'.format(epoch+1)]['batch_losses'] = []
		experiment_logger['val']['epoch_{}'.format(epoch+1)] = {}
		sum_loss = 0
		# set to training mode
		model.train()

		temp_auc_score = 0

		for i, batch in enumerate(train_dataloader):
			# extract data from dataloader and send them to device
			frames, labels, _, features, prior_rules, posterior_rules, *_ = batch
			frames, labels = frames.to(device), labels.to(device)
			features = [x.to(device) for x in features]
			prior_rules = [x.to(device) for x in prior_rules]
			posterior_rules = [x.to(device) for x in posterior_rules]

			pred_labels = model(frames)

			# determine losses
			if model_type in ['random', 'resnet3d_direct']:
				labels = torch.unsqueeze(labels, dim=1).type_as(pred_labels) / 100
				loss = bce_loss(pred_labels, labels)
			elif model_type == 'OF_PR':

				for branch_count, branch_labels in enumerate(pred_labels):
					
					if epoch >= 10 and branch_count not in train_branches_ofpr[event_category]:
						continue

					# extract target from the correct branch set (features or posterior rules)
					if branch_count < 24:
						target = features[branch_count]
					else:
						target = posterior_rules[branch_count - 24]
					# unsqueeze target dimension only for regression loss targets (singular output)
					# the assumption is that all regression outputs have only 1 scalar output
					if branch_labels.size(1) == 1:
						target = torch.unsqueeze(target, dim=1).type_as(branch_labels)
					# cumulate the losses from all 33 branches
					if epoch >= 10:
						if branch_count == train_branches_ofpr[event_category][0]:
							loss = loss_list[branch_count](branch_labels, target)
						else:
							loss += loss_list[branch_count](branch_labels, target)
					else:
						if branch_count == 0:
							loss = loss_list[branch_count](branch_labels, target)
						else:
							loss += loss_list[branch_count](branch_labels, target)
					# ###################################################################################################
					# if branch_count in [0,1,18,20]:
					# 	# print(target[:5], branch_labels[:5])
					# 	if branch_count == 0 and i == 0:
					# 		feat_loss = loss_list[branch_count](branch_labels, target) * frames.size(0)
					# 	else:
					# 		feat_loss += loss_list[branch_count](branch_labels, target) * frames.size(0)
					# ###################################################################################################
					# if branch_count in [25]:
					# 	# print(target[:5], branch_labels[:5])
					# 	if i == 0:
					# 		cat_loss = loss_list[branch_count](branch_labels, target) * frames.size(0)
					# 	else:
					# 		cat_loss += loss_list[branch_count](branch_labels, target) * frames.size(0)

			# # determine the surprise scores based on the predicted features/posterior rules and multi-target decision trees
			# temp_surprise_scores = determine_surprise_scores(dt_1, dt_2, dt_3, dt_4, pred_labels, posterior_rules, 
			# no_prior, combine_feature_prior, semi_oracle, model_type)
			# temp_labels = torch.unsqueeze(labels, dim=1).type_as(temp_surprise_scores) / 100
			# temp_auc_score += calculate_auc(temp_surprise_scores, temp_labels) * len(pred_labels)

			elif model_type == 'Ablation':

				for branch_count, branch_labels in enumerate(pred_labels):
					
					if epoch >= 10 and branch_count not in train_branches_ablation[event_category]:
						continue

					# extract target from the posterior rules set
					target = posterior_rules[branch_count]

					# cumulate the losses from all 9 branches
					if epoch >= 10:
						if branch_count == train_branches_ablation[event_category][0]:
							loss = loss_list[branch_count](branch_labels, target)
						else:
							loss += loss_list[branch_count](branch_labels, target)
					else:
						if branch_count == 0:
							loss = loss_list[branch_count](branch_labels, target)
						else:
							loss += loss_list[branch_count](branch_labels, target)

			# log cumulative loss
			sum_loss += loss.item() * frames.size(0)
			# log the loss
			experiment_logger['train']['epoch_{}'.format(epoch+1)]['batch_losses'].append({
				'batch' : i + 1,
				'train_loss_mean' : loss.item()		
			})

			# backpropagate weights and biases
			if model_type != 'random':
				model.zero_grad()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			# # log training for each epoch
			# if i < batches_per_epoch - 1:
			# 	print('Epoch {} Batch {}/{}'.format(epoch+1,i+1,batches_per_epoch), end = '\r')
			# else:
			# 	print('Epoch {} Batch {}/{}'.format(epoch+1,i+1,batches_per_epoch))

		# # # determine evaluation metrics (hit score and AUC)
		# print("train AUC: ", temp_auc_score / train_sample_size)

		# print("train loss: {}".format(sum_loss / train_sample_size))
		# print("train feat loss: {}".format(feat_loss.item() / train_sample_size))
		# print("train cat loss: {}".format(cat_loss.item() / train_sample_size))
		# log mean loss for this epoch
		experiment_logger['train']['epoch_{}'.format(epoch+1)]['mean_loss'] = sum_loss / train_sample_size
		# set to evaluation mode
		model.eval()

		# test the validation score
		with torch.no_grad():
			# val dataloader has only one batch (the full val dataset)
			# extract data from dataloader and send them to device
			val_dataloader_iter = iter(val_dataloader)
			frames, labels, _, features, prior_rules, posterior_rules, *_ = val_dataloader_iter.next()
			frames, labels = frames.to(device), labels.to(device)
			features = [x.to(device) for x in features]
			prior_rules = [x.to(device) for x in prior_rules]
			posterior_rules = [x.to(device) for x in posterior_rules]

			pred_labels = model(frames)

			# determine losses, surprise scores and labels
			if model_type in ['random', 'resnet3d_direct']:
				# surprise score is the direct output, use simple BCE loss
				surprise_scores = torch.clone(pred_labels)
				labels = torch.unsqueeze(labels, dim=1).type_as(surprise_scores) / 100
				loss = bce_loss(pred_labels, labels)

			elif model_type == 'OF_PR':
				for branch_count, branch_labels in enumerate(pred_labels):

					if epoch >= 10 and branch_count not in train_branches_ofpr[event_category]:
						continue

					# extract target from the correct branch set (features or posterior rules)
					if branch_count < 24:
						target = features[branch_count]
					else:
						target = posterior_rules[branch_count - 24]
					# unsqueeze target dimension only for regression loss targets (singular output)
					# the assumption is that all regression outputs have only 1 scalar output
					if branch_labels.size(1) == 1:
						target = torch.unsqueeze(target, dim=1).type_as(branch_labels)

					###########################################################################
					# cumulate the losses from all 33 branches
					if epoch >= 10:
						if branch_count == train_branches_ofpr[event_category][0]:
							loss = loss_list[branch_count](branch_labels, target)
						else:
							loss += loss_list[branch_count](branch_labels, target)
					else:
						if branch_count == 0:
							loss = loss_list[branch_count](branch_labels, target)
						else:
							loss += loss_list[branch_count](branch_labels, target)

					# ###################################################################################################
					# if branch_count in [0,1,18,20]:
					# 	# print(target[:5], branch_labels[:5])
					# 	if branch_count == 0:
					# 		feat_loss = loss_list[branch_count](branch_labels[0::1], target[0::1])
					# 	else:
					# 		feat_loss += loss_list[branch_count](branch_labels[0::1], target[0::1])
					# ###################################################################################################
					# if branch_count in [25]:
					# 	# print(target[:5], branch_labels[:5])
					# 	cat_loss = loss_list[branch_count](branch_labels[0::1], target[0::1])
				surprise_scores = determine_surprise_scores(dt_1, dt_2, dt_3, dt_4, pred_labels, posterior_rules, no_prior,
					combine_feature_prior, semi_oracle, model_type)
				labels = torch.unsqueeze(labels, dim=1).type_as(surprise_scores) / 100

			elif model_type == 'Ablation':

				for branch_count, branch_labels in enumerate(pred_labels):
					
					if epoch >= 10 and branch_count not in train_branches_ablation[event_category]:
						continue

					# extract target from the posterior rules set
					target = posterior_rules[branch_count]

					# cumulate the losses from all 9 branches
					if epoch >= 10:
						if branch_count == train_branches_ablation[event_category][0]:
							loss = loss_list[branch_count](branch_labels, target)
						else:
							loss += loss_list[branch_count](branch_labels, target)
					else:
						if branch_count == 0:
							loss = loss_list[branch_count](branch_labels, target)
						else:
							loss += loss_list[branch_count](branch_labels, target)	

				surprise_scores = determine_surprise_scores(None, None, None, None, pred_labels, posterior_rules,
					False, False, semi_oracle, model_type)
				labels = torch.unsqueeze(labels, dim=1).type_as(surprise_scores) / 100
				# print("val loss: {}".format(loss.item()))
				# print("val feat loss: {}".format(feat_loss.item()))
				# print("val cat loss: {}".format(cat_loss.item()))

				# determine the surprise scores based on the predicted features/posterior rules and multi-target decision trees

			# determine evaluation metrics (hit score and AUC)
			hit_score = calculate_hit_score(surprise_scores.squeeze().tolist())
			auc_score = calculate_auc(surprise_scores, labels)

			# update max scores and save best models
			if hit_score > max_hit_score:
				max_hit_score = hit_score
				torch.save(model.state_dict(), experiment_path + 'best_model.pth')
				# print('saving model...')
			elif (hit_score == max_hit_score) and (auc_score > max_auc_score):
				max_auc_score = auc_score
				torch.save(model.state_dict(), experiment_path + 'best_model.pth')
				# print('saving model...')
			if auc_score > max_auc_score:
				max_auc_score = auc_score
			# log the auc and hit scores as well as the expected and surprising predictions
			experiment_logger['val']['epoch_{}'.format(epoch+1)]['val_loss'] = loss.item()
			experiment_logger['val']['epoch_{}'.format(epoch+1)]['val_hit_score'] = hit_score
			experiment_logger['val']['epoch_{}'.format(epoch+1)]['val_auc_score'] = auc_score
			experiment_logger['val']['epoch_{}'.format(epoch+1)]['expected_predictions'] = surprise_scores.squeeze().tolist()[0::2]
			experiment_logger['val']['epoch_{}'.format(epoch+1)]['surprising_predictions'] = surprise_scores.squeeze().tolist()[1::2]

			# print('Epoch {}/{} -- Val Hit Score: {:.4f}, Val AUC: {:.4f}'.format(epoch+1, num_epochs, hit_score, auc_score))

			# save the log as json file after every epoch
			with open(experiment_path + 'train_val_logdata.json', 'w') as jsonfile:
				json.dump(experiment_logger, jsonfile, indent=4)

	experiment_logger['val']['max_hit_score'] = max_hit_score
	experiment_logger['val']['max_auc_score'] = max_auc_score

	# save the log as json file after every training
	with open(experiment_path + 'train_val_logdata.json', 'w') as jsonfile:
		data = json.dump(experiment_logger, jsonfile, indent=4)

	return experiment_id

def test(experiment_id, relative_save_path, relative_dataset_path, absolute_data_path, use_gpu = True, no_prior = False, combine_feature_prior = False,
	semi_oracle = False, dataset_efficiency = 'time'):

	# access experiment path
	experiment_path = os.path.join(relative_save_path, experiment_id + '/')

	# access event category and model type (can also access from exp id, but this is safer in case exp id protocol changes)
	with open(experiment_path + 'train_val_logdata.json', 'r') as jsonfile:
		experiment_logger = json.load(jsonfile)
	event_category = experiment_logger['event_category']
	model_type = experiment_logger['model_type']
	use_pretrained = experiment_logger['use_pretrained']

	# extract dataset made for resnet3d
	if model_type in ['Ablation', 'OF_PR']:
		data_model_type = 'resnet3d_extractor'
	else:
		data_model_type = model_type

	# load AVoE data (test)
	if dataset_efficiency == 'memory':
		(test_dataset,) = load_avoe_data(root = absolute_data_path, model_type = data_model_type, annotation_name = event_category, test = True,
			instance = False, depth = False, dataset_efficiency = 'memory')
		print('loading test dataset')

	elif dataset_efficiency == 'time':
		if not os.path.exists(relative_dataset_path + '{}_{}_test_dataset.pt'.format(event_category, data_model_type)):
			(test_dataset,) = load_avoe_data(root = absolute_data_path, model_type = data_model_type, annotation_name = event_category, test = True,
				instance = False, depth = False, dataset_efficiency = 'memory')
			print('saving test dataset to disk')
			torch.save(test_dataset, relative_dataset_path + '{}_{}_test_dataset.pt'.format(event_category, data_model_type))
		else:
			print('loading test dataset')
			test_dataset = torch.load(relative_dataset_path + '{}_{}_test_dataset.pt'.format(event_category, data_model_type))

	# number of videos in test dataset 
	num_videos_test = len(test_dataset)

	# create dataloader for test (full batch size)
	test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0) # SHUFFLE MUST BE FALSE

	# get device and load model
	device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
	model = Model(model_type = model_type, device = device, use_pretrained = use_pretrained).to(device)
	model.load_state_dict(torch.load(experiment_path + 'best_model.pth', map_location=device))
	model.device = device
	if model_type == 'resnet3d_direct':
		model.resnet3d_direct.device = device
	elif model_type == 'Ablation':
		model.Ablation.device = device
	elif model_type == 'OF_PR':
		model.OF_PR.device = device
		# load the decision tree classifiers
		dt_1, dt_2, dt_3, dt_4 = torch.load(experiment_path + 'dt_1.pth'), torch.load(experiment_path + 'dt_2.pth'), \
		torch.load(experiment_path + 'dt_3.pth'), torch.load(experiment_path + 'dt_4.pth')


	# loss function
	if model_type in ['random', 'resnet3d_direct']:
		bce_loss = nn.BCELoss().to(device) 
	elif model_type == 'OF_PR':
		loss_list = [nn.MSELoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),
				nn.MSELoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),nn.MSELoss().to(device),
				nn.MSELoss().to(device),nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device), nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device)]
	elif model_type == 'Ablation':
		loss_list = [nn.CrossEntropyLoss().to(device), nn.CrossEntropyLoss().to(device), nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device), nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device),
				nn.CrossEntropyLoss().to(device), nn.CrossEntropyLoss().to(device),nn.CrossEntropyLoss().to(device)]

	# test_data
	test_logger = {'test': {}}

	model.eval()

	# calculate the test score
	with torch.no_grad():
		print("testing ID: {} ...".format(experiment_id))
		hit_scores = []
		auc_scores = []
		# test dataloader has only one batch (the full test dataset)
		for i, batch in enumerate(test_dataloader):
			# test_dataloader_iter = iter(test_dataloader)
			frames, labels, _, features, prior_rules, posterior_rules, *_ = batch
			frames, labels = frames.to(device), labels.to(device)
			features = [x.to(device) for x in features]
			prior_rules = [x.to(device) for x in prior_rules]
			posterior_rules = [x.to(device) for x in posterior_rules]

			pred_labels = model(frames)

			# determine losses, surprise scores and labels
			if model_type in ['random', 'resnet3d_direct']:
				# surprise score is the direct output, use simple BCE loss
				surprise_scores = torch.clone(pred_labels)
				labels = torch.unsqueeze(labels, dim=1).type_as(surprise_scores) / 100
				loss = bce_loss(pred_labels, labels)
				hit_scores.append(calculate_hit_score(surprise_scores.squeeze().tolist()))
				auc_scores.append(calculate_auc(surprise_scores, labels))

			elif model_type == 'OF_PR':
				for branch_count, branch_labels in enumerate(pred_labels):
					# extract target from the correct branch set (features or posterior rules)
					if branch_count < 24:
						target = features[branch_count]
					else:
						target = posterior_rules[branch_count - 24]
					# unsqueeze target dimension only for regression loss targets (singular output)
					# the assumption is that all regression outputs have only 1 scalar output
					if branch_labels.size(1) == 1:
						target = torch.unsqueeze(target, dim=1).type_as(branch_labels)
					# cumulate the losses from all 33 branches
					if branch_count == 0:
						loss = loss_list[branch_count](branch_labels, target)
					else:
						loss += loss_list[branch_count](branch_labels, target)

				# determine the surprise scores based on the predicted features/posterior rules and multi-target decision trees
				surprise_scores = determine_surprise_scores(dt_1, dt_2, dt_3, dt_4, pred_labels, posterior_rules, no_prior,
					combine_feature_prior, semi_oracle, model_type)
				labels = torch.unsqueeze(labels, dim=1).type_as(surprise_scores) / 100
				hit_scores.append(calculate_hit_score(surprise_scores.squeeze().tolist()))
				auc_scores.append(calculate_auc(surprise_scores, labels))

			elif model_type == 'Ablation':

				for branch_count, branch_labels in enumerate(pred_labels):

					# extract target from the posterior rules set
					target = posterior_rules[branch_count]

					# cumulate the losses from all 9 branches
					if branch_count == 0:
						loss = loss_list[branch_count](branch_labels, target)
					else:
						loss += loss_list[branch_count](branch_labels, target)	

				surprise_scores = determine_surprise_scores(None, None, None, None, pred_labels, posterior_rules, 
					False, False, semi_oracle, model_type)
				labels = torch.unsqueeze(labels, dim=1).type_as(surprise_scores) / 100
				hit_scores.append(calculate_hit_score(surprise_scores.squeeze().tolist()))
				auc_scores.append(calculate_auc(surprise_scores, labels))

		# determine evaluation metrics (hit score and AUC)
		hit_score = sum(hit_scores)/len(hit_scores)
		auc_score = sum(auc_scores)/len(auc_scores)

		# log the auc and hit scores
		test_logger['test']['test_loss'] = loss.item()
		test_logger['test']['hit_score'] = hit_score
		test_logger['test']['auc_score'] = auc_score
		test_logger['test']['expected_predictions'] = surprise_scores.squeeze().tolist()[0::2]
		test_logger['test']['surprising_predictions'] = surprise_scores.squeeze().tolist()[1::2]

		# print test to console
		print('Test Hit Score: {:.4f}, Test AUC: {:.4f}'.format(hit_score, auc_score))

		# save the expt data as json file after every epoch
		with open(experiment_path + 'test_logdata.json', 'w') as jsonfile:
			json.dump(test_logger, jsonfile, indent=4)

if __name__ == '__main__':

	# load yaml config from argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('--config_file', required=True)
	parser.add_argument('--seed', required=False)
	parser.add_argument('--model_type', required=False)
	parser.add_argument('--event_category', required=False)
	parser.add_argument('--neat_expt_id', required=False)
	parser.add_argument('--experiment_id', required=False)
	args = parser.parse_args()
	with open(args.config_file, "r") as configs:
		cfg = yaml.safe_load(configs)

	# load bool for testing and training
	do_train = cfg['TRAIN']
	do_test = cfg['TEST']
	
	if do_train:

		# config for train
		if args.event_category == None:
			event_category = cfg['TRAINING']['EVENT_CATEGORY']
		else: 
			event_category = args.event_category
		if args.model_type == None:
			model_type = cfg['TRAINING']['MODEL_TYPE']
		else:
			model_type = args.model_type
		relative_save_path = cfg['TRAINING']['RELATIVE_SAVE_PATH']
		relative_dataset_path = cfg['TRAINING']['RELATIVE_DATASET_PATH']
		absolute_data_path = cfg['TRAINING']['ABSOLUTE_DATA_PATH']
		relative_load_model_path = cfg['TRAINING']['RELATIVE_LOAD_MODEL_PATH']
		decision_tree_type = cfg['TRAINING']['DECISION_TREE_TYPE']
		semi_oracle = cfg['TRAINING']['SEMI_ORACLE']
		use_gpu = cfg['TRAINING']['USE_GPU']
		learning_rate = cfg['TRAINING']['LEARNING_RATE']
		num_epochs = cfg['TRAINING']['NUM_EPOCHS']
		batch_size = cfg['TRAINING']['BATCH_SIZE']
		use_pretrained = cfg['TRAINING']['USE_PRETRAINED']
		freeze_pretrained_weights = cfg['TRAINING']['FREEZE_PRETRAINED_WEIGHTS']
		if args.seed == None:
			random_seed = cfg['TRAINING']['RANDOM_SEED']
		else:
			random_seed = int(args.seed)
		optimizer_type = cfg['TRAINING']['OPTIMIZER_TYPE']
		dataset_efficiency = cfg['TRAINING']['DATASET_EFFICIENCY']

		# assertions for configs
		assert event_category in ['A', 'B', 'C', 'D', 'E', 'combined']
		assert model_type in ['resnet3d_direct', 'random', 'OF_PR', 'Ablation']
		assert num_epochs > 0 and type(num_epochs) == int
		assert batch_size > 1 and type(batch_size) == int
		assert relative_load_model_path == None or type(relative_load_model_path) == str
		assert learning_rate > 0
		assert decision_tree_type in ["normal", "direct", "combined"]
		assert type(semi_oracle) == bool
		assert type(use_gpu) == bool
		assert type(use_pretrained) == bool
		assert type(freeze_pretrained_weights) == bool
		assert optimizer_type in ['adam', 'sgd']
		assert dataset_efficiency  in ['time', 'memory']
		# check paths and ensure they end with a '/'
		paths = [relative_save_path, relative_dataset_path, absolute_data_path, relative_load_model_path]
		for i, path in enumerate(paths):
			if path != None and path[-1] != '/':
				paths[i] += '/'
		# check paths and assert that they exist (better to manually create to avoid other issues)
		for path in paths:
			if path != None:
				assert os.path.exists(path)
		relative_save_path, relative_dataset_path, absolute_data_path, relative_load_model_path = paths	

		# decision tree type
		if decision_tree_type == "normal":
			no_prior = False
			combine_feature_prior = False
		elif decision_tree_type == "direct":
			no_prior = True
			combine_feature_prior = False
		elif decision_tree_type == "combined":
			no_prior = False
			combine_feature_prior = True

		# set seed values
		os.environ['PYTHONHASHSEED'] = str(random_seed)
		torch.manual_seed(random_seed)
		np.random.seed(random_seed)
		random.seed(random_seed)

		if args.neat_expt_id == None or bool(args.neat_expt_id) == False:
			experiment_id = None
		elif bool(args.neat_expt_id) == True:
			experiment_id = '{}_{}_seed_{}'.format(event_category, model_type, random_seed)
		else:
			raise ValueError("invalid --neat_expt_id argument (must be True or False)")

		# train the model
		experiment_id = train(
			experiment_id = experiment_id,
			event_category = event_category,
			model_type = model_type,
			relative_save_path = relative_save_path,
			relative_dataset_path = relative_dataset_path,
			absolute_data_path = absolute_data_path,
			relative_load_model_path = relative_load_model_path,
			learning_rate = learning_rate,
			use_gpu = use_gpu,
			num_epochs = num_epochs,
			batch_size = batch_size,
			use_pretrained = use_pretrained,
			no_prior = no_prior,
			combine_feature_prior = combine_feature_prior,
			semi_oracle = semi_oracle,
			freeze_pretrained_weights = freeze_pretrained_weights,
			optimizer_type = optimizer_type,
			dataset_efficiency = dataset_efficiency
		)

	if do_test:

		if args.experiment_id != None:
			experiment_id = args.experiment_id

		# config for test

		# if training just occured, overwrite it with the new experiment id, otherwise use what is in config
		try:
			experiment_id
			print("using existing experiment ID ...")
		except:
			experiment_id = cfg['TESTING']['EXPERIMENT_ID']				
		relative_save_path = cfg['TESTING']['RELATIVE_SAVE_PATH']
		relative_dataset_path = cfg['TESTING']['RELATIVE_DATASET_PATH']
		absolute_data_path = cfg['TESTING']['ABSOLUTE_DATA_PATH']
		decision_tree_type = cfg['TESTING']['DECISION_TREE_TYPE']
		semi_oracle = cfg['TRAINING']['SEMI_ORACLE']
		use_gpu = cfg['TESTING']['USE_GPU']
		dataset_efficiency = cfg['TESTING']['DATASET_EFFICIENCY']

		# assertions for configs
		assert experiment_id is not None
		assert decision_tree_type in ["normal", "direct", "combined"]
		assert type(semi_oracle) == bool
		assert type(use_gpu) == bool
		assert dataset_efficiency  in ['time', 'memory']
		# check paths and ensure they end with a '/'
		paths = [relative_save_path, relative_dataset_path, absolute_data_path]
		for i, path in enumerate(paths):
			if path[-1] != '/':
				paths[i] += '/'
		# check paths and assert that they exist (better to manually create to avoid other issues)
		for path in paths:
			if path != None:
				assert os.path.exists(path)
		relative_save_path, relative_dataset_path, absolute_data_path = paths	


		# decision tree type
		if decision_tree_type == "normal":
			no_prior = False
			combine_feature_prior = False
		elif decision_tree_type == "direct":
			no_prior = True
			combine_feature_prior = False
		elif decision_tree_type == "combined":
			no_prior = False
			combine_feature_prior = True

		# test the model
		test(
			experiment_id = experiment_id,
			relative_save_path = relative_save_path,
			relative_dataset_path = relative_dataset_path,
			absolute_data_path = absolute_data_path,
			use_gpu = use_gpu,
			no_prior = no_prior,
			combine_feature_prior = combine_feature_prior,
			semi_oracle = semi_oracle,
			dataset_efficiency = dataset_efficiency
		)