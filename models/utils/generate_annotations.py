"""
This file generates all frames of the AVoE dataset from every mp4 video
It also generates the annotation.txt file needed for Video Data Loading
"""

import cv2
import random
random.seed(42) # use seed value 42
import numpy as np

#~~~~~~~~~~ SETTINGS ~~~~~~~~~~~#
# for processing testing videos
root_path = ''
generate_frames = False
generate_annotations = True
annotations_event_categories = ['A_support', 'B_occlusion', 'C_container', 'D_collision', 'E_barrier'] # DATASET ANNOTATIONS
random_sample = True # sample the annotations evenly from all the event categories mentioned (Total annotation size is equivalent to one event category)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

annotation_name = "annotations_" + annotations_event_categories[0][0]

if generate_frames:
	video_paths = []
	# generate frames for all these event categories
	video_event_categories = ['A_support', 'B_occlusion', 'C_container', 'D_collision', 'E_barrier']
	# video_event_categories = ['A_support', 'C_container']

	# Test videos
	for ec in video_event_categories:
		ec_path = root_path + ec + '/'
		segment_path = ec_path + 'test/expected/'
		for num in range(4501,4551):
			video_paths.append(segment_path + 'trial_{}/rgb.avi'.format(num))
		segment_path = ec_path + 'test/surprising/'
		for num in range(4501,4551):
			video_paths.append(segment_path + 'trial_{}/rgb.avi'.format(num))

	# Val videos
	for ec in video_event_categories:
		ec_path = root_path + ec + '/'
		segment_path = ec_path + 'validation/expected/'
		for num in range(3751,3826):
			video_paths.append(segment_path + 'trial_{}/rgb.avi'.format(num))
		segment_path = ec_path + 'validation/surprising/'
		for num in range(3751,3826):
			video_paths.append(segment_path + 'trial_{}/rgb.avi'.format(num))

	# Train videos
	for ec in video_event_categories:
		ec_path = root_path + ec + '/'
		# test (expected)
		segment_path = ec_path + 'train/expected/'
		for num in range(1,376):
			video_paths.append(segment_path + 'trial_{}/rgb.avi'.format(num))
		# test (surprising)
		segment_path = ec_path + 'train/surprising/'
		for num in range(1,376):
			video_paths.append(segment_path + 'trial_{}/rgb.avi'.format(num))

	num_video = len(video_paths)
	for i, path in enumerate(video_paths):
		vidcap = cv2.VideoCapture(path)
		success, image = vidcap.read()
		count = 0
		while success:
			cv2.imwrite(path[:-7] + "rgb_{:02d}.jpg".format(count + 1), image)   # save frame as jpg file      
			success, image = vidcap.read()
			count += 1
		print("Frames Generated for {}/{} Videos".format(i+1, num_video), end = '\r')

if generate_annotations:
	train_annotations = []
	val_annotations = []
	test_annotations = []

	full_test_range = list(range(4501,4551))
	full_val_range = list(range(3751,3826))
	full_train_range = list(range(1,376))

	# sample evenly from all event categories instead of using all of them
	if random_sample:
		fraction_to_sample = 1  / len(annotations_event_categories)

	# Test annotations
	for ec in annotations_event_categories:

		# sample for every ec (but for test set, give the full set for combined dataset)
		if random_sample and len(annotations_event_categories) < 5:
			test_range = random.choices(full_test_range, k = int(fraction_to_sample * len(full_test_range)))
		else:
			test_range = full_test_range[:]

		relative_ec_path = ec + '/'
		for num in test_range:
			test_annotations.append("{}trial_{} 0".format(relative_ec_path + 'test/expected/', num)) 
			test_annotations.append("{}trial_{} 100".format(relative_ec_path + 'test/surprising/', num))

	f = open("annotations/" + annotation_name + '_test.txt', 'w')
	for line in test_annotations:
		f.write(line + '\n')
	f.close()
	
	# Val annotations
	for ec in annotations_event_categories:

		# sample for every ec
		if random_sample:
			val_range = random.choices(full_val_range, k = int(fraction_to_sample * len(full_val_range)))

		relative_ec_path = ec + '/'
		for num in val_range:
			val_annotations.append("{}trial_{} 0".format(relative_ec_path + 'validation/expected/', num)) 
			val_annotations.append("{}trial_{} 100".format(relative_ec_path + 'validation/surprising/', num))

	f = open("annotations/" + annotation_name + '_val.txt', 'w')
	for line in val_annotations:
		f.write(line + '\n')
	f.close()

	# Train annotations
	for ec in annotations_event_categories:

		# sample for every ec
		if random_sample:
			train_range = random.choices(full_train_range, k = int(fraction_to_sample * len(full_train_range)))

		relative_ec_path = ec + '/'
		for num in train_range:
			train_annotations.append("{}trial_{} 0".format(relative_ec_path + 'train/expected/', num)) 
			train_annotations.append("{}trial_{} 100".format(relative_ec_path + 'train/surprising/', num))

	# only training videos are shuffled
	random.shuffle(train_annotations)

	f = open("annotations/" + annotation_name + '_train.txt', 'w')
	for line in train_annotations:
		f.write(line + '\n')
	f.close()
	# print("{:02d}".format(55))
