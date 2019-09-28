import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from loader.A2DClsLoader import A2DClassification
from model.net import getJointClassifier
from glob import p_parse

from keras.utils import to_categorical


def gen_joint_feature(args):
	val_transform = transforms.Compose([
		transforms.Resize((args.input_size, args.input_size)),
		transforms.ToTensor(),
		transforms.Normalize([0.4569, 0.4335, 0.3892],
							 [0.2093, 0.2065, 0.2046])
	])

	train_dataset = A2DClassification(args, val_transform, mode='train')
	train_loader = DataLoader(train_dataset, batch_size=64, num_workers=args.num_workers,
							  pin_memory=True, drop_last=False, shuffle=False)

	val_dataset = A2DClassification(args, val_transform, mode='val')
	val_loader = DataLoader(val_dataset, batch_size=64, num_workers=args.num_workers,
							pin_memory=True, drop_last=False, shuffle=False)

	model = getJointClassifier(args)
	model.load_state_dict(torch.load(os.path.join(
		args.save_root, 'joint_classification/snap/snap_22.pth.tar'), map_location='cpu')['state_dict'])
	if args.cuda:
		model = model.cuda()

	res = {}
	with torch.no_grad():
		for iter, pack in enumerate(train_loader):
			id = pack[0]
			imgs = pack[1]  # (N,t,c,m,n)
			labels = pack[2]  # (N,t,c,m,n)

			if args.cuda:
				imgs = imgs.cuda()
				labels = labels.cuda()

			fc = model.gen_feature(imgs).detach().cpu().numpy()
			for s_id in id:
				res[s_id] = [fc[id.index(s_id)]]

		for iter, pack in enumerate(val_loader):
			id = pack[0]
			imgs = pack[1]  # (N,t,c,m,n)
			labels = pack[2]  # (N,t,c,m,n)

			if args.cuda:
				imgs = imgs.cuda()
				labels = labels.cuda()

			fc = model.gen_feature(imgs).detach().cpu().numpy()

			for s_id in id:
				res[s_id] = [fc[id.index(s_id)]]

	np.save('./repo/joint_img_feature_mono.npy', res)


def gen_joint_feature_aug(args):
	train_transform = transforms.Compose([
		transforms.Resize((args.input_size, args.input_size)),
		# transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.4569, 0.4335, 0.3892],
							 [0.2093, 0.2065, 0.2046])
	])

	train_hp_transform = transforms.Compose([
		transforms.Resize((args.input_size, args.input_size)),
		transforms.RandomHorizontalFlip(1.0),
		transforms.ToTensor(),
		transforms.Normalize([0.4569, 0.4335, 0.3892],
							 [0.2093, 0.2065, 0.2046])
	])

	val_transform = transforms.Compose([
		transforms.Resize((args.input_size, args.input_size)),
		transforms.ToTensor(),
		transforms.Normalize([0.4569, 0.4335, 0.3892],
							 [0.2093, 0.2065, 0.2046])
	])

	train_dataset = A2DClassification(args, train_transform, mode='train')
	train_loader = DataLoader(train_dataset, batch_size=64, num_workers=args.num_workers,
							  pin_memory=True, drop_last=False, shuffle=False)

	train_hp_dataset = A2DClassification(
		args, train_hp_transform, mode='train')
	train_hp_loader = DataLoader(train_hp_dataset, batch_size=64, num_workers=args.num_workers,
								 pin_memory=True, drop_last=False, shuffle=False)

	val_dataset = A2DClassification(args, val_transform, mode='val')
	val_loader = DataLoader(val_dataset, batch_size=64, num_workers=args.num_workers,
							pin_memory=True, drop_last=False, shuffle=False)

	model = getJointClassifier(args)
	# model.load_state_dict(torch.load(os.path.join(
	# 	args.save_root, 'joint_classification/snap/snap_31.pth.tar'), map_location='cpu')['state_dict'])
	if args.cuda:
		model = model.cuda()

	res = {}
	with torch.no_grad():
		for iter, pack in enumerate(train_loader):
			id = pack[0]
			imgs = pack[1]  # (N,t,c,m,n)
			labels = pack[2]  # (N,t,c,m,n)

			if args.cuda:
				imgs = imgs.cuda()
				labels = labels.cuda()

			fc = model.gen_feature(imgs).detach().cpu().numpy()
			for s_id in id:
				res[s_id] = [fc[id.index(s_id)]]

		for iter, pack in enumerate(val_loader):
			id = pack[0]
			imgs = pack[1]  # (N,t,c,m,n)
			labels = pack[2]  # (N,t,c,m,n)

			if args.cuda:
				imgs = imgs.cuda()
				labels = labels.cuda()

			fc = model.gen_feature(imgs).detach().cpu().numpy()

			for s_id in id:
				res[s_id] = [fc[id.index(s_id)]]

		for iter, pack in enumerate(train_hp_loader):
			id = pack[0]
			imgs = pack[1]  # (N,t,c,m,n)
			labels = pack[2]  # (N,t,c,m,n)

			if args.cuda:
				imgs = imgs.cuda()
				labels = labels.cuda()

			fc = model.gen_feature(imgs).detach().cpu().numpy()
			for s_id in id:
				res[s_id].append(fc[id.index(s_id)])

	np.save('./repo/joint_img_feature.npy', res)


if __name__ == '__main__':
	args = p_parse()
	# gen_joint_feature(args)
	gen_joint_feature_aug(args)
