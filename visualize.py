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
import shutil

from loader.A2DClsLoader import A2DClassification
from model.net import getJointClassifier
from model.gradCAM import GradCam
from utils.helper import show_cam_on_image, np_sigmoid

from keras.utils import to_categorical


def p_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument("--session_name", default="a2d", type=str)
	# data
	parser.add_argument(
		"--a2d_root", default='/mnt/lustre/jiangsu/dlar/home/zyk17/data/A2D/Release', type=str)
	parser.add_argument('--csv_path', default='./repo/newVideoSet.csv')
	parser.add_argument('--t', type=int, default=0)
	parser.add_argument('--input_size', type=int, default=224)
	parser.add_argument('--class_num', type=int, default=43)
	parser.add_argument('--actor_num', type=int, default=7)
	parser.add_argument('--action_num', type=int, default=9)
	# config
	parser.add_argument("--num_workers", default=8, type=int)
	parser.add_argument("--batch_size", default=64, type=int)
	parser.add_argument("--cuda", default=False, type=bool)
	parser.add_argument("--pretrained", default=True, type=bool)
	# save
	parser.add_argument("--save_root", default='./save/', type=str)

	args = parser.parse_args()

	return args


def joint_plot(args, thd=0.669):
	dir_path = os.path.join(
		args.save_root, 'joint_classification', 'imgs', 'res_samples')

	shutil.rmtree(dir_path)
	os.makedirs(dir_path)
	os.makedirs(os.path.join(dir_path, '1'))
	os.makedirs(os.path.join(dir_path, '0'))
	os.makedirs(os.path.join(dir_path, '10'))

	val_transform = transforms.Compose([
		transforms.Resize((args.input_size, args.input_size)),
		transforms.ToTensor(),
		transforms.Normalize([0.4569, 0.4335, 0.3892],
							 [0.2093, 0.2065, 0.2046])
	])

	val_dataset = A2DClassification(args, val_transform, mode='val')
	val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0,
							pin_memory=True, drop_last=False, shuffle=False)

	model = getJointClassifier(args)
	model.load_state_dict(torch.load(os.path.join(
		args.save_root, 'joint_classification/snap_25.pth.tar'), map_location='cpu')['state_dict'])
	grad_model = GradCam(model=model, target_layer_names=[
						 "7"], args=args)

	for iter, pack in enumerate(val_loader):
		if iter==300:
			break
		part_path = pack[0][-1]
		imgs = pack[1]  # (N,t,c,m,n)
		labels = pack[2]  # (N,t,c,m,n)

		# 1: label 1 res 1
		# 0: label 1 res 0
		# 10: label 0 res 1
		gt = np.where(labels.squeeze().cpu().numpy() == 1)[0]
		res = None
		for index in gt:
			mask, output = grad_model(imgs, index)
			output = torch.sigmoid(output).detach().cpu().numpy()
			output[output >= thd] = 1
			output[output < thd] = 0
			show_cam_on_image(args, os.path.join(args.a2d_root, part_path), mask, os.path.join(dir_path, '1', part_path.split(
				'/')[-1][:-4]+'_'+str(index)+'.jpg') if output[index] == 1 else os.path.join(dir_path, '0', part_path.split('/')[-1][:-4]+'_'+str(index)+'.jpg'))

			res = output

		res = np.where(output == 1)[0]
		for index in res:
			if labels.squeeze()[index] == 0:
				mask, output = grad_model(imgs, index)
				show_cam_on_image(args, os.path.join(args.a2d_root, part_path), mask, os.path.join(
					dir_path, '10', part_path.split('/')[-1][:-4]+'_'+str(index)+'.jpg'))


if __name__ == '__main__':
	joint_plot(p_parse())
