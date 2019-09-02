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

    val_dataset = A2DClassification(args, val_transform, mode='train')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0,
                            pin_memory=True, drop_last=False, shuffle=False)

    model = getJointClassifier(args)
    model.load_state_dict(torch.load(os.path.join(
        args.save_root, 'joint_classification/snap_25.pth.tar'), map_location='cpu')['state_dict'])
    if args.cuda:
        model = model.cuda()

    total_id = []
    total_res = []
    with torch.no_grad():
        for iter, pack in enumerate(val_loader):
            id = pack[0]
            imgs = pack[1]  # (N,t,c,m,n)
            labels = pack[2]  # (N,t,c,m,n)

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            fc = model.gen_feature(imgs)

            total_id.append(id)
            total_res.append(fc.detach().cpu().numpy())

    total_id = np.concatenate(total_id, axis=0)
    total_res = np.concatenate(total_res, axis=0)
    np.save('./repo/joint_img_feature.npy', {'id': total_id, 'res': total_res})


if __name__ == '__main__':
    args = p_parse()
    gen_joint_feature(args)
