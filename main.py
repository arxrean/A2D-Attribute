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
from utils.opt import get_finetune_optimizer


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
    # config
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--cuda", default=False, type=bool)
    parser.add_argument("--pretrained", default=True, type=bool)
    # train
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--lr_dec", default=1, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    # save
    parser.add_argument("--save_root", default='./save/', type=str)

    args = parser.parse_args()

    return args


def joint_classification(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
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

    train_dataset = A2DClassification(args, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                              pin_memory=True, drop_last=True, shuffle=True)

    val_dataset = A2DClassification(args, val_transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0,
                            pin_memory=True, drop_last=False, shuffle=False)

    model = getJointClassifier(args)
    if args.cuda:
        model = model.cuda()

    criterion = None
    if args.cuda:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(
            np.load('./repo/joint_label_frac.npy')).type(torch.FloatTensor).cuda())
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(
            np.load('./repo/joint_label_frac.npy')).type(torch.FloatTensor))

    train_loss = []
    val_loss = []
    for epoch in range(args.max_epoches):
        opt = get_finetune_optimizer(args, model, epoch)

        train_loss.append(0)
        val_loss.append(0)
        for iter, pack in enumerate(train_loader):
            imgs = pack[1]  # (N,t,c,m,n)
            labels = pack[2]  # (N,t,c,m,n)

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            out, fc = model(imgs)
            loss = criterion(out, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss[-1] += loss.item()

        with torch.no_grad():
            for iter, pack in enumerate(val_loader):
                imgs = pack[1]  # (N,t,c,m,n)
                labels = pack[2]  # (N,t,c,m,n)

                out, fc = model(imgs)
                loss = criterion(out, labels)

                val_loss[-1] += loss.item()

        print('epoch:{} train_loss:{} val_loss:{}'.format(
            epoch, train_loss[-1], val_loss[-1]))
        snap_shot = {'epoch': epoch, 'train_loss': train_loss,
                     'val_loss': val_loss, 'state_dict': model.state_dict()}
        np.save('./save/joint_classification/snap/snap_{}/npy'.format(epoch), snap_shot)


if __name__ == '__main__':
    args = p_parse()
    joint_classification(args)
