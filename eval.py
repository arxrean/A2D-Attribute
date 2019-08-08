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

from utils.helper import Precision, Recall, F1
from loader.A2DClsLoader import A2DClassification
from model.net import getJointClassifier


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
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--cuda", default=False, type=bool)
    parser.add_argument("--pretrained", default=True, type=bool)
    # save
    parser.add_argument("--save_root", default='./save/', type=str)

    args = parser.parse_args()

    return args


def eval_joint_classification(args):
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.4569, 0.4335, 0.3892],
                             [0.2093, 0.2065, 0.2046])
    ])

    val_dataset = A2DClassification(args, val_transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0,
                            pin_memory=True, drop_last=False, shuffle=False)

    model = getJointClassifier(args)
    if args.cuda:
        model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(
        args.save_root, 'joint_classification/snap/snap_19.pth.tar')['state_dict'], map_location='cpu'))

    total_res = []
    total_label = []
    with torch.no_grad():
        for iter, pack in enumerate(val_loader):
            imgs = pack[1]  # (N,t,c,m,n)
            labels = pack[2]  # (N,t,c,m,n)

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            out, fc = model(imgs)
            out = F.sigmoid(out)
            total_res.append(out.detach().cpu().numpy())
            total_label.append(labels.cpu().numpy())

    total_res = np.concatenate(total_res, axis=0)
    total_label = np.concatenate(total_label, axis=0)
    P = Precision(total_res, total_label)
    R = Recall(total_res, total_label)
    F = F1(total_res, total_label)
    print('Precision: {:.1f} Recall: {:.1f} F1: {:.1f}'.format(
        100 * P, 100 * R, 100 * F))


if __name__ == '__main__':
    args = p_parse()
    eval_joint_classification(args)
