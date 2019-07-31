import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor

from loader.A2DClsLoader import A2DClassification


def p_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="a2d", type=str)
    # data
    parser.add_argument(
        "--a2d_root", default='/mnt/lustre/jiangsu/dlar/home/zyk17/data/A2D/Release', type=str)
    parser.add_argument('--csv_path', default='./repo/newVideoSet.csv')
    # config
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    # train
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    # save
    parser.add_argument("--save_weights", default='save/weights', type=str)

    args = parser.parse_args()

    return args


def main():
    train_img_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = A2DClassification(args, train_img_transform)
    for iter, pack in enumerate(loader):
        imgs = pack[0]  # (N,t,c,m,n)
        labels = pack[1]  # (N,t,c,m,n)


if __name__ == '__main__':
    global args
    args = p_parse()
    main()
