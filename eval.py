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

from utils.helper import get_eval
from loader.A2DClsLoader import A2DClassification, A2DClassificationWithActorAction
from model.net import getJointClassifier, getSplitClassifier

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
    # config
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_epoches", default=99999999, type=int)
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
        args.save_root, 'joint_classification/best.pth.tar'), map_location='cpu')['state_dict'])

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
    get_eval(total_res, total_label)


def eval_split_classification(args):
    valid = {11: 0, 12: 1, 13: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 7, 21: 8, 
             22: 9, 26: 10, 28: 11, 29: 12, 34: 13, 35: 14, 36: 15, 39: 16,
             41: 17, 43: 18, 44: 19, 45: 20, 46: 21, 48: 22, 49: 23, 54: 24,
             55: 25, 56: 26, 57: 27, 59: 28, 61: 29, 63: 30, 65: 31, 66: 32,
             67: 33, 68: 34, 69: 35, 72: 36, 73: 37, 75: 38, 76: 39, 77: 40,
             78: 41, 79: 42}
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.4569, 0.4335, 0.3892],
                             [0.2093, 0.2065, 0.2046])
    ])

    val_dataset = A2DClassificationWithActorAction(args, val_transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0,
                            pin_memory=True, drop_last=False, shuffle=False)

    model = getSplitClassifier(args)
    if args.cuda:
        model = model.cuda()

    #need to be modified
    model.load_state_dict(torch.load(os.path.join(
        args.save_root, 'joint_classification/best.pth.tar'), map_location='cpu')['state_dict'])
    
    total_res = []
    total_label = []
    with torch.no_grad():
        for iter, pack in enumerate(val_loader):
            imgs = pack[0]  # (N,t,c,m,n)
            labels = pack[1]
            #actor_labels = pack[2]
            #action_labels = pack[3]

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
                #actor_labels = actor_labels.cuda()
                #action_labels = action_labels.cuda()

            actor_out, action_out = model(imgs)
            actor_out = F.sigmoid(actor_out)
            action_out = F.sigmoid(action_out)
            sample_num = len(actor_out)
            actor_action = []
            for i in range(sample_num):
                actor_action.append(
                    np.outer(actor_out.detach().cpu().numpy()[i],action_out.detach().cpu().numpy()[i]).tolist())
            actor_action = np.array(actor_action)

            actor_action_labels = []
            for sample in actor_action:
                sample_labels = []
                row_num = 1
                column_num = 1
                for row in sample:
                    for item in row:
                        position = row_num * 10 + column_num
                        if position in valid and item == 1:
                            sample_labels.append(valid[position])
                            #actor_action_label.append(valid[position])
                        column_num += 1
                    row_num += 1
                sample_labels = to_categorical(sample_labels, num_classes=len(valid))
                sample_labels = np.array(sample_labels)
                sample_labels = sample_labels.sum(axis=0)
                sample_labels = np.where(sample_labels > 1, 1, sample_labels)
                actor_action_labels.append(sample_labels)

            total_res.append(actor_action_labels)
            total_label.append(labels.cpu().numpy())

    total_res = np.array(total_res)
    total_label = np.array(total_label)    
    total_res = np.concatenate(total_res, axis=0)
    total_label = np.concatenate(total_label, axis=0)
    get_eval(total_res, total_label)

if __name__ == '__main__':
    args = p_parse()
    # eval_joint_classification(args)
