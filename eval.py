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
import pdb

from loader.A2DClsLoader import A2DClassification, A2DClassificationWithActorAction
from loader.A2DCompositionLoader import A2DComposition
from model.net import getJointClassifier, getSplitClassifier, ManifoldModel
from model.net import ManifoldModel
from glob import p_parse

from keras.utils import to_categorical

from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt


def Precision(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x*y)/(np.sum(x) + 1e-8)
    return p/N


def meanAveragePrecision(X_pre, X_gt):
    sample_num = len(X_pre)
    class_num = len(X_pre[0])

    ave_pre = []
    for index_class in range(class_num):
        x = []
        y = []
        for index_sample in range(sample_num):
            x.append(X_pre[index_sample][index_class])
            y.append(X_gt[index_sample][index_class])
        ave_pre.append(average_precision_score(y_true=y, y_score=x))

    return np.sum(ave_pre)/class_num


def Recall(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(y)
    return p/N


def F1(X_pre, X_gt):
    N = len(X_pre)
    p = 0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += 2*np.sum(x * y) / (np.sum(x) + np.sum(y))
    return p/N


def get_eval(X_pre, X_gt, mode='mAP'):
    if mode == 'mAP':
        res = meanAveragePrecision(X_pre, X_gt)
    elif mode == 'F1':
        res = F1(X_pre, X_gt)
    else:
        raise RuntimeError('mode must be mAP or F1')  
    return res


def eval_joint_classification(args, model_path='joint_classification/snap/'):
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.4569, 0.4335, 0.3892],
                             [0.2093, 0.2065, 0.2046])
    ])

    val_dataset = A2DClassification(args, val_transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0,
                            pin_memory=True, drop_last=False, shuffle=False)

    snapList = os.listdir(os.path.join(args.save_root, model_path))
    snapList.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
    mapList = []
    bestmodelNum = None
    best_mAP = None

    for snap in snapList:
        model = getJointClassifier(args)
        if args.cuda:
            model = model.cuda()
        model.load_state_dict(torch.load(os.path.join(
            args.save_root, model_path, snap), map_location='cpu')['state_dict'])

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

        mAP = get_eval(total_res, total_label)
        print('snap:{} mAP:{}'.format(snap, mAP))
        mapList.append(mAP)
        if best_mAP is None or mAP > best_mAP:
            best_mAP = mAP
            bestmodelNum = int(snap.split('.')[0].split('_')[1])
            
    print('best mAP:{}'.format(best_mAP))
    print('best model:{}'.format('snap_'+str(bestmodelNum)))
    plt.figure()
    plt.plot(range(len(mapList)), mapList, label='joint_model mAP')
    plt.legend()
    if not os.path.exists('./save/joint_classification/imgs'):
        os.makedirs('./save/joint_classification/imgs')
    plt.savefig('./save/joint_classification/imgs/snaps_eval_mAP.png')
    plt.close()


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

    val_dataset = A2DClassificationWithActorAction(
        args, val_transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0,
                            pin_memory=True, drop_last=False, shuffle=False)

    model = getSplitClassifier(args)
    if args.cuda:
        model = model.cuda()

    # need to be modified
    model.load_state_dict(torch.load(os.path.join(
        args.save_root, 'split_classification/snap/snap_329.pth.tar'), map_location='cpu')['state_dict'])

    total_res = []
    total_label = []
    with torch.no_grad():
        for iter, pack in enumerate(val_loader):
            imgs = pack[0]  # (N,t,c,m,n)
            labels = pack[1]
            actor_labels = pack[2]
            action_labels = pack[3]

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
                actor_labels = actor_labels.cuda()
                action_labels = action_labels.cuda()

            actor_out, action_out = model(imgs)
            actor_out = F.sigmoid(actor_out)
            action_out = F.sigmoid(action_out)

            sample_num = len(actor_out)
            actor_action = []
            for i in range(sample_num):
                actor_action.append(
                    np.outer(actor_out.detach().cpu().numpy()[i], action_out.detach().cpu().numpy()[i]).tolist())

            actor_action_labels = []
            for sample in actor_action:
                sample_labels = np.zeros(len(valid)).tolist()

                for position in range(len(sample_labels)):
                    key = list(valid.keys())[
                        list(valid.values()).index(position)]
                    sample_labels[position] = sample[key//10 - 1][key % 10 - 1]

                actor_action_labels.append(sample_labels)

            total_res.append(actor_action_labels)
            total_label.append(labels.cpu().numpy())

    total_res = np.array(total_res)
    total_label = np.array(total_label)
    total_res = np.concatenate(total_res, axis=0)
    total_label = np.concatenate(total_label, axis=0)
    get_eval(total_res, total_label)


def eval_actor_or_action_classification(args):
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.4569, 0.4335, 0.3892],
                             [0.2093, 0.2065, 0.2046])
    ])

    val_dataset = A2DClassificationWithActorAction(
        args, val_transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=0,
                            pin_memory=True, drop_last=False, shuffle=False)

    model = getSplitClassifier(args)
    if args.cuda:
        model = model.cuda()

    # need to be modified
    model.load_state_dict(torch.load(os.path.join(
        args.save_root, 'split_classification/snap/snap_329.pth.tar'), map_location='cpu')['state_dict'])

    total_res = []
    total_label = []
    with torch.no_grad():
        for iter, pack in enumerate(val_loader):
            imgs = pack[0]  # (N,t,c,m,n)
            labels = pack[1]
            actor_labels = pack[2]
            action_labels = pack[3]

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
                actor_labels = actor_labels.cuda()
                action_labels = action_labels.cuda()

            actor_out, action_out = model(imgs)
            actor_out = F.sigmoid(actor_out)
            action_out = F.sigmoid(action_out)

            # total_res.append(actor_out.detach().cpu().numpy())
            # total_label.append(actor_labels.cpu().numpy())
            total_res.append(action_out.detach().cpu().numpy())
            total_label.append(action_labels.cpu().numpy())

    total_res = np.concatenate(total_res, axis=0)
    total_label = np.concatenate(total_label, axis=0)
    get_eval(total_res, total_label)


def eval_composition_1(args, model_path='./composition_train/snap/'):
    val_dataset = A2DComposition(args, None, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0,
                            pin_memory=True, drop_last=False, shuffle=False)

    snapList = os.listdir(os.path.join(args.save_root, model_path))
    snapList.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
    eval_res_List = []
    mode = 'mAP'
    bestmodelNum = None
    best_eval_res = None
    for snap in snapList:
        model = ManifoldModel(dset=val_dataset, args=args)
        model.load_state_dict(torch.load(os.path.join(
            args.save_root, model_path, snap), map_location='cpu')['state_dict'])
        model.gen_joint_aa()

        if args.cuda:
            model.cuda()

        res = []
        label = []
        for _, pack in enumerate(val_loader):
            res_i, label_i = model.infer_forward(pack)
            res.append(res_i)
            label.append(label_i)

        res = np.concatenate(res, axis=0)
        label = np.concatenate(label, axis=0)
        
        eval_res = get_eval(res, label, mode=mode)
        print('snap:{} {}:{}'.format(snap, mode, eval_res))
        eval_res_List.append(eval_res)
        if best_eval_res is None or eval_res > best_eval_res:
            best_eval_res = eval_res
            bestmodelNum = int(snap.split('.')[0].split('_')[1])

    print('best {}:{}'.format(mode, best_eval_res))
    print('best model:{}'.format('snap_'+str(bestmodelNum)))
    plt.figure()
    plt.plot(range(len(eval_res_List)), eval_res_List, label='composition {}'.format(mode))
    plt.legend()
    if not os.path.exists('./save/composition_train/imgs'):
        os.makedirs('./save/composition_train/imgs')
    plt.savefig('./save/composition_train/imgs/snaps_eval_{}.png'.format(mode))
    plt.close()


if __name__ == '__main__':
    args = p_parse()
    # eval_joint_classification(args)
    # eval_split_classification(args)
    # eval_actor_or_action_classification(args)

    eval_composition_1(args)
