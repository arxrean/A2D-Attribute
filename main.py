import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import shutil
import matplotlib.pyplot as plt
import time

from loader.A2DClsLoader import A2DClassification, A2DClassificationWithActorAction
from loader.A2DCompositionLoader import A2DComposition
from model.net import getJointClassifier, getSplitClassifier, ManifoldModel
from utils.opt import get_finetune_optimizer, get_op_optimizer
from utils.helper import get_pos_weight, bce_weight_loss
from glob import p_parse
import pdb


def joint_classification(args):
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop((args.input_size, args.input_size)),
        transforms.Resize((args.input_size, args.input_size)),
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)

    val_dataset = A2DClassification(args, val_transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)

    model = getJointClassifier(args)
    if args.cuda:
        model = model.cuda()

    criterion = bce_weight_loss(args=args)

    opt = get_finetune_optimizer(args, model)

    if os.path.exists('./save/joint_classification/snap/'):
        shutil.rmtree('./save/joint_classification/snap/')
    os.makedirs('./save/joint_classification/snap/')

    train_loss = []
    val_loss = []
    for epoch in range(args.max_epoches):
        train_loss.append(0)
        val_loss.append(0)
        for _, pack in enumerate(train_loader):
            imgs = pack[1]  # (N,t,c,m,n)
            labels = pack[2]  # (N,t,c,m,n)

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            out, _ = model(imgs)
            loss = criterion.get_loss(out, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss[-1] += loss.item()

        with torch.no_grad():
            for _, pack in enumerate(val_loader):
                imgs = pack[1]  # (N,t,c,m,n)
                labels = pack[2]  # (N,t,c,m,n)

                if args.cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()

                out, _ = model(imgs)
                loss = criterion.get_loss(out, labels)

                val_loss[-1] += loss.item()

        print('epoch:{} train_loss:{:.3f} val_loss:{:.4f}'.format(
            epoch, train_loss[-1], val_loss[-1]), flush=True)

        # plot
        plt.figure()
        plt.plot(range(len(train_loss)), train_loss, label='train_loss')
        plt.plot(range(len(val_loss)), val_loss, label='val_loss')
        plt.legend()
        if not os.path.exists('./save/joint_classification/imgs'):
            os.makedirs('./save/joint_classification/imgs')
        plt.savefig('./save/joint_classification/imgs/train_line.png')
        plt.close()

        snap_shot = {'epoch': epoch, 'train_loss': train_loss,
                     'val_loss': val_loss, 'state_dict': model.state_dict()}
        torch.save(
            snap_shot, './save/joint_classification/snap/snap_{}.pth.tar'.format(epoch))


def split_classification(args):
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

    train_dataset = A2DClassificationWithActorAction(args, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)

    val_dataset = A2DClassificationWithActorAction(
        args, val_transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)

    model = getSplitClassifier(args)
    if args.cuda:
        model = model.cuda()

    criterion = bce_weight_loss(args=args)

    if os.path.exists('./save/split_classification/snap/'):
        shutil.rmtree('./save/split_classification/snap/')
    os.makedirs('./save/split_classification/snap/')

    train_loss = []
    val_top1_acc = []
    for epoch in range(args.max_epoches):
        opt = get_finetune_optimizer(args, model, epoch)

        plt.figure()
        train_loss.append(0)
        val_top1_acc.append([])
        for _, pack in enumerate(train_loader):
            imgs = pack[0]  # (N,t,c,m,n)
            actor_labels = pack[2]
            action_labels = pack[3]

            if args.cuda:
                imgs = imgs.cuda()
                actor_labels = actor_labels.cuda()
                action_labels = action_labels.cuda()

            actor_out, action_out = model(imgs)
            actor_loss = criterion.get_loss(actor_out, actor_labels)
            action_loss = criterion.get_loss(
                action_out, action_labels)
            opt.zero_grad()
            (actor_loss+action_loss).backward()
            opt.step()

            train_loss[-1] += actor_loss.item()+action_loss.item()

        with torch.no_grad():
            for _, pack in enumerate(val_loader):
                imgs = pack[0]  # (N,t,c,m,n)
                actor_labels = pack[2]
                action_labels = pack[3]

                if args.cuda:
                    imgs = imgs.cuda()
                    actor_labels = actor_labels.cuda()
                    action_labels = action_labels.cuda()

                actor_out, action_out = model(imgs)

                res = torch.sigmoid(actor_out).detach().cpu().numpy() > 0.5
                gt = actor_labels.cpu().numpy() > 0.5
                val_top1_acc[-1].extend((res == gt).reshape(-1))

        val_top1_acc[-1] = np.mean(val_top1_acc[-1])
        print('epoch:{} train_loss:{:.3f} val_acc:{:.3f}'.format(
            epoch, train_loss[-1], val_top1_acc[-1]), flush=True)

        # plot
        plt.plot(range(len(train_loss)), train_loss, label='train_loss')
        plt.plot(range(len(val_top1_acc)), val_top1_acc, label='val_acc')
        plt.legend()
        if not os.path.exists('./save/split_classification/imgs'):
            os.makedirs('./save/split_classification/imgs')
        plt.savefig('./save/split_classification/imgs/train_line.png')
        plt.close()

        # save
        snap_shot = {'epoch': epoch, 'train_loss': train_loss,
                     'val_acc': val_top1_acc, 'state_dict': model.state_dict()}
        torch.save(
            snap_shot, './save/split_classification/snap/snap_{}.pth.tar'.format(epoch))


def composition_train(args):
    train_dataset = A2DComposition(
        args, None, mode='train')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)

    model = ManifoldModel(dset=train_dataset, args=args)
    if args.cuda:
        model.cuda()

    opt = get_op_optimizer(args, model)

    cls_criterion = bce_weight_loss(args=args)

    if os.path.exists('./save/composition_train/snap/'):
        shutil.rmtree('./save/composition_train/snap/')
    os.makedirs('./save/composition_train/snap/')

    train_loss = []
    for epoch in range(args.max_epoches):
        plt.figure()
        train_loss.append(0)
        for _, pack in enumerate(train_loader):
            trip_loss, actor_pred, action_pred = model.train_forward(pack)

            total_loss = trip_loss

            if actor_pred is not None:
	            # one-hot 0/1 vector
	            pos_actors = pack[2].cuda() if args.cuda else pack[2]
	            pos_actions = pack[3].cuda() if args.cuda else pack[3]
	            actor_pred_loss = cls_criterion.get_loss(actor_pred, pos_actors)
	            action_pred_loss = cls_criterion.get_loss(action_pred, pos_actions)

	            total_loss = trip_loss + \
	                (actor_pred_loss+action_pred_loss)*args.constraint_cls

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            train_loss[-1] += total_loss.item()

        print('epoch:{} train_loss:{:.3f}'.format(
            epoch, train_loss[-1], flush=True))

        # plot
        plt.plot(range(len(train_loss)), train_loss, label='train_loss')
        plt.legend()
        if not os.path.exists('./save/composition_train/imgs'):
            os.makedirs('./save/composition_train/imgs')
        plt.savefig('./save/composition_train/imgs/train_line.png')
        plt.close()

        snap_shot = {'epoch': epoch, 'train_loss': train_loss,
                     'state_dict': model.state_dict()}
        torch.save(
            snap_shot, './save/composition_train/snap/snap_{}.pth.tar'.format(epoch))


if __name__ == '__main__':
    args = p_parse()
    joint_classification(args)
    # split_classification(args)
    # composition_train(args)
