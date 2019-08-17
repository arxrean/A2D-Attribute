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

from loader.A2DClsLoader import A2DClassification, A2DClassificationWithActorAction
from model.net import getJointClassifier, getSplitClassifier
from utils.opt import get_finetune_optimizer
from utils.helper import get_pos_weight, bce_weight_loss


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
    parser.add_argument("--batch_size", default=48, type=int)
    parser.add_argument("--max_epoches", default=1000, type=int)
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)

    val_dataset = A2DClassification(args, val_transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
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

    if os.path.exists('./save/joint_classification/snap/'):
        shutil.rmtree('./save/joint_classification/snap/')
    os.makedirs('./save/joint_classification/snap/')

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

                if args.cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()

                out, fc = model(imgs)
                loss = criterion(out, labels)

                val_loss[-1] += loss.item()

        print('epoch:{} train_loss:{:.3f} val_loss:{:.4f}'.format(
            epoch, train_loss[-1], val_loss[-1]), flush=True)
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
    val_loss = []
    for epoch in range(args.max_epoches):
        opt = get_finetune_optimizer(args, model, epoch)

        train_loss.append(0)
        val_loss.append(0)
        for iter, pack in enumerate(train_loader):
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
            for iter, pack in enumerate(val_loader):
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

                val_loss[-1] += actor_loss.item()+action_loss.item()

        print('epoch:{} train_loss:{:.3f} val_loss:{:.3f}'.format(
            epoch, train_loss[-1], val_loss[-1]), flush=True)
        snap_shot = {'epoch': epoch, 'train_loss': train_loss,
                     'val_loss': val_loss, 'state_dict': model.state_dict()}
        torch.save(
            snap_shot, './save/split_classification/snap/snap_{}.pth.tar'.format(epoch))


def split_classification_total_frac(args):
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

    actor_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(np.load('./repo/actor_label_frac.npy')).type(torch.FloatTensor).cuda() if args.cuda else torch.from_numpy(np.load('./repo/actor_label_frac.npy')).type(torch.FloatTensor))
    action_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(np.load('./repo/action_label_frac.npy')).type(torch.FloatTensor).cuda() if args.cuda else torch.from_numpy(np.load('./repo/action_label_frac.npy')).type(torch.FloatTensor))

    model_save_path = './save/{}/snap/'.format(
        split_classification_total_frac.__name__)
    if os.path.exists(model_save_path):
        shutil.rmtree(model_save_path)
    os.makedirs(model_save_path)

    train_loss = []
    val_loss = []
    for epoch in range(args.max_epoches):
        opt = get_finetune_optimizer(args, model, epoch)

        train_loss.append(0)
        val_loss.append(0)
        for iter, pack in enumerate(train_loader):
            imgs = pack[0]  # (N,t,c,m,n)
            actor_labels = pack[2]
            action_labels = pack[3]

            if args.cuda:
                imgs = imgs.cuda()
                actor_labels = actor_labels.cuda()
                action_labels = action_labels.cuda()

            actor_out, action_out = model(imgs)
            actor_loss = actor_criterion(actor_out, actor_labels)
            action_loss = action_criterion(action_out, action_labels)
            opt.zero_grad()
            (actor_loss+action_loss).backward()
            opt.step()

            train_loss[-1] += actor_loss.item()+action_loss.item()

        with torch.no_grad():
            for iter, pack in enumerate(val_loader):
                imgs = pack[0]  # (N,t,c,m,n)
                actor_labels = pack[2]
                action_labels = pack[3]

                if args.cuda:
                    imgs = imgs.cuda()
                    actor_labels = actor_labels.cuda()
                    action_labels = action_labels.cuda()

                actor_out, action_out = model(imgs)
                actor_loss = actor_criterion(actor_out, actor_labels)
                action_loss = action_criterion(action_out, action_labels)

                val_loss[-1] += actor_loss.item()+action_loss.item()

        print('epoch:{} train_loss:{:.3f} val_loss:{:.3f}'.format(
            epoch, train_loss[-1], val_loss[-1]), flush=True)
        snap_shot = {'epoch': epoch, 'train_loss': train_loss,
                     'val_loss': val_loss, 'state_dict': model.state_dict()}
        torch.save(
            snap_shot, os.path.join(model_save_path, 'snap_{}.pth.tar'.format(epoch)))


if __name__ == '__main__':
    args = p_parse()
    # joint_classification(args)
    # split_classification(args)
    split_classification_total_frac(args)
