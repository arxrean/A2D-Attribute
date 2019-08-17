import numpy as np
from heapq import nlargest
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
# data format
# The dimension of X_pre and X_gt are NXnum_cls, whether N is the number of samples and num_cls is the number of classes


def Precision(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x*y)/(np.sum(x) + 1e-8)
    return p/N


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


def get_eval(X_pre, X_gt):
    best_f1 = None
    best_prec = None
    best_recall = None
    caltype = None

    best_f1_threshold = None
    best_prec_threshold = None
    best_recall_threshold = None
    Threshold = None

    best_f1_maxnum = None
    best_prec_maxnum = None
    best_recall_maxnum = None
    maxnum = None

    for thd in np.arange(0, 1, 0.01):
        X_pre_new = np.array(X_pre > thd, dtype='float64')
        f1 = F1(X_pre_new, X_gt)
        if best_f1_threshold is None or f1 > best_f1_threshold:
            best_f1_threshold = f1
            best_prec_threshold = Precision(X_pre_new, X_gt)
            best_recall_threshold = Recall(X_pre_new, X_gt)
            Threshold = thd

    for num in range(5):
        for row in range(len(X_pre)):
            largest = nlargest(num+1, X_pre[row, :])
            X_pre_new = X_pre
            X_pre_new[row, :] = np.where(X_pre[row, :] >= min(largest), 1, 0)
        f1 = F1(X_pre_new, X_gt)
        if best_f1_maxnum is None or f1 > best_f1_maxnum:
            best_f1_maxnum = f1
            best_prec_maxnum = Precision(X_pre_new, X_gt)
            best_recall_maxnum = Recall(X_pre_new, X_gt)
            maxnum = num+1

    if(best_f1_maxnum > best_f1_threshold):
        best_f1 = best_f1_maxnum
        best_prec = best_prec_maxnum
        best_recall = best_recall_maxnum
        caltype = 'maxnum'
    else:
        best_f1 = best_f1_threshold
        best_prec = best_prec_threshold
        best_recall = best_recall_threshold
        caltype = 'threshold'

    print('best_f1:{}'.format(best_f1))
    print('best_prec:{}'.format(best_prec))
    print('best_recall:{}'.format(best_recall))
    print('caltype:{}'.format(caltype))
    if caltype == 'threshold':
        print('Threshold:{}'.format(Threshold))
        return best_f1, best_prec, best_recall, caltype, Threshold
    else:
        print('maxnum:{}'.format(maxnum))
        return best_f1, best_prec, best_recall, caltype, maxnum


def get_pos_weight(gt_labels, args):
    pos_sum = torch.sum(gt_labels, dim=0)

    return (len(gt_labels)-pos_sum)/(pos_sum+1e-5)


class bce_weight_loss:
    def __init__(self, args, reduce=True, mean=True, nouse=False):
        self.reduce = reduce
        self.mean = mean
        self.args = args
        self.nouse = nouse
        self.criterion = None

    def get_loss(self, output, target):
        if self.nouse:
            return nn.BCEWithLogitsLoss()(output, target)
        pos_weight = get_pos_weight(target, self.args)
        if self.args.cuda:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight.type(torch.FloatTensor).cuda())
        else:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight.type(torch.FloatTensor))

        return self.criterion(output, target)


def plot_loss_acc(slurm_or_snap):
    if os.path.isfile(slurm_or_snap):
        lines = open(slurm_or_snap, 'r').readlines()[1:]
        train_loss = list(map(lambda x: float(x.split(
            ' ')[1].split(':')[-1].strip()), lines))
        val_loss = list(map(lambda x: float(x.split(
            ' ')[2].split(':')[-1].strip()), lines))

        plt.figure()
        plt.plot(range(len(train_loss)), train_loss, label='train_loss')
        plt.plot(range(len(val_loss)), val_loss, label='val_loss')

        plt.legend()
        plt.savefig(
            '/mnt/lustre/jiangsu/dlar/home/zyk17/newcode/A2D-Attribute/save/split_classification/imgs/train_line.png')
    else:
        pass


if __name__ == '__main__':
    plot_loss_acc(
        '/mnt/lustre/jiangsu/dlar/home/zyk17/newcode/A2D-Attribute/slurm-8391093.out')
