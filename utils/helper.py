import numpy as np
from heapq import nlargest
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import pdb
import cv2
# data format
# The dimension of X_pre and X_gt are NXnum_cls, whether N is the number of samples and num_cls is the number of classes


def get_pos_weight(gt_labels, args):
    pos_sum = torch.sum(gt_labels, dim=0)

    return (len(gt_labels)-pos_sum)/(pos_sum+1e-5)


def show_cam_on_image(args, img_path, mask, save_path):
    img = cv2.imread(img_path, 1)
    img = np.float32(cv2.resize(img, (args.input_size, args.input_size))) / 255

    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_path, np.uint8(255 * cam))

def np_sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


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
