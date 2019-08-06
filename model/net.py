import argparse
import os
import shutil
import time
import math
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parameter import Parameter
import numpy as np
import cv2

from backbone import res_block_50


class JointClassifier(nn.Module):

    def __init__(self, backbone):
        super(FC_ResNet, self).__init__()
        # feature encoding
        self.backbone = backbone

        # classifier
        self.adapool = model.avgpool
        self.fc2 = nn.Linear(2048, self.args.class_num)

    def forward(self, x):
        x = self.backbone(x)
        fc = self.adapool(x).squeeze()
        out = self.fc2(fc)
        return out, fc


def getJointClassifier(args):
    net = JointClassifier(backbone=res_block_50(args), args=args)
