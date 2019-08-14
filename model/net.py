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

from model.backbone import res_block_50

class JointClassifier(nn.Module):

    def __init__(self, backbone, args):
        super(JointClassifier, self).__init__()
        # feature encoding
        self.backbone = backbone
        self.args = args

        # classifier
        self.adapool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc2 = nn.Linear(2048, self.args.class_num)

    def forward(self, x):
        x = self.backbone(x)
        fc = self.adapool(x).squeeze()
        out = self.fc2(fc)

        return out, fc


class SplitClassifier(nn.Module):
    def __init__(self, backbone, args):
        super(SplitClassifier, self).__init__()
        # feature encoding
        self.backbone = backbone
        self.args = args

        # classifier
        self.adapool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.actor_fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Linear(2048, self.args.actor_num)
        )
        self.action_fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.Linear(2048, self.args.action_num)
        )

    def forward(self, x):
        x = self.backbone(x)
        fc = self.adapool(x).squeeze()
        actor_out = self.actor_fc(fc)
        action_out = self.action_fc(fc)

        return actor_out, action_out


def getJointClassifier(args):
    net = JointClassifier(backbone=res_block_50(args), args=args)

    return net

def getSplitClassifier(args):
    net = SplitClassifier(backbone=res_block_50(args), args=args)

    return net
