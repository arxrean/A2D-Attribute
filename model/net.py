import argparse
import os
import shutil
import time
import math
import pdb
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

    def draw_cam(self, x):
        last_conv = self.backbone(x)
        last_conv.requires_grad_()
        last_conv.retain_grad()

        fc = self.adapool(last_conv).squeeze()
        out = self.fc2(fc)

        return out, last_conv

    def gen_feature(self, x):
        last_conv = self.backbone(x)
        last_conv.requires_grad_()
        last_conv.retain_grad()

        fc = self.adapool(last_conv).squeeze()

        return fc


class SplitClassifier(nn.Module):
    def __init__(self, backbone, args):
        super(SplitClassifier, self).__init__()
        # feature encoding
        self.backbone = backbone
        self.args = args

        # classifier
        self.adapool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.actor_fc = nn.Sequential(
            nn.Linear(2048, self.args.actor_num)
        )
        self.action_fc = nn.Sequential(
            nn.Linear(2048, self.args.action_num)
        )

    def forward(self, x):
        x = self.backbone(x)
        fc = self.adapool(x).squeeze()
        actor_out = self.actor_fc(fc)
        action_out = self.action_fc(fc)

        return actor_out, action_out

def load_word_embeddings(emb_file, vocab):

    vocab = [v.lower() for v in vocab]

    embeds = torch.from_numpy(np.load(emb_file))

    return embeds

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True):
        super(MLP, self).__init__()
        mod = []
        for L in range(num_layers-1):
            mod.append(nn.Linear(inp_dim, inp_dim, bias=bias))
            mod.append(nn.ReLU(True))

        mod.append(nn.Linear(inp_dim, out_dim, bias=bias))
        if relu:
            mod.append(nn.ReLU(True))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output


class ManifoldModel(nn.Module):

    def __init__(self, dset, args):
        super(ManifoldModel, self).__init__()
        self.args = args
        self.dset = dset
        self.margin = 0.5

        # precompute validation pairs
        actors, actions = list(map(lambda x: x.split(' ')[0], self.dset.pairs)), list(
            map(lambda x: x.split(' ')[1], self.dset.pairs))
        actions = [dset.action2idx[action] for action in actions]
        actors = [dset.actor2idx[actor] for actor in actors]
        self.val_attrs = torch.LongTensor(actions).cuda() if args.cuda else torch.LongTensor(actions)
        self.val_objs = torch.LongTensor(actors).cuda() if args.cuda else torch.LongTensor(actors)

        # OP
        self.image_embedder = MLP(dset.feat_dim, args.op_img_dim)
        self.compare_metric = lambda img_feats, pair_embed: - \
            F.pairwise_distance(img_feats, pair_embed)

        self.action_ops = nn.ParameterList([nn.Parameter(torch.eye(args.op_img_dim)) for _ in range(len(self.dset.actions))])
        self.obj_embedder = nn.Embedding(len(dset.actors), args.op_img_dim)

        glove_pretrained_weight = load_word_embeddings('./repo/actors_glove.npy', dset.actors)
        self.obj_embedder.weight.data.copy_(glove_pretrained_weight)


def getJointClassifier(args):
    net = JointClassifier(backbone=res_block_50(args), args=args)

    return net


def getSplitClassifier(args):
    net = SplitClassifier(backbone=res_block_50(args), args=args)

    return net
