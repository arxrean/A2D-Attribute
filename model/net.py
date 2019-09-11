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
        self.val_attrs = torch.LongTensor(actions).cuda(
        ) if args.cuda else torch.LongTensor(actions)
        self.val_objs = torch.LongTensor(actors).cuda(
        ) if args.cuda else torch.LongTensor(actors)

        # OP
        self.image_embedder = MLP(dset.feat_dim, args.op_img_dim)
        self.compare_metric = lambda img_feats, pair_embed: - \
            F.pairwise_distance(img_feats, pair_embed)

        self.action_ops = nn.ParameterList(
            [nn.Parameter(torch.eye(args.op_img_dim)) for _ in range(len(self.dset.actions))])
        self.obj_embedder = nn.Embedding(len(dset.actors), args.op_img_dim)

        glove_pretrained_weight = load_word_embeddings(
            './repo/actors_glove.npy', dset.actors)
        self.obj_embedder.weight.data.copy_(glove_pretrained_weight)

    def train_forward(self, x):
        img, actions, actors, neg_action, neg_actor = x[0], x[1], x[2], x[-2], x[-1]

        actors = [torch.where(actor == 1)[0].cuda() if self.args.cuda else torch.where(
            actor == 1)[0] for actor in actors]
        actions = [torch.where(action == 1)[0] for action in actions]
        neg_actor = [torch.where(actor == 1)[0].cuda() if self.args.cuda else torch.where(
            actor == 1)[0] for actor in neg_actor]
        neg_action = [torch.where(action == 1)[0] for action in neg_action]

        loss=[]
        # pdb.set_trace()

        feature = self.image_embedder(img.cuda() if self.args.cuda else img)

        # positive
        actor_emb = torch.cat([torch.mean(self.obj_embedder(
            actor), dim=0, keepdim=True) for actor in actors], dim=0)
        pos_actions = torch.stack([torch.mean(torch.stack(
            [self.action_ops[idx] for idx in action]), dim=0) for action in actions])
        positive = self.apply_ops(pos_actions, actor_emb)

        # negative
        neg_actor_emb = torch.cat([torch.mean(self.obj_embedder(
            actor), dim=0, keepdim=True) for actor in neg_actor], dim=0)
        neg_actions = torch.stack([self.action_ops[action.item()]
                               for action in neg_action])
        negative = self.apply_ops(neg_actions, neg_actor_emb)

        loss_triplet = F.triplet_margin_loss(
            feature, positive, negative, margin=self.margin)
        loss.append(loss_triplet)

        loss = sum(loss)
        return loss, None

    def compose(self, actions, actors):
        obj_rep = self.obj_embedder(actors)
        attr_ops = torch.stack([self.action_ops[attr.item()]
                                for attr in attrs])
        embedded_reps = self.apply_ops(attr_ops, obj_rep)
        return embedded_reps

    def apply_ops(self, actions, actors):
        out = torch.bmm(actions, actors.unsqueeze(2)).squeeze(2)
        out = F.relu(out)
        return out


def getJointClassifier(args):
    net = JointClassifier(backbone=res_block_50(args), args=args)

    return net


def getSplitClassifier(args):
    net = SplitClassifier(backbone=res_block_50(args), args=args)

    return net
