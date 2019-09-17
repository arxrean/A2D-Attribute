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
        self.margin = args.triplet_margin

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

        self.valid = {11: 0, 12: 1, 13: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 7, 21: 8,
                      22: 9, 26: 10, 28: 11, 29: 12, 34: 13, 35: 14, 36: 15, 39: 16,
                      41: 17, 43: 18, 44: 19, 45: 20, 46: 21, 48: 22, 49: 23, 54: 24,
                      55: 25, 56: 26, 57: 27, 59: 28, 61: 29, 63: 30, 65: 31, 66: 32,
                      67: 33, 68: 34, 69: 35, 72: 36, 73: 37, 75: 38, 76: 39, 77: 40,
                      78: 41, 79: 42}
        self.valid_actor = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
        self.valid_action = {1: 0, 2: 1, 3: 2,
                             4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}

    def onehot2idx(self, pairs):
        actors_idx = []
        actions_idx = []

        for pair in pairs.numpy():
            pairindexs = np.where(pair == 1)[0]
            actor_idx = []
            action_idx = []
            for pairindex in pairindexs:
                pairlabel = list(self.valid.keys())[list(
                    self.valid.values()).index(pairindex)]
                actor_idx.append(self.valid_actor[pairlabel // 10])
                action_idx.append(self.valid_action[pairlabel % 10])
            actors_idx.append(torch.Tensor(actor_idx).type(torch.LongTensor).cuda(
            ) if self.args.cuda else torch.Tensor(actor_idx).type(torch.LongTensor))
            actions_idx.append(torch.Tensor(action_idx).type(torch.LongTensor).cuda(
            ) if self.args.cuda else torch.Tensor(action_idx).type(torch.LongTensor))
        return actors_idx, actions_idx

    def train_forward(self, x):
        img, pairs, neg_action, neg_actor, img_path = x[0], x[1], x[2], x[3], x[4]

        actors, actions = self.onehot2idx(pairs)

        neg_actor = list(map(lambda i: torch.Tensor(
            [i]).type(torch.LongTensor).cuda() if self.args.cuda else torch.Tensor([i]).type(torch.LongTensor), np.where(neg_actor.numpy() == 1)[1]))

        neg_action = list(map(lambda i: torch.Tensor(
            [i]).type(torch.LongTensor).cuda() if self.args.cuda else torch.Tensor([i]).type(torch.LongTensor), np.where(neg_action.numpy() == 1)[1]))

        loss = []
        # pdb.set_trace()

        feature = self.image_embedder(img.cuda() if self.args.cuda else img)

        # positive
        actor_emb = list(map(lambda x: self.obj_embedder(x), actors))
        pos_actions = list(map(lambda action: torch.stack(
            list(map(lambda idx: self.action_ops[idx], action))), actions))

        pos_pairs = list(zip(actor_emb, pos_actions))

        positive = torch.stack(list(map(lambda pair: torch.mean(self.apply_ops(
            pair[1], pair[0]), dim=0, keepdim=True), pos_pairs)), dim=0).squeeze()

        # negative
        neg_actor_emb = torch.stack(
            list(map(lambda actor: self.obj_embedder(actor), neg_actor))).squeeze()
        neg_actions = torch.stack(
            list(map(lambda action: self.action_ops[action.item()], neg_action)))
        negative = self.apply_ops(neg_actions, neg_actor_emb)

        loss_triplet = F.triplet_margin_loss(
            feature, positive, negative, margin=self.margin)
        loss.append(loss_triplet)

        loss = sum(loss)
        return loss, None

    def infer_forward(self, x):
        img, pairs, img_path = x[0], x[1], x[2]
        img_emd = self.image_embedder(
            img.cuda() if self.args.cuda else img).detach().cpu().numpy()
        distance_distribution = np.expand_dims(1/np.array(list(
            map(lambda j: np.linalg.norm(img_emd-j), self.composition_res))), 0)

        return distance_distribution, pairs

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

    def gen_joint_aa(self):
        eval_joint_dict = {}
        val_actors = self.obj_embedder(
            torch.LongTensor([0, 1, 2, 3, 4, 5, 6]))
        for i in range(len(val_actors)):
            for j in range(len(self.valid_action)):
                joint_aa = int('{}{}'.format(i+1, j+1))
                if joint_aa in self.valid:
                    eval_joint_dict[self.valid[joint_aa]] = torch.bmm(torch.stack(
                        [self.action_ops[j]]), val_actors[i:i+1].unsqueeze(2)).squeeze()

        list_res = []
        for i in range(43):
            list_res.append(eval_joint_dict[i].detach().cpu().numpy())

        self.composition_res = np.stack(list_res)


def getJointClassifier(args):
    net = JointClassifier(backbone=res_block_50(args), args=args)

    return net


def getSplitClassifier(args):
    net = SplitClassifier(backbone=res_block_50(args), args=args)

    return net
