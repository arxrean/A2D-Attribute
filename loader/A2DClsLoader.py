# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:21:00 2019

@author: Slash
"""

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from keras.utils import to_categorical
import pdb


class A2DClassification(Dataset):
    def __init__(self, args, transform=None, mode='train'):
        # img_id(dir_file),path,label,is_train
        self.args = args
        self.csv = pd.read_csv(args.csv_path)
        self.transform = transform

        self.mode = mode
        if self.mode == 'train':
            self.csv = self.csv[self.csv['is_train'] == 1]
        elif self.mode == 'val':
            self.csv = self.csv[self.csv['is_train'] == 0]
        else:
            self.csv = self.csv[self.csv['is_train'] == 2]
        self.csv.reset_index(drop=True, inplace=True)

        self.valid = {11: 0, 12: 1, 13: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 7, 21: 8,
                      22: 9, 26: 10, 28: 11, 29: 12, 34: 13, 35: 14, 36: 15, 39: 16,
                      41: 17, 43: 18, 44: 19, 45: 20, 46: 21, 48: 22, 49: 23, 54: 24,
                      55: 25, 56: 26, 57: 27, 59: 28, 61: 29, 63: 30, 65: 31, 66: 32,
                      67: 33, 68: 34, 69: 35, 72: 36, 73: 37, 75: 38, 76: 39, 77: 40,
                      78: 41, 79: 42}
        self.actor = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
        self.action = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}

        self.num_valid = len(self.valid)
        self.num_actor = len(self.actor)
        self.num_action = len(self.action)

    def __getitem__(self, index):
        item = self.csv.loc[index]

        video_name = item['img_id'].split('/')[1]

        if self.args.t == 0:
            img = Image.open(os.path.join(self.args.a2d_root, item['img_id']))
            img = self.transform(img)
        else:
            img = []
            img_frame = int(item['img_id'].split('/')[-1].split('.')[0])
            for i in range(-self.args.t, self.args.t+1):
                t_img_name = str(img_frame + i).zfill(5)
                t_img = Image.open(os.path.join(
                    self.args.a2d_root, 'pngs320H', video_name, t_img_name + '.png'))
                t_img = self.transform(t_img)
                img.append(t_img)

        listlabel = item['label'].split(' ')
        label = []
        for i in listlabel:
            i = int(float(i))
            if i in self.valid:
                label.append(self.valid[i])

        label = to_categorical(label, num_classes=self.num_valid)

        pdb.set_trace()
        if len(label) == 1:
            label = label[0]

        return img, label, item['img_id']

    # def __getitemActor__(self, index):
    #     item = self.csv.loc[index]

    #     video_name = item['img_id'].split('/')[1]

    #     if self.args.t == 0:
    #         img = Image.open(os.path.join(self.args.a2d_root, item['img_id']))
    #         img = self.transform(img)
    #     else:
    #         img = []
    #         img_frame = int(item['img_id'].split('/')[-1].split('.')[0])
    #         for i in range(-self.args.t, self.args.t+1):
    #             t_img_name = str(img_frame + i).zfill(5)
    #             t_img = Image.open(os.path.join(
    #                 self.args.a2d_root, 'pngs320H', video_name, t_img_name + '.png'))
    #             t_img = self.transform(t_img)
    #             img.append(t_img)

    #     listlabel = item['label'].split(' ')
    #     label = []
    #     for i in listlabel:
    #         i = int(float(i)) // 10
    #         if i in self.actor:
    #             label.append(self.actor[i])

    #     label = to_categorical(label, num_classes=self.num_actor)

    #     if len(label) == 1:
    #         label = label[0]

    #     return img, label, video_name

    # def __getitemAction__(self, index):
    #     item = self.csv.loc[index]

    #     video_name = item['img_id'].split('/')[1]

    #     if self.args.t == 0:
    #         img = Image.open(os.path.join(self.args.a2d_root, item['img_id']))
    #         img = self.transform(img)
    #     else:
    #         img = []
    #         img_frame = int(item['img_id'].split('/')[-1].split('.')[0])
    #         for i in range(-self.args.t, self.args.t+1):
    #             t_img_name = str(img_frame + i).zfill(5)
    #             t_img = Image.open(os.path.join(
    #                 self.args.a2d_root, 'pngs320H', video_name, t_img_name + '.png'))
    #             t_img = self.transform(t_img)
    #             img.append(t_img)

    #     listlabel = item['label'].split(' ')
    #     label = []
    #     for i in listlabel:
    #         i = int(float(i)) % 10
    #         if i in self.action:
    #             label.append(self.action[i])

    #     label = to_categorical(label, num_classes=self.num_action)

    #     if len(label) == 1:
    #         label = label[0]

    #     return img, label, video_name

    def __len__(self):
        return len(self.csv)


class A2DClassificationWithActorAction(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return img, label, actor_label, action_label, video_name

    def __len__(self):
        pass
