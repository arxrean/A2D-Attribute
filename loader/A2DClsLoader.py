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

    def __getitem__(self, index):
        t = self.args.t
        rootPath = self.args.a2d_root
        item = self.csv.loc[index]

        video_name = item['img_id'].split('/')[1]

        if t == 0:
            img = Image.open(os.path.join(rootPath, item['img_id']))
            img = self.transform(img)
        else:
            img = []
            img_frame = int(item['img_id'].split('/')[-1].split('.')[0])
            for i in range(-t, t+1):
                t_img_name = str(img_frame + i).zfill(5)
                t_img = Image.open(os.path.join(
                    rootPath, 'pngs320H', video_name, t_img_name + '.png'))
                t_img = self.transform(t_img)
                img.append(t_img)

        label = item['label']

        return item['img_id'], img, label

    def __len__(self):
        return len(self.csv)
