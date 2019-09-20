import torch
import torch.utils.data as tdata
import numpy as np
import pandas as pd
import os
import pdb
from PIL import Image

from keras.utils import to_categorical


class A2DComposition(tdata.Dataset):
    def __init__(self, args, transform=None, backbone_embedder=None, mode='train'):
        self.args = args
        self.transform = transform
        self.mode = mode
        self.csv = pd.read_csv(args.csv_path)
        self.backbone_embedder = backbone_embedder

        self.maxlenlabel, self.train_label_len_probability = self.train_sample_probability()

        # pdb.set_trace()

        self.actors_dict = {1: 'adult', 2: 'baby',
                            3: 'ball', 4: 'bird', 5: 'car', 6: 'cat', 7: 'dog'}
        self.actions_dict = {1: 'climbing', 2: 'crawling', 3: 'eating', 4: 'flying',
                             5: 'jumping', 6: 'rolling', 7: 'running', 8: 'walking', 9: 'none'}

        self.actions, self.actors, self.pairs, self.train_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.test_data = self.get_split_info()
        self.data = self.train_data if self.mode == 'train' else self.test_data

        self.action2idx = {action: idx for idx,
                           action in enumerate(self.actions)}
        self.actor2idx = {actor: idx for idx, actor in enumerate(self.actors)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        self.actor_affordance = {}
        self.train_actor_affordance = {}

        for _actor in self.actors:
            candidates = []
            for _, _, actor, action in self.train_data+self.test_data:
                for a in actor:
                    if a == _actor:
                        candidates.append(action[actor.index(a)])
            self.actor_affordance[_actor] = list(set(candidates))

            candidates = []
            for _, _, actor, action in self.train_data+self.test_data:
                for a in actor:
                    if a == _actor:
                        candidates.append(action[actor.index(a)])
            self.train_actor_affordance[_actor] = list(set(candidates))

        if self.args.use_feat:
            # pre-joint features path
            feat_file = args.feature_path
            activation_data = np.load(feat_file, allow_pickle=True).tolist()
            # activation_data = torch.load(feat_file) # [features,names] features=>(N,d)
            self.activations = dict(
                zip(activation_data['id'], activation_data['res']))
            self.feat_dim = len(activation_data['res'][0])  # d
        else:
            self.activations = None

    def parse_split(self):
        tr_actions = []
        tr_actors = []
        tr_pairs = []

        ts_actions = []
        ts_actors = []
        ts_pairs = []

        tr_labels = self.csv[self.csv['usage'] == 0]['label'].values
        ts_labels = self.csv[self.csv['usage'] == 1]['label'].values

        tr_actors, tr_actions, tr_pairs = self.getlabel(tr_labels)
        ts_actors, ts_actions, ts_pairs = self.getlabel(ts_labels)

        all_actions, all_actors = sorted(
            list(set(tr_actions+ts_actions))), sorted(list(set(tr_actors+ts_actors)))
        all_pairs = sorted(list(set(tr_pairs + ts_pairs)))

        all_actions = self.indexaction2string(all_actions)
        all_actors = self.indexactor2string(all_actors)
        all_pairs = self.indexpair2string(all_pairs)

        tr_pairs = self.indexpair2string(tr_pairs)
        ts_pairs = self.indexpair2string(ts_pairs)

        return all_actions, all_actors, all_pairs, tr_pairs, ts_pairs

    def getlabel(self, labels):
        indexPair = []
        indexActors = []
        indexActions = []

        for item in labels:
            label_list = item.split(' ')
            indexPair += list(map(lambda label: int(float(label)), label_list))
            indexActors += list(map(lambda label: int(float(label)
                                                      ) // 10, label_list))
            indexActions += list(map(lambda label: int(float(label)) %
                                     10, label_list))
            '''
            for label in label_list:
                label = int(float(label))
                indexPair.append(label)
                indexActors.append(label // 10)
                indexActions.append(label % 10)
            '''
        indexActors = sorted(np.unique(indexActors).tolist())
        indexActions = sorted(np.unique(indexActions).tolist())
        indexPair = sorted(np.unique(indexPair).tolist())

        return indexActors, indexActions, indexPair

    def indexactor2string(self, indexlabel):
        #strlabel = []
        strlabel = list(map(lambda item: self.actors_dict[item], indexlabel))
        '''
        for item in indexlabel:
            strlabel.append(self.actors_dict[item])
        '''
        return strlabel

    def indexaction2string(self, indexlabel):
        #strlabel = []
        strlabel = list(map(lambda item: self.actions_dict[item], indexlabel))
        '''
        for item in indexlabel:
            strlabel.append(self.actions_dict[item])
        '''
        return strlabel

    def indexpair2string(self, indexlabel):
        #strlabel = []
        strlabel = list(map(
            lambda item: self.actors_dict[item // 10] + ' ' + self.actions_dict[item % 10], indexlabel))
        '''
        for item in indexlabel:
            strlabel.append(
                self.actors_dict[item // 10] + ' ' + self.actions_dict[item % 10])
        '''
        return strlabel

    def get_split_info(self):
        # [img_path, [t img_path], [actor], [action]]
        ##################
        csv_len = len(self.csv)
        train_data = []
        test_data = []

        for index in range(csv_len):
            item_data = []
            item = self.csv.loc[index]
            img_path = os.path.join(self.args.a2d_root, item['img_id'])
            video_name = item['img_id'].split('/')[1]
            img_frame = int(item['img_id'].split('/')[-1].split('.')[0])
            #t_img_path = []

            if self.args.t == 0:
                t_img_path = list(os.path.join(
                    self.args.a2d_root, 'pngs320H', video_name, str(img_frame).zfill(5) + '.png'))
            else:
                t_img_path = list(map(lambda t: os.path.join(self.args.a2d_root, 'pngs320H', video_name, str(
                    img_frame + t).zfill(5) + '.png'), list(range(-self.args.t, self.args.t + 1))))

            '''
            for i in range(-self.args.t, self.args.t + 1):
                t_img_name = str(img_frame + i).zfill(5)
                t_img_path.append(os.path.join(
                    self.args.a2d_root, 'pngs320H', video_name, t_img_name + '.png'))
            '''

            #actors = []
            #actions = []

            labels = item['label']
            label_list = labels.split(' ')

            actors = list(
                map(lambda label: self.actors_dict[int(float(label)) // 10], label_list))
            actions = list(
                map(lambda label: self.actions_dict[int(float(label)) % 10], label_list))

            '''
            for label in label_list:
                label = int(float(label))
                strActor = self.actors_dict[label // 10]
                strAction = self.actions_dict[label % 10]
                actors.append(strActor)
                actions.append(strAction)
            '''

            item_data.append(img_path)
            item_data.append(t_img_path)
            item_data.append(actors)
            item_data.append(actions)

            if item['usage'] == 0:
                train_data.append(item_data)
            else:
                test_data.append(item_data)
        ##################
        return train_data, test_data

    def train_sample_probability(self):
        tr_labels = self.csv[self.csv['usage'] == 0]['label'].values
        train_sample_num = len(tr_labels)
        maxlenlabel = 0
        for label in tr_labels:
            if len(label.split(' ')) > maxlenlabel:
                maxlenlabel = len(label.split(' '))

        train_label_len_probability = np.zeros(maxlenlabel)
        for label in tr_labels:
            train_label_len_probability[len(label.split(' ')) - 1] += 1
        train_label_len_probability = (
            np.array(train_label_len_probability) / train_sample_num).tolist()

        return maxlenlabel, train_label_len_probability

    def sample_negative(self, actions, actors):
        sample_num = np.random.choice(
            self.maxlenlabel, p=self.train_label_len_probability) + 1

        while(True):
            choice_index_list = np.random.choice(
                len(self.train_pairs), sample_num)
            if len(choice_index_list) == len(set(choice_index_list)):
                break

        pos_pairs = list(map(lambda actor, action: actor +
                             ' '+action, actors, actions))
        samples = list(
            map(lambda index: self.train_pairs[index], choice_index_list))

        if sorted(samples) == sorted(pos_pairs):
            return self.sample_negative(actions, actors)
        else:
            return [self.pair2idx[sample] for sample in samples]

    def idx2hot(self, idx_list, class_num):
        label = to_categorical(idx_list, num_classes=class_num)
        label = label.sum(axis=0)
        #label = np.where(label > 1, 1, label)

        return label

    def __getitem__(self, index):
        img_path, t_img_path, actors, actions = self.data[index]
        # pdb.set_trace()

        if self.args.t == 0:
            #pdb.set_trace()
            img = self.transform(Image.open(img_path).convert('RGB'))

            data = [img, self.idx2hot([self.pair2idx['{} {}'.format(
                actors[i], actions[i])] for i in range(len(actors))], class_num=len(self.pairs))]
            if self.mode == 'train':
                neg_pairs = self.sample_negative(actions, actors)
                data += [self.idx2hot(neg_pairs, class_num=len(self.pairs))]

            data.append(img_path)

            return data

    def __len__(self):
        return len(self.data)
