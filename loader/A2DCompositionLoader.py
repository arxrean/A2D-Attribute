import torch
import torch.utils.data as tdata
import numpy as np
import pandas as pd
import os
import pdb
from PIL import Image


class A2DComposition(tdata.Dataset):
    def __init__(self, args, transform=None, mode='train'):
        self.args = args
        self.transform = transform
        self.mode = mode
        self.csv = pd.read_csv(args.csv_path)

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
            candidates = [action for (
                _, _, action, actor) in self.train_data+self.test_data if actor == _actor]
            self.actor_affordance[_actor] = list(set(candidates))

            candidates = [actor for (_, _, action, actor)
                          in self.train_data if actor == _actor]
            self.train_actor_affordance[_actor] = list(set(candidates))

        feat_file = args.feature_path  # pre-joint features path
        activation_data = np.load(feat_file, allow_pickle=True).tolist()
        # activation_data = torch.load(feat_file) # [features,names] features=>(N,d)
        self.activations = dict(
            zip(activation_data['id'], activation_data['res']))
        self.feat_dim = len(activation_data['res'][0])  # d

    def parse_split(self):
        ##################
        tr_actions = []
        tr_actors = []
        tr_pairs = []

        ts_actions = []
        ts_actors = []
        ts_pairs = []

        tr_labels = self.csv[self.csv['usage'] == 0]['label'].values
        ts_labels = self.csv[self.csv['usage'] == 1]['label'].values

        tr_actors, tr_actions, tr_pairs = self.getstringlabel(tr_labels)
        ts_actors, ts_actions, ts_pairs = self.getstringlabel(ts_labels)
        ##################

        all_actions, all_actors = sorted(
            list(set(tr_actions+ts_actions))), sorted(list(set(tr_actors+ts_actors)))
        all_pairs = sorted(list(set(tr_pairs + ts_pairs)))

        return all_actions, all_actors, all_pairs, tr_pairs, ts_pairs

    def getstringlabel(self, labels):
        pairs = []
        actors = []
        actions = []

        for item in labels:
            label_list = item.split(' ')
            for label in label_list:
                label = int(float(label))
                strActor = self.actors_dict[label // 10]
                strAction = self.actions_dict[label % 10]
                actors.append(strActor)
                actions.append(strAction)
                pairs.append(strActor + ' ' + strAction)
        actors = np.unique(actors).tolist()
        actions = np.unique(actions).tolist()
        pairs = np.unique(pairs).tolist()

        return actors, actions, pairs

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
            t_img_path = []

            for i in range(-self.args.t, self.args.t + 1):
                t_img_name = str(img_frame + i).zfill(5)
                t_img_path.append(os.path.join(
                    self.args.a2d_root, 'pngs320H', video_name, t_img_name + '.png'))

            actors = []
            actions = []

            labels = item['label']
            label_list = labels.split(' ')
            for label in label_list:
                label = int(float(label))
                strActor = self.actors_dict[label // 10]
                strAction = self.actions_dict[label % 10]
                actors.append(strActor)
                actions.append(strAction)

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

    def sample_negative(self, actions, actors):
        sample = self.train_pairs[np.random.choice(
            len(self.train_pairs))]
        new_actor, new_action = sample.split(' ')[0], sample.split(' ')[1]
        if new_action in actions and new_actor in actors:
            return self.sample_negative(actions, actors)
        return self.action2idx[new_action], self.actor2idx[new_actor]

    def __getitem__(self, index):
        img_path, t_img_path, actors, actions = self.data[index]

        if self.args.t == 0:
            img = self.transform(Image.open(img_path).convert('RGB'))
            data = [img, [self.action2idx[action] for action in actions], [self.actor2idx[actor]
                                                                           for actor in actors], [self.pair2idx['{} {}'.format(actors[i], actions[i])] for i in range(len(actors))]]
            pdb.set_trace()
            if self.mode == 'train':
                neg_action, neg_actor = self.sample_negative(actions, actors)
                data += [neg_action, neg_actor]

            if self.activations is not None:
                data[0] = torch.from_numpy(self.activations[img_path[img_path.index('pngs320H'):]])

            return data

    def __len__(self):
        return len(self.data)
