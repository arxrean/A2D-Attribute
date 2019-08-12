import numpy as np
import pandas as pd
from keras.utils import to_categorical
import sys

def joint_frac(embed_size=43):
    res = np.zeros(embed_size)

    valid = {11: 0, 12: 1, 13: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 7, 21: 8, 
             22: 9, 26: 10, 28: 11, 29: 12, 34: 13, 35: 14, 36: 15, 39: 16,
             41: 17, 43: 18, 44: 19, 45: 20, 46: 21, 48: 22, 49: 23, 54: 24,
             55: 25, 56: 26, 57: 27, 59: 28, 61: 29, 63: 30, 65: 31, 66: 32,
             67: 33, 68: 34, 69: 35, 72: 36, 73: 37, 75: 38, 76: 39, 77: 40,
             78: 41, 79: 42}
    csv = pd.read_csv('./repo/newVideoSet.csv')
    num_items = len(csv)

    for i in range(num_items):
        item = csv.loc[i]
        listlabel = item['label'].split(' ')
        labels = []
        for label in listlabel:
            label = int(float(label))
            if label in valid:
                labels.append(valid[label])
            else:
                print('label:%d in csv[%d] is not in valid'%(label, i))
        labels = to_categorical(labels, num_classes=embed_size)
        labels = labels.sum(axis=0)
        res += labels
    print(res)
    N = np.ones(embed_size) * num_items
    print(num_items)
    res = res / (N-res)
    print(res)
    
    np.save('./repo/joint_label_frac.npy', res)

    return res
 

def actor_action_frac(actor_embed_size=7,action_embed_size=9):
    actor_res = np.zeros(actor_embed_size)
    action_res = np.zeros(action_embed_size)

    actor_valid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    action_valid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
    
    csv = pd.read_csv('./repo/newVideoSet.csv')
    num_items = len(csv)

    for i in range(num_items):
        item = csv.loc[i]
        listlabel = item['label'].split(' ')
        actor_label = []
        action_label = []
        for label in listlabel:
            label = int(float(label))
            actor = label // 10
            action = label % 10
            if actor in actor_valid:
                actor_label.append(actor_valid[actor])
            else:
                print('actor_label:%d in csv[%d] is not in actor_valid'%(actor, i))
            if action in action_valid:
                action_label.append(action_valid[action])
            else:
                print('action_label:%d in csv[%d] is not in action_valid'%(action, i))
            
        actor_label = to_categorical(actor_label, num_classes=actor_embed_size)
        actor_label = actor_label.sum(axis=0)
        actor_res += actor_label

        action_label = to_categorical(action_label, num_classes=action_embed_size)
        action_label = action_label.sum(axis=0)
        action_res += action_label
    print('actor_res:', actor_res)
    print('action_res:', action_res)

    N_actor = np.ones(actor_embed_size) * num_items
    actor_res = actor_res / (N_actor - actor_res)

    N_action = np.ones(action_embed_size) * num_items
    action_res = action_res / (N_action - action_res)
    print('actor_res:', actor_res)
    print('action_res:', action_res)

    np.save('./repo/actor_label_frac.npy', actor_res)
    np.save('./repo/action_label_frac.npy', action_res)

    return actor_res, action_res

if __name__ == '__main__':
    res = joint_frac()
    actor_res, action_res = actor_action_frac()
