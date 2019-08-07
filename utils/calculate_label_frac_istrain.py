import numpy as np
import pandas as pd
from keras.utils import to_categorical

def joint_frac_istrain(embed_size=43):
    res = np.zeros(embed_size)

    valid = {11: 0, 12: 1, 13: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 7, 21: 8, 
             22: 9, 26: 10, 28: 11, 29: 12, 34: 13, 35: 14, 36: 15, 39: 16,
             41: 17, 43: 18, 44: 19, 45: 20, 46: 21, 48: 22, 49: 23, 54: 24,
             55: 25, 56: 26, 57: 27, 59: 28, 61: 29, 63: 30, 65: 31, 66: 32,
             67: 33, 68: 34, 69: 35, 72: 36, 73: 37, 75: 38, 76: 39, 77: 40,
             78: 41, 79: 42}
    csv = pd.read_csv('./repo/newVideoSet.csv')
    num_items = len(csv)

    print(1)
    #print(type(csv.loc[0]['is_train']))

    for i in range(num_items):
        item = csv.loc[i]
        if item['is_train'] == 0:
            csv.drop([i],inplace=True) 
    
    num_items = len(csv)
    print(num_items)

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
    res = res / (N-res)
    print(res)

    #np.save('./repo/joint_label_frac_istrain.npy', res)

    return res
    
if __name__ == '__main__':
    #res = joint_frac_istrain()
    joint_frac_istrain()
