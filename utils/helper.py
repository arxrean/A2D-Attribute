import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def calculateF1(embed, label):
    # embed (N,d) [R]
    # label (N,d) [0/1]

    F1 = 0.
    ###write###
    bestF1 = 0
    threshold = 0
    bestRecall = 0
    bestPrecision = 0
    for i in range(0, 1, 0.01):
        newEmbed = np.array(embed > i, dtype='float64')
        recall = recall_score(label,newEmbed,average='micro')
        precision = precision_score(label,newEmbed,average='micro')
        f1 = f1_score(label,newEmbed,average='micro')
        if f1 > bestF1:
            bestF1 = f1
            bestPrecision = precision
            bestRecall = recall
            threshold = i

    ###########

    print('F1 score is:{}'.format(F1))


if __name__ == '__main__':
    a = [[0.2,0.1],
         [0.3,0.9],
         [0.5,0.8],
         [0.8,0.1]]
    a = np.array(a, dtype='float64')
    print(a)
    b = np.array(a > 0.5,dtype='float64')
    print(b)
    c = [[0,0],
         [1,0],
         [0,1],
         [1,0]]
    c = np.array(c)
    recall = recall_score(c,b,average='macro')
    pre = precision_score(c,b,average='macro')
    f1 = f1_score(c,b,average='macro')
    print(f1)
    print(2*pre*recall/(pre+recall))
    #calculateF1()