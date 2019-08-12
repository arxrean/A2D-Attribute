import numpy as np
from heapq import nlargest
# data format
# The dimension of X_pre and X_gt are NXnum_cls, whether N is the number of samples and num_cls is the number of classes

def Precision(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x*y)/(np.sum(x) + 1e-8)
    return p/N


def Recall(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(y)
    return p/N


def F1(X_pre, X_gt):
    N = len(X_pre)
    p = 0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += 2*np.sum(x * y) / (np.sum(x) + np.sum(y))
    return p/N


def get_eval(X_pre, X_gt):
    best_f1 = None
    best_prec = None
    best_recall= None
    caltype = None

    best_f1_threshold = None
    best_prec_threshold = None
    best_recall_threshold = None
    Threshold = None

    best_f1_maxnum = None
    best_prec_maxnum = None
    best_recall_maxnum = None
    maxnum = None

    for thd in np.arange(0, 1, 0.01):
        X_pre_new = np.array(X_pre > thd, dtype='float64')
        f1 = F1(X_pre_new, X_gt)
        if best_f1_threshold is None or f1 > best_f1_threshold:
            best_f1_threshold = f1
            best_prec_threshold = Precision(X_pre_new, X_gt)
            best_recall_threshold = Recall(X_pre_new, X_gt)
            Threshold = thd

    for num in range(5):
        for row in range(len(X_pre)):
            largest = nlargest(num+1, X_pre[row, :])
            X_pre_new = X_pre
            X_pre_new[row, :] = np.where(X_pre[row, :] >= min(largest), 1, 0)
        f1 = F1(X_pre_new, X_gt)
        if best_f1_maxnum is None or f1 > best_f1_maxnum:
            best_f1_maxnum = f1
            best_prec_maxnum= Precision(X_pre_new, X_gt)
            best_recall_maxnum = Recall(X_pre_new, X_gt)
            maxnum = num+1

    if(best_f1_maxnum > best_f1_threshold):
        best_f1 = best_f1_maxnum
        best_prec = best_prec_maxnum
        best_recall= best_recall_maxnum
        caltype = 'maxnum' 
    else:
        best_f1 = best_f1_threshold
        best_prec = best_prec_threshold
        best_recall= best_recall_threshold
        caltype = 'threshold'    
    
    print('best_f1:{}'.format(best_f1))
    print('best_prec:{}'.format(best_prec))
    print('best_recall:{}'.format(best_recall))
    print('caltype:{}'.format(caltype))
    if caltype == 'threshold':
        print('Threshold:{}'.format(Threshold))
        return best_f1, best_prec, best_recall, caltype, Threshold
    else:
        print('maxnum:{}'.format(maxnum))
        return best_f1, best_prec, best_recall, caltype, maxnum


if __name__ == '__main__':  
    pass  