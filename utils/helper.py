import numpy as np

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
    best_recall = None
    Threshold = None
    for thd in np.arange(0, 1, 0.01):
        X_pre_new = np.array(X_pre > thd, dtype='float64')
        f1 = F1(X_pre_new, X_gt)
        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_prec = Precision(X_pre_new, X_gt)
            best_recall = Recall(X_pre_new, X_gt)
            Threshold = thd

    print('best_f1:{}'.format(best_f1))
    print('best_prec:{}'.format(best_prec))
    print('best_recall:{}'.format(best_recall))
    print('Threshold:{}'.format(Threshold))
    return best_f1, best_prec, best_recall, Threshold


if __name__ == '__main__':
    pass