import numpy as np


def joint_frac(embed_size=43):
    res = np.zeros(43)

    ###
    # write
    ###
    
    np.save('../repo/joint_label_frac.npy', res)


if __name__ == '__main__':
    joint_frac()
