import numpy as np
import pickle

import os




'''

This file compute N(u, v) in MR algorithm, i.e., all item commonly rated by u and v in training set.

'''


dir = ".\\result"


def N(num):
    d = dir + "\\" + str(num) + "\\"
    print("Finding Commonly Rated Item for subsample ", num)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    train = pickle.load(f)
    f.close()
    n2 = len(train[0])
    n1 = len(train)
    f = open(d + "Nuv_train.pkl", 'wb')
    N = []

    for u in range(n2):
        #print(u)
        N_u = []
        for v in range(n2):
            N_uv = []
            for l in range(n1):
                if (train[l, u] != -99 and train[l, v] != -99):  N_uv.append(l)
            N_u.append(N_uv)
            #print(len(N_uv))
        N.append(N_u)

    pickle.dump(N, f)
    f.close()



