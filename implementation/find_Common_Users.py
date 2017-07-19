import numpy as np
import pickle

import os


'''

This file compute N(i, j). i.e., all users commonly rating i and j in training set.

'''


dir = ".\\result"


def find_Common_Users(num):
    d = dir + "\\" + str(num) + "\\"
    #print("Finding Commonly Rated Item in train set for subsample ", num)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    train = pickle.load(f)
    f.close()
    n2 = len(train[0])
    n1 = len(train)
    f = open(d + "Nij_train.pkl", 'wb')
    N = []

    for i in range(n1):
        #print(u)
        N_i = []
        for j in range(n1):
            N_ij = []
            for u in range(n2):
                if (train[i, u] != -99 and train[j, u] != -99):  N_ij.append(u)
            N_i.append(N_ij)
            #print(len(N_uv))
        N.append(N_i)
    #print("Finished.")
    pickle.dump(N, f)
    f.close()