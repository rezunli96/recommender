import numpy as np
import pickle
import os

dir = ".\\result"


def cal_freq(num):
    d = dir + "\\" + str(num) + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()
    n1 = len(H)
    n2 = len(H[0])
    total = 0
    for i in range(n1):
        for j in range(n2):
            if(H[i, j] != -99): total += 1

    return total/(n1 * n2)