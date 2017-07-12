import numpy as np
import os
import pickle



dir = ".\\result"

def cal_freq(H):
    n1 = len(H)
    n2 = len(H[0])
    total = 0
    for i in range(n1):
        for j in range(n2):
            if(H[i, j] != -99): total += 1

    return total/(n1 * n2)



def process_New(num):
    d = dir + "\\" + str(num) + "\\"  # directory to store file
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)


    f = open(d + "train_data.pkl", "rb")

    train_data = pickle.load(f)

    f.close()

    freq = cal_freq(train_data)


    