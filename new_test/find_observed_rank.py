import numpy as np
import pickle
import os
from cal_freq import cal_freq

'''

This file finds beta v for u such that with smallest dis[u][v]

'''


dir = ".\\result"


def find_observed_rank(num, K, freq):
    print("Finding observed rank for subsample", num)
    d = dir + "\\" + str(num) + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)

    '''
    f = open(d + "dis_uv.pkl", 'rb')
    dis_uv = pickle.load(f)
    f.close()
    '''
    observed_rank = []
    for u in range(n2):
        rate = data[:, u]
        rate = list(zip(rate, range(len(rate))))
        rate.sort(key=lambda x: x[0], reverse=True)
        rate = [x[1] for x in rate if x[0] != -99]
        res = {}
        for i in range(len(rate)):
            rank_candidate = int((i+1)/freq)
            if(rank_candidate <= K): res[rate[i]] = rank_candidate
            else: res[rate[i]] = K + 1
        observed_rank.append(res)


    f = open(d + "observed_rank.pkl", "wb")

    pickle.dump(observed_rank, f)
    print("Finish Calculating.")
    f.close()