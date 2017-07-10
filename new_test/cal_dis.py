import numpy as np
import pickle
import math

import os
from metric import distance1

'''
This file calculate Top-K distance among all pairs of (u, v) from ground-truth and stored it in dis[u][v]

'''



dir = ".\\result"

test_num = 100


K = 10 # K in Top-K

def cal_dis(num):
    d = dir + "\\" + str(num) + "\\"
    print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "sampled_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()
    f = open(d + "true_rank.pkl", "rb")
    true_rank = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)
    f = open(d + "dis_uv.pkl", 'wb')

    dis = np.zeros((n2, n2))
    m = 1000
    for u in range(n2):
        for v in range(n2):
            complete_u = true_rank[u]
            complete_v = true_rank[v]
            dis[u][v] = distance1(complete_u, complete_v, K)
    pickle.dump(dis, f)
    f.close()

for i in range(test_num):
    cal_dis(i)
