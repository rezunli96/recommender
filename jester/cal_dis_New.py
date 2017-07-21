import numpy as np
import pickle
import os
from metric import new_distance

'''
This file calculate Top-K distance among all pairs of (u, v) from observed data and stored it in dis[u][v]

'''

dir = ".\\result"


def cal_dis_New(num, K):
    d = dir + "\\" + str(num) + "\\"
    #print(d)
    #print("Calculating distance in New Algorithm for subsample", num)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()

    f = open(d + "observed_rank_new.pkl", "rb")
    observed_rank = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)

    f = open(d + "dis_uv_new.pkl", 'wb')

    dis = np.zeros((n2, n2))

    for u in range(n2):
        # print(u)
        for v in range(n2):
            u_rank_obs = observed_rank[u]
            v_rank_obs = observed_rank[v]
            dis[u][v] = new_distance(u_rank_obs, v_rank_obs, K)

    pickle.dump(dis, f)
    f.close()
    #print("Finish Calculating")

