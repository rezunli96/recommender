import numpy as np
import pickle
import math

import os
from metric import distance1, spearman_rho, kendall_tau

'''
This file calculate Top-K distance among all pairs of (u, v) from ground-truth and stored it in dis[u][v]

'''



dir = ".\\result"

test_num = 1


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
    for u in range(n2):
        # print(u)
        for v in range(n2):
            u_rank = true_rank[u]
            v_rank = true_rank[v]
            dis[u][v] = distance1(u_rank, v_rank, K)


            #print(len(u_rank), len(v_rank))
            '''
            u_rank = true_rank[u]
            u_rank = list(zip(u_rank, range(len(u_rank))))
            u_rank.sort(key=lambda x: x[0], reverse=True)
            u_rank = set([x[1] for x in u_rank][:10])
            v_rank = true_rank[v]
            v_rank = list(zip(v_rank, range(len(v_rank))))
            v_rank.sort(key=lambda x: x[0], reverse=True)
            v_rank = set([x[1] for x in v_rank][:10])
            # print(len(u_rank & v_rank))
            dis[u][v] = len(u_rank & v_rank)
            '''
            # print(u_rank & v_rank)
        #print(dis[u])
    pickle.dump(dis, f)
    f.close()

