import numpy as np
import pickle
import math
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import os
from metric import distance1


dir = ".\\result"

test_num = 100


K = 30

def cal_dis(num):
    d = dir + "\\" + str(num) + "\\"
    print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "complete_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()
    f = open(d + "true_rank.pkl", "rb")
    true_rank = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)
    f = open(d + "dis_uv.pkl", 'wb')
    N = []

    dis = np.zeros((n2, n2))
    for u in range(n2):
        for v in range(n2):
            d = 0
            complete_u = true_rank[u]
            complete_v = true_rank[v]
            dis[u][v] = distance1(complete_u, complete_v, K)
    pickle.dump(dis, f)
    f.close()

for i in range(test_num):
    cal_dis(i)
