import numpy as np
import pickle
import os


'''

This file finds beta v for u such that with smallest dis[u][v]

'''


dir = ".\\result"





K = 10

def find_N(num, beta):
    d = dir + "\\" + str(num) + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "sampled_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)
    f = open(d + "dis_uv.pkl", 'rb')
    dis_uv = pickle.load(f)
    f.close()

    N = []
    for u in range(n2):
        dis_u = dis_uv[u]
        D = list(zip(dis_u, range(len(dis_u))))
        # D.sort(key=lambda x: x[0])
        D.sort(key=lambda x: x[0])
        N.append([x[1] for x in D][:beta])
    f = open(d + "N_u.pkl", "wb")
    pickle.dump(N, f)
    f.close()

