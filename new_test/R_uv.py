import pickle
import numpy as np
import os
'''
This file compute R(u, v) in MR algorithm.

'''




dir = ".\\result"


def R(num):
    print("Finding Ruv for ", num)
    d = dir + "\\" + str(num) + "\\"
    #print(d)
    if not os.path.exists(d):
        os.makedirs(d)
    f = open(d + "Nuv_train.pkl", 'rb')
    N = pickle.load(f)
    f.close()

    f = open(d + "train_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()

    n2 = len(H[0])
    n1 = len(H)

    # print(N)
    def I(u, v):
        N_uv = N[u][v]
        res = []
        for i in range(int(len(N_uv) / 2)):
            res.append((N_uv[2 * i], N_uv[2 * i + 1]))
        return res

    def R(u, v):
        res = 0
        I_uv = I(u, v)
        if (len(I_uv) == 0): return 0
        for (s, t) in I_uv:
            res += ((H[s, u] - H[t, u]) * (H[s, v] - H[t, v]) >= 0)
        return res / (len(I_uv))

    R_mat = np.zeros((n2, n2))
    for i in range(n2):
        for j in range(n2):
            R_mat[i][j] = R(i, j)
        #print(i)

    f = open(d + "Ruv_train.pkl", 'wb')
    pickle.dump(R_mat, f)
    f.close()
    #print(R_mat)






