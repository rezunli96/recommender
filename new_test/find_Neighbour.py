import numpy as np
import pickle
import os


'''

This file finds beta v for u such that with smallest dis[u][v]

'''


dir = ".\\result"


def find_Neighbour(num, beta, k):
    d = dir + "\\" + str(num) + "\\"
    print("Finding possible neighbourhood in new algorithm for subsample", num,"with", (beta,k))
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(d + "train_data.pkl", 'rb')
    data = pickle.load(f)
    f.close()
    n2 = len(data[0])
    n1 = len(data)

    f = open(d + "dis_uv.pkl", 'rb')
    dis = pickle.load(f)
    f.close()

    '''
    f = open(d + "dis_uv.pkl", 'rb')
    dis_uv = pickle.load(f)
    f.close()
    '''

    f = open(d + "Nuv_train.pkl", 'rb')
    N_uv = pickle.load(f)
    f.close()

    N = []
    for u in range(n2):
        dis_u = dis[u]
        dis_u = list(zip(dis_u, range(len(dis_u))))
        dis_u.sort(key=lambda x: x[0])
        res = [x[1] for x in dis_u]
        res = [v for v in res if len(N_uv[u][v]) >= beta][:k]
        N.append(res)

    f = open(d + "neighbour_in_new.pkl", "wb")
    pickle.dump(N, f)
    f.close()
    print("Finishing Calculating.")