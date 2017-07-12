import numpy as np
import pickle
import os

'''
This file store all v for u such that |N(u, v)| >= beta in MR algorithm. The result is stored in a 

list N_candidate, with N_candidate[u] stores all such neighbourhood for u.


'''


dir = ".\\result"


def N_C(num):
    print("Find Neighbour Candidate for:", num)
    d = dir + "\\" + str(num) + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)
    f = open(d + "train_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()

    f = open(d + "Nuv_train.pkl", 'rb')
    N_uv = pickle.load(f)
    f.close()

    f = open(d + "Nuv_candidate.pkl", 'wb')
    n2 = len(H[0])
    n1 = len(H)

    #print(n2)

    beta = 5

    N_candidate = []


    for u in range(n2):
        #print(u)
        Nu_candidate = []
        for v in range(n2):
            if (u != v):
                if (len(N_uv[u][v]) >= beta):
                    Nu_candidate.append(v)
        #print(len(Nu_candidate))
        N_candidate.append(Nu_candidate)

    pickle.dump(N_candidate, f)
    f.close()

