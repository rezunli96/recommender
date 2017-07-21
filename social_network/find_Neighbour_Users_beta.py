import numpy as np
import pickle
import os

'''

This file store all v for u such that |N(u, v)| >= beta.

'''


dir = ".\\result"


def find_Neighbour_Users_beta(num, beta):
    #print("Find Neighbour Candidate for:", num)
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

    f = open(d + "neighbour_users_beta.pkl", 'wb')
    n2 = len(H[0])
    n1 = len(H)

    #print(n2)


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
    #print("Finished.")

