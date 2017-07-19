import numpy as np
import pickle
import os

'''

This file store all v for u such that |N(i, j)| >= beta.

'''


dir = ".\\result"


def find_Neighbour_Items_beta(num, beta):
    #print("Find Neighbour Candidate for:", num)
    d = dir + "\\" + str(num) + "\\"
    # print(d)
    if not os.path.exists(d):
        os.makedirs(d)
    f = open(d + "train_data.pkl", 'rb')
    H = pickle.load(f)
    f.close()

    f = open(d + "Nij_train.pkl", 'rb')
    N_ij = pickle.load(f)
    f.close()

    f = open(d + "neighbour_items_beta.pkl", 'wb')
    n2 = len(H[0])
    n1 = len(H)

    #print(n2)


    N_candidate = []


    for i in range(n1):
        #print(u)
        Ni_candidate = []
        for j in range(n1):
            if (i != j):
                if (len(N_ij[i][j]) >= beta):
                    Ni_candidate.append(j)
        #print(len(Nu_candidate))
        N_candidate.append(Ni_candidate)

    pickle.dump(N_candidate, f)
    f.close()
    #print("Finished.")