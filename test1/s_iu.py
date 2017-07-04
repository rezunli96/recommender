import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("train_data.pkl", 'rb')
H = pickle.load(f)
f.close()


f = open("N2ij_train.pkl", 'rb')
N_ij = pickle.load(f)
f.close()


n2 = len(H[0])
n1 = len(H)

f = open("s_iu.pkl", "wb")

beta = 50


for i in range(n1):
    print(i)
    s_i = []
    for u in range(n2):
        s_iu = []
        for j in range(n1):
            if(i != j and len(N_ij[i][j]) >= beta and H[j, u]): s_iu.append(j)
        s_i.append(s_iu)
    pickle.dump(s_i, f)

f.close()