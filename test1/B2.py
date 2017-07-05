import numpy as np
import pickle
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

f = open("sparse_train_data.pkl", 'rb')
S = pickle.load(f)
f.close()

f = open("train_data.pkl", 'rb')
H = pickle.load(f)
f.close()

n2 = 1390
n1 = 1000

f = open("Nuv_train.pkl", 'rb')
N_uv = pickle.load(f)
f.close()

f = open("N2ij_train.pkl", 'rb')
N_ij = pickle.load(f)
f.close()

beta = 10


f = open("B.pkl", "wb")
B = []

for u in range(n2):
    print(u)
    B_ui = []
    for i in range(n1):
        print(u, i)
        B_i = []
        for (v, j) in S:
            if(u != v and len(N_uv[u][v]) >= beta and  H[i, v]):
                if(j != i and len(N_ij[i][j]) >= beta and H[j, u]): B_i.append((v, j))
        B_ui.append(B_i)
    B.append(B_ui)

pickle.dump(B, f)
f.close()